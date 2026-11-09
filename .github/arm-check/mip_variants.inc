// #879 review variants of the NEON MIP matrix multiply (included by bench_mip_variants.cpp after the tree's kernels,
// which are extracted into namespace ld4). uzp: vld4q_u8 replaced by four plain loads and a two-level vuzpq_u8 tree.
// mull: no de-interleaving; each output's weights times the inputs with vmull/vmlal, then pairwise reductions.

namespace uzp
{
// as vld4q_u8( w ): val[k] lane n = w[4 n + k]
static inline uint8x16x4_t mipLoadTaps4_neon( const uint8_t* w )
{
  const uint8x16_t v0 = vld1q_u8( w );
  const uint8x16_t v1 = vld1q_u8( w + 16 );
  const uint8x16_t v2 = vld1q_u8( w + 32 );
  const uint8x16_t v3 = vld1q_u8( w + 48 );
  // taps 0 and 2, 1 and 3 alternating, of outputs 0..7 (a) and 8..15 (b)
  const uint8x16x2_t a = vuzpq_u8( v0, v1 );
  const uint8x16x2_t b = vuzpq_u8( v2, v3 );
  const uint8x16x2_t even = vuzpq_u8( a.val[0], b.val[0] );
  const uint8x16x2_t odd  = vuzpq_u8( a.val[1], b.val[1] );
  uint8x16x4_t r;
  r.val[0] = even.val[0];
  r.val[1] = odd.val[0];
  r.val[2] = even.val[1];
  r.val[3] = odd.val[1];
  return r;
}
#pragma push_macro( "vld4q_u8" )
#undef vld4q_u8
#define vld4q_u8 mipLoadTaps4_neon
#include "mip_kernel.inc"
#undef vld4q_u8
#pragma pop_macro( "vld4q_u8" )
} // namespace uzp

namespace mull
{
using ld4::mipOffset_neon;
using ld4::mipClip_neon;

// four outputs' sums: the dot products of 8 inputs and the 8 weights of outputs n .. n + 3 (32 bytes)
static inline int32x4_t mipDots4x8_neon( const uint8_t* w, const int16x8_t in )
{
  const uint8x16_t w01 = vld1q_u8( w );
  const uint8x16_t w23 = vld1q_u8( w + 16 );
  const int16x8_t  w0  = vreinterpretq_s16_u16( vmovl_u8( vget_low_u8( w01 ) ) );
  const int16x8_t  w1  = vreinterpretq_s16_u16( vmovl_u8( vget_high_u8( w01 ) ) );
  const int16x8_t  w2  = vreinterpretq_s16_u16( vmovl_u8( vget_low_u8( w23 ) ) );
  const int16x8_t  w3  = vreinterpretq_s16_u16( vmovl_u8( vget_high_u8( w23 ) ) );
  int32x4_t d0 = vmull_s16( vget_low_s16( w0 ), vget_low_s16( in ) );
  int32x4_t d1 = vmull_s16( vget_low_s16( w1 ), vget_low_s16( in ) );
  int32x4_t d2 = vmull_s16( vget_low_s16( w2 ), vget_low_s16( in ) );
  int32x4_t d3 = vmull_s16( vget_low_s16( w3 ), vget_low_s16( in ) );
  d0 = vmlal_s16( d0, vget_high_s16( w0 ), vget_high_s16( in ) );
  d1 = vmlal_s16( d1, vget_high_s16( w1 ), vget_high_s16( in ) );
  d2 = vmlal_s16( d2, vget_high_s16( w2 ), vget_high_s16( in ) );
  d3 = vmlal_s16( d3, vget_high_s16( w3 ), vget_high_s16( in ) );
  return horizontal_add_4d_s32x4( d0, d1, d2, d3 );
}

// four outputs' sums for 4 inputs (16 bytes of weights); in4 holds the 4 inputs twice
static inline int32x4_t mipDots4x4_neon( const uint8_t* w, const int16x8_t in4 )
{
  const uint8x16_t w03 = vld1q_u8( w );
  const int16x8_t  w01 = vreinterpretq_s16_u16( vmovl_u8( vget_low_u8( w03 ) ) );
  const int16x8_t  w23 = vreinterpretq_s16_u16( vmovl_u8( vget_high_u8( w03 ) ) );
  const int32x4_t  d0  = vmull_s16( vget_low_s16( w01 ), vget_low_s16( in4 ) );
  const int32x4_t  d1  = vmull_s16( vget_high_s16( w01 ), vget_high_s16( in4 ) );
  const int32x4_t  d2  = vmull_s16( vget_low_s16( w23 ), vget_low_s16( in4 ) );
  const int32x4_t  d3  = vmull_s16( vget_high_s16( w23 ), vget_high_s16( in4 ) );
  return horizontal_add_4d_s32x4( d0, d1, d2, d3 );
}

static inline void mip16Outputs8Inputs_neon( const uint8_t* weight, const Pel* input, const int32x4_t offset,
                                             const int16x8_t vmax, int16x8_t* rows )
{
  const int16x8_t in = vld1q_s16( input );
  int32x4_t       acc[4];
  for( int i = 0; i < 4; i++ )
  {
    acc[i] = vaddq_s32( offset, mipDots4x8_neon( weight + 32 * i, in ) );
  }
  rows[0] = mipClip_neon( acc[0], acc[1], vmax );
  rows[1] = mipClip_neon( acc[2], acc[3], vmax );
}

static void mipMatrixMul_4_4_neon( Pel* res, const Pel* input, const uint8_t* weight, const int maxVal,
                                   const int inputOffset, bool transpose )
{
  const int32x4_t offset = mipOffset_neon( input, 4, inputOffset );
  const int16x4_t in     = vld1_s16( input );
  const int16x8_t in4    = vcombine_s16( in, in );
  int32x4_t       acc[4];
  for( int i = 0; i < 4; i++ )
  {
    acc[i] = vaddq_s32( offset, mipDots4x4_neon( weight + 16 * i, in4 ) );
  }
  const int16x8_t vmax    = vdupq_n_s16( maxVal );
  int16x8_t       rows[2] = { mipClip_neon( acc[0], acc[1], vmax ), mipClip_neon( acc[2], acc[3], vmax ) };
  if( transpose )
  {
    transpose_4x4_s16_paired( rows );
  }
  vst1q_s16( res, rows[0] );
  vst1q_s16( res + 8, rows[1] );
}

static void mipMatrixMul_8_4_neon( Pel* res, const Pel* input, const uint8_t* weight, const int maxVal,
                                   const int inputOffset, bool transpose )
{
  int16x8_t rows[2];
  mip16Outputs8Inputs_neon( weight, input, mipOffset_neon( input, 8, inputOffset ), vdupq_n_s16( maxVal ), rows );
  if( transpose )
  {
    transpose_4x4_s16_paired( rows );
  }
  vst1q_s16( res, rows[0] );
  vst1q_s16( res + 8, rows[1] );
}

static void mipMatrixMul_8_8_neon( Pel* res, const Pel* input, const uint8_t* weight, const int maxVal,
                                   const int inputOffset, bool transpose )
{
  const int32x4_t offset = mipOffset_neon( input, 8, inputOffset );
  const int16x8_t vmax   = vdupq_n_s16( maxVal );
  int16x8_t       rows[8];
  for( int i = 0; i < 4; i++ )
  {
    mip16Outputs8Inputs_neon( weight + 128 * i, input, offset, vmax, rows + 2 * i );
  }
  if( transpose )
  {
    transpose_8x8_s16( rows );
  }
  for( int i = 0; i < 8; i++ )
  {
    vst1q_s16( res + 8 * i, rows[i] );
  }
}
} // namespace mull
