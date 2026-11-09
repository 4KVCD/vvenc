// N8 (#909) audit and N27 feasibility on Neoverse N2: MCTF integer motion error kernels in an emulated integer search
// (raster over +-range, running best, only error < best replaces it) on two synthetic 1080p 10-bit pictures, the second
// the first moved by (3, -2) plus noise. Kernels pasted from the trees (see make_bench_mctf.py). The chosen vector and
// error of every block must equal master's; best of 5 search times.
#include <arm_neon.h>
#include <arm_sve.h>
#include <arm_neon_sve_bridge.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

typedef int16_t Pel;
#define CHECKD( c, m )
static inline int32_t horizontal_add_s32x4( int32x4_t v ) { return vaddvq_s32( v ); }
static inline int64x2_t vvenc_sdotq_s16( int64x2_t acc, int16x8_t x, int16x8_t y )
{
  return svget_neonq_s64( svdot_s64( svset_neonq_s64( svundef_s64(), acc ), svset_neonq_s16( svundef_s16(), x ),
                                     svset_neonq_s16( svundef_s16(), y ) ) );
}

static int masterNeon( const Pel* org, const ptrdiff_t origStride, const Pel* buf,
                                    const ptrdiff_t buffStride, const int w, int h, const int besterror )
{
  CHECKD( w % 8 != 0, "Width must be a multiple of eight" );
  CHECKD( h % 2 != 0, "Height must be a multiple of two" );

  int error = 0;
  do
  {
    int32x4_t acc_lo = vdupq_n_s32( 0 );
    int32x4_t acc_hi = vdupq_n_s32( 0 );

    int x1 = 0;
    do
    {
      int16x8_t o1 = vld1q_s16( org + x1 );
      int16x8_t b1 = vld1q_s16( buf + x1 );
      int16x8_t o2 = vld1q_s16( org + origStride + x1 );
      int16x8_t b2 = vld1q_s16( buf + buffStride + x1 );

      int16x8_t diff1 = vabdq_s16( o1, b1 );
      int16x8_t diff2 = vabdq_s16( o2, b2 );
      acc_lo = vmlal_s16( acc_lo, vget_low_s16( diff1 ), vget_low_s16( diff1 ) );
      acc_hi = vmlal_s16( acc_hi, vget_high_s16( diff1 ), vget_high_s16( diff1 ) );

      acc_lo = vmlal_s16( acc_lo, vget_low_s16( diff2 ), vget_low_s16( diff2 ) );
      acc_hi = vmlal_s16( acc_hi, vget_high_s16( diff2 ), vget_high_s16( diff2 ) );

      x1 += 8;
    } while( x1 != w );

    int32x4_t diff2_sum = vaddq_s32( acc_lo, acc_hi );
    error += horizontal_add_s32x4( diff2_sum );
    if( error > besterror )
    {
      return error;
    }

    org += 2 * origStride;
    buf += 2 * buffStride;
    h -= 2;
  } while( h != 0 );

  return error;
}

static inline void addRowPairSquaredDiffs_neon( const Pel* org, const ptrdiff_t origStride, const Pel* buf,
                                                const ptrdiff_t buffStride, const int w, int32x4_t& accLo,
                                                int32x4_t& accHi )
{
  int x1 = 0;
  do
  {
    int16x8_t o1 = vld1q_s16( org + x1 );
    int16x8_t b1 = vld1q_s16( buf + x1 );
    int16x8_t o2 = vld1q_s16( org + origStride + x1 );
    int16x8_t b2 = vld1q_s16( buf + buffStride + x1 );

    int16x8_t diff1 = vabdq_s16( o1, b1 );
    int16x8_t diff2 = vabdq_s16( o2, b2 );
    accLo = vmlal_s16( accLo, vget_low_s16( diff1 ), vget_low_s16( diff1 ) );
    accHi = vmlal_s16( accHi, vget_high_s16( diff1 ), vget_high_s16( diff1 ) );

    accLo = vmlal_s16( accLo, vget_low_s16( diff2 ), vget_low_s16( diff2 ) );
    accHi = vmlal_s16( accHi, vget_high_s16( diff2 ), vget_high_s16( diff2 ) );

    x1 += 8;
  } while( x1 != w );
}

static int n8Neon( const Pel* org, const ptrdiff_t origStride, const Pel* buf,
                                    const ptrdiff_t buffStride, const int w, int h, const int besterror )
{
  CHECKD( w % 8 != 0, "Width must be a multiple of eight" );
  CHECKD( h % 2 != 0, "Height must be a multiple of two" );
  CHECKD( w > 64, "Width must be at most 64" );

  // Like every MCTF error kernel, this returns the exact error when it is not above besterror and a value above
  // besterror otherwise: the callers only test error < besterror. Blocks of up to 16 rows are summed in full with one
  // reduction at the end, since the early exit after a row pair rarely saves enough rows there to pay for the reduction
  // and the branch after every pair. The sum cannot overflow: at most 64 x 16 samples of at most 10 bits. Taller blocks
  // check the running error after every row pair.
  if( h <= 16 )
  {
    int32x4_t accLo = vdupq_n_s32( 0 );
    int32x4_t accHi = vdupq_n_s32( 0 );
    do
    {
      addRowPairSquaredDiffs_neon( org, origStride, buf, buffStride, w, accLo, accHi );
      org += 2 * origStride;
      buf += 2 * buffStride;
      h -= 2;
    } while( h != 0 );
    return horizontal_add_s32x4( vaddq_s32( accLo, accHi ) );
  }

  int error = 0;
  do
  {
    int32x4_t accLo = vdupq_n_s32( 0 );
    int32x4_t accHi = vdupq_n_s32( 0 );
    addRowPairSquaredDiffs_neon( org, origStride, buf, buffStride, w, accLo, accHi );

    error += horizontal_add_s32x4( vaddq_s32( accLo, accHi ) );
    if( error > besterror )
    {
      return error;
    }

    org += 2 * origStride;
    buf += 2 * buffStride;
    h -= 2;
  } while( h != 0 );

  return error;
}

static int masterSve( const Pel* org, const ptrdiff_t origStride, const Pel* buf, const ptrdiff_t buffStride,
                            const int w, int h, const int besterror )
{
  CHECKD( w % 8 != 0, "Width must be a multiple of eight" );
  CHECKD( h % 4 != 0, "Height must be a multiple of four" );

  int error = 0;
  do
  {
    int64x2_t acc1 = vdupq_n_s64( 0 );
    int64x2_t acc2 = vdupq_n_s64( 0 );

    int x1 = 0;
    do
    {
      int16x8_t o1 = vld1q_s16( org + 0 * origStride + x1 );
      int16x8_t b1 = vld1q_s16( buf + 0 * origStride + x1 );
      int16x8_t o2 = vld1q_s16( org + 1 * origStride + x1 );
      int16x8_t b2 = vld1q_s16( buf + 1 * buffStride + x1 );
      int16x8_t o3 = vld1q_s16( org + 2 * origStride + x1 );
      int16x8_t b3 = vld1q_s16( buf + 2 * buffStride + x1 );
      int16x8_t o4 = vld1q_s16( org + 3 * origStride + x1 );
      int16x8_t b4 = vld1q_s16( buf + 3 * buffStride + x1 );

      int16x8_t diff1 = vabdq_s16( o1, b1 );
      int16x8_t diff2 = vabdq_s16( o2, b2 );
      int16x8_t diff3 = vabdq_s16( o3, b3 );
      int16x8_t diff4 = vabdq_s16( o4, b4 );

      acc1 = vvenc_sdotq_s16( acc1, diff1, diff1 );
      acc2 = vvenc_sdotq_s16( acc2, diff2, diff2 );
      acc1 = vvenc_sdotq_s16( acc1, diff3, diff3 );
      acc2 = vvenc_sdotq_s16( acc2, diff4, diff4 );

      x1 += 8;
    } while( x1 != w );

    int64x2_t diff2_sum = vaddq_s64( acc1, acc2 );
    error += ( int32_t )vaddvq_s64( diff2_sum );
    if( error > besterror )
    {
      return error;
    }

    org += 4 * origStride;
    buf += 4 * buffStride;
    h -= 4;
  } while( h != 0 );

  return error;
}

// N27 prototype: master SVE's 4-row step, but blocks of up to 16 rows are summed in full with one reduction at the end.
static inline void addFourRowsSquaredDiffs_sve( const Pel* org, const ptrdiff_t origStride, const Pel* buf,
                                                const ptrdiff_t buffStride, const int w, int64x2_t& acc1, int64x2_t& acc2 )
{
  int x1 = 0;
  do
  {
    int16x8_t diff1 = vabdq_s16( vld1q_s16( org + 0 * origStride + x1 ), vld1q_s16( buf + 0 * buffStride + x1 ) );
    int16x8_t diff2 = vabdq_s16( vld1q_s16( org + 1 * origStride + x1 ), vld1q_s16( buf + 1 * buffStride + x1 ) );
    int16x8_t diff3 = vabdq_s16( vld1q_s16( org + 2 * origStride + x1 ), vld1q_s16( buf + 2 * buffStride + x1 ) );
    int16x8_t diff4 = vabdq_s16( vld1q_s16( org + 3 * origStride + x1 ), vld1q_s16( buf + 3 * buffStride + x1 ) );
    acc1 = vvenc_sdotq_s16( acc1, diff1, diff1 );
    acc2 = vvenc_sdotq_s16( acc2, diff2, diff2 );
    acc1 = vvenc_sdotq_s16( acc1, diff3, diff3 );
    acc2 = vvenc_sdotq_s16( acc2, diff4, diff4 );
    x1 += 8;
  } while( x1 != w );
}

static int n27Sve( const Pel* org, const ptrdiff_t origStride, const Pel* buf, const ptrdiff_t buffStride, const int w,
                   int h, const int besterror )
{
  if( h <= 16 )
  {
    int64x2_t acc1 = vdupq_n_s64( 0 ), acc2 = vdupq_n_s64( 0 );
    do
    {
      addFourRowsSquaredDiffs_sve( org, origStride, buf, buffStride, w, acc1, acc2 );
      org += 4 * origStride;
      buf += 4 * buffStride;
      h -= 4;
    } while( h != 0 );
    return ( int32_t )vaddvq_s64( vaddq_s64( acc1, acc2 ) );
  }
  int error = 0;
  do
  {
    int64x2_t acc1 = vdupq_n_s64( 0 ), acc2 = vdupq_n_s64( 0 );
    addFourRowsSquaredDiffs_sve( org, origStride, buf, buffStride, w, acc1, acc2 );
    error += ( int32_t )vaddvq_s64( vaddq_s64( acc1, acc2 ) );
    if( error > besterror ) return error;
    org += 4 * origStride;
    buf += 4 * buffStride;
    h -= 4;
  } while( h != 0 );
  return error;
}

typedef int ( *Fn )( const Pel*, ptrdiff_t, const Pel*, ptrdiff_t, int, int, int );

int main()
{
  const int W = 1920, H = 1080, M = 80, S = W + 2 * M;
  std::mt19937 rng( 8 );
  std::vector<Pel> pic[2];
  auto field = [&]( double x, double y )
  {
    return 512 + 250 * std::sin( x / 41.0 ) * std::cos( y / 29.0 ) + 90 * std::sin( ( x - 2 * y ) / 7.3 ) + 40 * std::cos( ( 3 * x + y ) / 3.1 );
  };
  for( int p = 0; p < 2; p++ )
  {
    pic[p].assign( S * ( H + 2 * M ), 0 );
    for( int y = -M; y < H + M; y++ )
      for( int x = -M; x < W + M; x++ )
      {
        const double v = p == 0 ? field( x, y ) : field( x + 3, y - 2 ) + 6;
        pic[p][( y + M ) * S + x + M] = Pel( std::min( 1023, std::max( 0, int( v ) + int( rng() % 13 ) - 6 ) ) );
      }
  }
  const Pel* org = pic[1].data() + M * S + M;
  const Pel* ref = pic[0].data() + M * S + M;
  // read through a volatile array so the kernels are called through a pointer, as the encoder calls them
  static Fn volatile fns[4] = { masterNeon, n8Neon, masterSve, n27Sve };
  const char* names[4] = { "master NEON", "#909 NEON", "master SVE", "N27 SVE" };
  int ok = 1;
  for( int bs : { 8, 16, 32 } )
  {
    const int range = bs == 32 ? 8 : 6;
    std::vector<int> bestErr[4], bestV[4];
    double best[4] = { 1e30, 1e30, 1e30, 1e30 };
    for( int round = 0; round < 5; round++ )
      for( int v = 0; v < 4; v++ )
      {
        bestErr[v].clear();
        bestV[v].clear();
        const Fn fn = fns[v];
        auto t0 = std::chrono::steady_clock::now();
        for( int by = 0; by + bs <= H; by += bs )
          for( int bx = 0; bx + bs <= W; bx += bs )
          {
            int be = 0x7fffffff, bv = 0;
            for( int dy = -range; dy <= range; dy++ )
              for( int dx = -range; dx <= range; dx++ )
              {
                const int e = fn( org + by * S + bx, S, ref + ( by + dy ) * S + bx + dx, S, bs, bs, be );
                if( e < be ) { be = e; bv = dy * 64 + dx; }
              }
            bestErr[v].push_back( be );
            bestV[v].push_back( bv );
          }
        const double ms = std::chrono::duration<double, std::milli>( std::chrono::steady_clock::now() - t0 ).count();
        if( ms < best[v] ) best[v] = ms;
      }
    for( int v = 1; v < 4; v++ )
      if( bestErr[v] != bestErr[0] || bestV[v] != bestV[0] ) { printf( "MISMATCH bs %d %s\n", bs, names[v] ); ok = 0; }
    printf( "%2dx%-2d  %s %.2f ms | %s %.2f ms (%.1f %% faster) | %s %.2f ms | %s %.2f ms (%.1f %% faster)\n", bs, bs,
            names[0], best[0], names[1], best[1], ( best[0] / best[1] - 1 ) * 100, names[2], best[2], names[3], best[3],
            ( best[2] / best[3] - 1 ) * 100 );
  }
  return ok ? 0 : 1;
}
