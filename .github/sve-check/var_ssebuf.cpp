// Variant bench (fork-only SVE check, not for upstream): why is the SVE sumSquaredError slower than NEON from 32x32 up
// on Neoverse N2? Kernels for widths that are multiples of 16 only (no tail), 10-bit random samples, best of 5.
//   neon2  : #871's NEON accumulator (vmull + vmlal + vpadal into uint64), 2 chains (the shipped NEON kernel)
//   sve2   : one vvenc_sdotq_s16 per 8 differences, 2 chains, 16 columns per step (N20 as committed)
//   sve4c  : 4 chains, 32 columns per step (16-column remainder on chains 0 and 1)
//   sve4r  : 4 chains, 16 columns of 2 rows per step (odd last row on chains 0 and 1)
//   neon4r : the NEON accumulator on 4 chains, as sve4r
#include <arm_neon.h>
#include <arm_neon_sve_bridge.h>
#include <arm_sve.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

typedef int16_t Pel;

static inline int64x2_t sdot( int64x2_t acc, int16x8_t x )
{
  return svget_neonq_s64( svdot_s64( svset_neonq_s64( svundef_s64(), acc ), svset_neonq_s16( svundef_s16(), x ),
                                     svset_neonq_s16( svundef_s16(), x ) ) );
}

static inline uint64x2_t neonAdd( uint64x2_t acc, int16x8_t d )
{
  int32x4_t sq = vmull_s16( vget_low_s16( d ), vget_low_s16( d ) );
  sq           = vmlal_s16( sq, vget_high_s16( d ), vget_high_s16( d ) );
  return vpadalq_u32( acc, vreinterpretq_u32_s32( sq ) );
}

static inline int16x8_t diff8( const Pel* a, const Pel* b ) { return vsubq_s16( vld1q_s16( a ), vld1q_s16( b ) ); }

__attribute__( ( noinline ) ) uint64_t neon2( const Pel* a, ptrdiff_t sa, const Pel* b, ptrdiff_t sb, int w, int h )
{
  uint64x2_t acc0 = vdupq_n_u64( 0 ), acc1 = vdupq_n_u64( 0 );
  for( int y = 0; y < h; y++, a += sa, b += sb )
    for( int x = 0; x < w; x += 16 )
    {
      acc0 = neonAdd( acc0, diff8( a + x, b + x ) );
      acc1 = neonAdd( acc1, diff8( a + x + 8, b + x + 8 ) );
    }
  return vaddvq_u64( vaddq_u64( acc0, acc1 ) );
}

__attribute__( ( noinline ) ) uint64_t sve2( const Pel* a, ptrdiff_t sa, const Pel* b, ptrdiff_t sb, int w, int h )
{
  int64x2_t acc0 = vdupq_n_s64( 0 ), acc1 = vdupq_n_s64( 0 );
  for( int y = 0; y < h; y++, a += sa, b += sb )
    for( int x = 0; x < w; x += 16 )
    {
      acc0 = sdot( acc0, diff8( a + x, b + x ) );
      acc1 = sdot( acc1, diff8( a + x + 8, b + x + 8 ) );
    }
  return uint64_t( vaddvq_s64( vaddq_s64( acc0, acc1 ) ) );
}

__attribute__( ( noinline ) ) uint64_t sve4c( const Pel* a, ptrdiff_t sa, const Pel* b, ptrdiff_t sb, int w, int h )
{
  int64x2_t acc0 = vdupq_n_s64( 0 ), acc1 = vdupq_n_s64( 0 ), acc2 = vdupq_n_s64( 0 ), acc3 = vdupq_n_s64( 0 );
  for( int y = 0; y < h; y++, a += sa, b += sb )
  {
    int x = 0;
    for( ; x + 32 <= w; x += 32 )
    {
      acc0 = sdot( acc0, diff8( a + x, b + x ) );
      acc1 = sdot( acc1, diff8( a + x + 8, b + x + 8 ) );
      acc2 = sdot( acc2, diff8( a + x + 16, b + x + 16 ) );
      acc3 = sdot( acc3, diff8( a + x + 24, b + x + 24 ) );
    }
    if( x < w )
    {
      acc0 = sdot( acc0, diff8( a + x, b + x ) );
      acc1 = sdot( acc1, diff8( a + x + 8, b + x + 8 ) );
    }
  }
  return uint64_t( vaddvq_s64( vaddq_s64( vaddq_s64( acc0, acc1 ), vaddq_s64( acc2, acc3 ) ) ) );
}

__attribute__( ( noinline ) ) uint64_t sve4r( const Pel* a, ptrdiff_t sa, const Pel* b, ptrdiff_t sb, int w, int h )
{
  int64x2_t acc0 = vdupq_n_s64( 0 ), acc1 = vdupq_n_s64( 0 ), acc2 = vdupq_n_s64( 0 ), acc3 = vdupq_n_s64( 0 );
  int       y    = 0;
  for( ; y + 2 <= h; y += 2, a += 2 * sa, b += 2 * sb )
    for( int x = 0; x < w; x += 16 )
    {
      acc0 = sdot( acc0, diff8( a + x, b + x ) );
      acc1 = sdot( acc1, diff8( a + x + 8, b + x + 8 ) );
      acc2 = sdot( acc2, diff8( a + sa + x, b + sb + x ) );
      acc3 = sdot( acc3, diff8( a + sa + x + 8, b + sb + x + 8 ) );
    }
  if( y < h )
    for( int x = 0; x < w; x += 16 )
    {
      acc0 = sdot( acc0, diff8( a + x, b + x ) );
      acc1 = sdot( acc1, diff8( a + x + 8, b + x + 8 ) );
    }
  return uint64_t( vaddvq_s64( vaddq_s64( vaddq_s64( acc0, acc1 ), vaddq_s64( acc2, acc3 ) ) ) );
}

__attribute__( ( noinline ) ) uint64_t neon4r( const Pel* a, ptrdiff_t sa, const Pel* b, ptrdiff_t sb, int w, int h )
{
  uint64x2_t acc0 = vdupq_n_u64( 0 ), acc1 = vdupq_n_u64( 0 ), acc2 = vdupq_n_u64( 0 ), acc3 = vdupq_n_u64( 0 );
  int        y    = 0;
  for( ; y + 2 <= h; y += 2, a += 2 * sa, b += 2 * sb )
    for( int x = 0; x < w; x += 16 )
    {
      acc0 = neonAdd( acc0, diff8( a + x, b + x ) );
      acc1 = neonAdd( acc1, diff8( a + x + 8, b + x + 8 ) );
      acc2 = neonAdd( acc2, diff8( a + sa + x, b + sb + x ) );
      acc3 = neonAdd( acc3, diff8( a + sa + x + 8, b + sb + x + 8 ) );
    }
  if( y < h )
    for( int x = 0; x < w; x += 16 )
    {
      acc0 = neonAdd( acc0, diff8( a + x, b + x ) );
      acc1 = neonAdd( acc1, diff8( a + x + 8, b + x + 8 ) );
    }
  return vaddvq_u64( vaddq_u64( vaddq_u64( acc0, acc1 ), vaddq_u64( acc2, acc3 ) ) );
}

typedef uint64_t ( *Kernel )( const Pel*, ptrdiff_t, const Pel*, ptrdiff_t, int, int );

int main()
{
  std::mt19937     rng( 5 );
  const int        stride = 1936;
  std::vector<Pel> a( stride * 1088 ), b( stride * 1088 );
  for( auto& v : a ) v = Pel( rng() & 1023 );
  for( auto& v : b ) v = Pel( rng() & 1023 );

  const struct { const char* name; Kernel k; } kernels[] = { { "neon2", neon2 }, { "sve2", sve2 }, { "sve4c", sve4c },
                                                             { "sve4r", sve4r }, { "neon4r", neon4r } };
  const int sizes[][2] = { { 16, 16 }, { 32, 32 }, { 64, 64 }, { 128, 128 }, { 1920, 1080 } };
  for( int round = 1; round <= 3; round++ )
  {
    printf( "== round %d\n", round );
    for( const auto& s : sizes )
    {
      const int w = s[0], h = s[1];
      const int calls = std::max( 20, 4000000 / ( w * h ) );
      printf( "%4dx%-4d", w, h );
      for( const auto& kn : kernels )
      {
        double   best  = 1e30;
        uint64_t check = 0;
        for( int rep = 0; rep < 5; rep++ )
        {
          const auto t0 = std::chrono::steady_clock::now();
          for( int i = 0; i < calls; i++ )
          {
            check += kn.k( a.data() + ( i & 7 ), stride, b.data(), stride, w, h );
          }
          best = std::min( best, std::chrono::duration<double, std::nano>( std::chrono::steady_clock::now() - t0 ).count() / calls );
        }
        printf( "  %s %9.1f ns (%llu)", kn.name, best, (unsigned long long) ( check % 1000000 ) );
      }
      printf( "\n" );
    }
  }
  return 0;
}
