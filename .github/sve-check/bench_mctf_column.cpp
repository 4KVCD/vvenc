// Per-call bench for NEON item A4: the errors of num subpel vectors with one horizontal position (MCTF low-res search).
//   column: one m_motionErrorLumaFracColumn call (only when the tree sets it)
//   single: num m_motionErrorLumaFrac8[1] calls, as MCTF.cpp does without the column kernel: besterror INT_MAX (no
//           early exit) and besterror = the first vector's error (an early exit as the search's best error allows)
// 10-bit random samples, w x h 16x16 / 32x32 / 16x8 / 32x16, num 2..4 (MAX_COLUMN), row offsets 0 / 1 alternating.
#include "CommonLib/MCTF.h"

#include <chrono>
#include <climits>
#include <cstdio>
#include <random>
#include <vector>

using namespace vvenc;

static volatile int sink = 0;

template<class F> static double timeIt( F f )
{
  const int calls = 200000;
  const auto t0 = std::chrono::steady_clock::now();
  for( int i = 0; i < calls; i++ )
  {
    sink = sink + f();
  }
  return std::chrono::duration<double, std::nano>( std::chrono::steady_clock::now() - t0 ).count() / calls;
}

int main()
{
  MCTF mctf;   // the constructor installs the SIMD kernels of the build
  std::mt19937 rng( 3 );
  const int stride = 64;
  std::vector<Pel> org( stride * 40 ), buf( stride * 48 );
  for( auto& v : org ) v = Pel( rng() & 1023 );
  for( auto& v : buf ) v = Pel( rng() & 1023 );
  const Pel* o = org.data();
  const Pel* b = buf.data() + 4 * stride + 4;
  printf( "m_motionErrorLumaFracColumn %s\n", mctf.m_motionErrorLumaFracColumn ? "set" : "null" );

  const int sizes[][2] = { { 16, 8 }, { 16, 16 }, { 32, 16 }, { 32, 32 } };
  for( const auto& s : sizes )
  for( int num = 2; num <= 4; num++ )
  {
    const int w = s[0], h = s[1];
    const int16_t* xFilter = MCTF::m_interpolationFilter4[5];
    const int16_t* yFilter[4] = { MCTF::m_interpolationFilter4[3], MCTF::m_interpolationFilter4[0], MCTF::m_interpolationFilter4[8], MCTF::m_interpolationFilter4[13] };
    const int rowOffset[4] = { 0, 1, 0, 1 };
    int errs[4];

    // early: 0 none, 1 besterror = the first vector's error, 2 besterror = 1/8 of it (most vectors exit after a few rows)
    auto single = [&]( int early )
    {
      int sum = 0, best = INT_MAX;
      for( int i = 0; i < num; i++ )
      {
        const int e = mctf.m_motionErrorLumaFrac8[1]( o, stride, b + rowOffset[i] * stride, stride, w, h, xFilter, yFilter[i], 10, early ? best : INT_MAX );
        if( early && i == 0 ) best = early == 2 ? e / 8 : e;
        sum += e;
      }
      return sum;
    };
    auto column = [&]()
    {
      mctf.m_motionErrorLumaFracColumn( o, stride, b, stride, w, h, xFilter, num, yFilter, rowOffset, 10, errs );
      return errs[0] + errs[num - 1];
    };
    auto singleFull  = [&]() { return single( 0 ); };
    auto singleEarly = [&]() { return single( 1 ); };
    auto singleEighth = [&]() { return single( 2 ); };

    const double s1 = timeIt( singleFull ), e1 = timeIt( singleEarly ), q1 = timeIt( singleEighth );
    if( mctf.m_motionErrorLumaFracColumn )
    {
      // the column kernel's errors must equal the per-vector kernel's
      column();
      for( int i = 0; i < num; i++ )
      {
        const int e = mctf.m_motionErrorLumaFrac8[1]( o, stride, b + rowOffset[i] * stride, stride, w, h, xFilter, yFilter[i], 10, INT_MAX );
        if( e != errs[i] ) { printf( "MISMATCH %dx%d num %d vector %d: %d vs %d\n", w, h, num, i, errs[i], e ); return 1; }
      }
      const double c1 = timeIt( column ), c2 = timeIt( column );
      const double s2 = timeIt( singleFull ), e2 = timeIt( singleEarly ), q2 = timeIt( singleEighth );
      const double c = ( c1 + c2 ) / 2;
      printf( "%2dx%-2d num %d  single %7.1f / exit@e0 %7.1f / exit@e0/8 %7.1f ns  column %7.1f ns  %5.1f / %5.1f / %5.1f %% faster\n", w, h, num,
              ( s1 + s2 ) / 2, ( e1 + e2 ) / 2, ( q1 + q2 ) / 2, c, 100 * ( ( s1 + s2 ) / 2 / c - 1 ), 100 * ( ( e1 + e2 ) / 2 / c - 1 ),
              100 * ( ( q1 + q2 ) / 2 / c - 1 ) );
    }
    else
    {
      printf( "%2dx%-2d num %d  single %7.1f ns  single(early exit) %7.1f ns\n", w, h, num, s1, e1 );
    }
  }
  return 0;
}
