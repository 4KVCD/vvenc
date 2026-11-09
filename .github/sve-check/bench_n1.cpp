// Bench for N1: xGetSAD_generic_neon of a tree against master's NEON SAD pointers (DF_SAD4..128). Equality over every
// width 4..128 x height 1..128, subShift 0 / 1, random samples (original as bi-prediction search makes it, -1023..2046)
// and the extreme difference 2046 vs 0 everywhere; then A B B A timing per call.
#include "CommonLib/CommonDef.h"
#include "CommonLib/RdCost.h"

#include <arm_neon.h>
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

namespace vvenc
{
// the tree's kernel, extracted from its RdCost_neon.cpp by bench_n1.sh
#include "n1_kernel.inc"
}   // namespace vvenc

using namespace vvenc;
static volatile Distortion sink = 0;

int main()
{
  RdCost rc;
  rc.create( /*enableOpt=*/true );   // master's Arm pointers (NEON SAD)
  std::mt19937     rng( 31 );
  const int        S = 256;
  std::vector<Pel> org( S * 260 ), cur( S * 260 );
  DistParam        dp;
  dp.bitDepth    = 10;
  dp.applyWeight = 0;
  bool same  = true;
  int  cases = 0;
  for( int kind = 0; kind < 3; kind++ )
  {
    for( size_t i = 0; i < org.size(); i++ )
    {
      org[i] = Pel( kind == 0 ? 2046 : kind == 1 ? -1023 : int( rng() % 3070 ) - 1023 );
      cur[i] = Pel( kind == 0 ? 0 : kind == 1 ? 1023 : int( rng() % 1024 ) );
    }
    for( int lw = 2; lw <= 7; lw++ )
      for( int lh = 0; lh <= 7; lh++ )
        for( int ss = 0; ss < 2; ss++ )
        {
          const int w = 1 << lw, h = 1 << lh;
          if( ( h >> ss ) < 1 ) continue;
          for( int r = 0; r < 4; r++ )
          {
            dp.org.buf    = org.data() + rng() % 64;
            dp.org.stride = S;
            dp.cur.buf    = cur.data() + rng() % 64;
            dp.cur.stride = S - 8 * ( r & 1 );
            dp.org.width  = w;
            dp.org.height = h;
            dp.subShift   = ss;
            const Distortion a = rc.m_afpDistortFunc[0][DF_SAD + lw]( dp );
            const Distortion b = xGetSAD_generic_neon( dp, w );
            if( a != b ) { printf( "DIFFERENT %dx%d sub %d kind %d: %llu vs %llu\n", w, h, ss, kind, (unsigned long long) a, (unsigned long long) b ); same = false; }
            cases++;
          }
        }
  }
  printf( "equality: %s (%d cases)\n", same ? "all equal" : "DIFFERENT", cases );

  for( auto& v : org ) v = Pel( rng() % 1024 );
  for( auto& v : cur ) v = Pel( rng() % 1024 );
  const int sizes[][3] = { { 128, 128, 1 }, { 128, 128, 0 }, { 128, 64, 1 }, { 64, 64, 1 }, { 64, 64, 0 }, { 64, 32, 1 }, { 32, 32, 1 }, { 32, 32, 0 }, { 32, 16, 1 }, { 16, 16, 1 } };
  for( auto& s : sizes )
  {
    const int w = s[0], h = s[1], lw = floorLog2( w );
    dp.org.width = w; dp.org.height = h; dp.subShift = s[2]; dp.org.stride = S; dp.cur.stride = S;
    const int n = 40000000 / ( w * h );
    auto runA = [&]( int k ) { const auto t0 = std::chrono::steady_clock::now(); for( int i = 0; i < k; i++ ) { dp.org.buf = org.data() + ( i & 7 ); dp.cur.buf = cur.data() + ( i & 3 ); sink = sink + rc.m_afpDistortFunc[0][DF_SAD + lw]( dp ); } return std::chrono::duration<double, std::nano>( std::chrono::steady_clock::now() - t0 ).count() / k; };
    auto runB = [&]( int k ) { const auto t0 = std::chrono::steady_clock::now(); for( int i = 0; i < k; i++ ) { dp.org.buf = org.data() + ( i & 7 ); dp.cur.buf = cur.data() + ( i & 3 ); sink = sink + xGetSAD_generic_neon( dp, w ); } return std::chrono::duration<double, std::nano>( std::chrono::steady_clock::now() - t0 ).count() / k; };
    runA( n / 4 ); runB( n / 4 );
    const double a1 = runA( n ), b1 = runB( n ), b2 = runB( n ), a2 = runA( n );
    const double ta = ( a1 + a2 ) / 2, tb = ( b1 + b2 ) / 2;
    printf( "%3dx%-3d sub%d master %7.1f ns  NEON %7.1f ns  %6.1f %% faster\n", w, h, s[2], ta, tb, ( ta / tb - 1 ) * 100 );
  }
  return same ? 0 : 1;
}
