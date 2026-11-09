// N1 check (fork-only, not for upstream): per call through the library's DF_SAD pointers, so master's and the tree's
// libraries are compared as the encoder calls them. Prints a checksum of all results (must match) and best-of-7 ns.
#include "CommonLib/RdCost.h"
#include "vvenc/vvenc.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

using namespace vvenc;
static volatile Distortion sink = 0;

int main( int argc, char** argv )
{
  printf( "SIMD %s\n", vvenc_set_SIMD_extension( argc > 1 ? argv[1] : "NEON" ) );
  RdCost rc;
  rc.create( /*enableOpt=*/true );
  std::mt19937     rng( 31 );
  const int        S = 256;
  std::vector<Pel> org( S * 260 ), cur( S * 260 );
  for( auto& v : org ) v = Pel( rng() % 1024 );
  for( auto& v : cur ) v = Pel( rng() % 1024 );
  DistParam dp;
  dp.bitDepth    = 10;
  dp.applyWeight = 0;
  unsigned long long check = 0;
  const int sizes[][3] = { { 128, 128, 1 }, { 128, 128, 0 }, { 128, 64, 1 }, { 64, 64, 1 }, { 64, 64, 0 }, { 64, 32, 1 },
                           { 32, 32, 1 }, { 32, 32, 0 }, { 32, 16, 1 }, { 16, 16, 1 } };
  for( auto& s : sizes )
  {
    const int w = s[0], h = s[1], lw = floorLog2( w );
    dp.org.width = w; dp.org.height = h; dp.subShift = s[2]; dp.org.stride = S; dp.cur.stride = S;
    const FpDistFunc f = rc.m_afpDistortFunc[0][DF_SAD + lw];
    const int n = 40000000 / ( w * h );
    double best = 1e30;
    for( int rep = 0; rep < 7; rep++ )
    {
      const auto t0 = std::chrono::steady_clock::now();
      for( int i = 0; i < n; i++ )
      {
        dp.org.buf = org.data() + ( i & 7 );
        dp.cur.buf = cur.data() + ( i & 3 );
        const Distortion d = f( dp );
        sink = sink + d;
        if( rep == 0 && i < 64 ) check = check * 31 + d;
      }
      best = std::min( best, std::chrono::duration<double, std::nano>( std::chrono::steady_clock::now() - t0 ).count() / n );
    }
    printf( "%3dx%-3d sub%d %8.1f ns\n", w, h, s[2], best );
  }
  printf( "checksum %llu\n", check );
  return 0;
}
