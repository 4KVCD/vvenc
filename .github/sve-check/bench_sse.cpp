// Per-call bench (fork-only SVE check, not for upstream), argv[1] = SIMD level (NEON, SVE, SVE2): DF_SSE4 .. DF_SSE128
// on square blocks (4x8 for width 4) of 10-bit random samples, best of 5, with a checksum of the results.
#include "CommonLib/RdCost.h"
#include "vvenc/vvenc.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

using namespace vvenc;

int main( int argc, char** argv )
{
  const char* simd = argc > 1 ? argv[1] : "NEON";
  printf( "SIMD %s -> %s\n", simd, vvenc_set_SIMD_extension( simd ) );
  RdCost rdCost;
  rdCost.create( /*enableOpt=*/true );

  std::mt19937     rng( 5 );
  const int        stride = 160;
  std::vector<Pel> org( stride * 128 ), cur( stride * 128 );
  for( auto& v : org ) v = Pel( rng() & 1023 );
  for( auto& v : cur ) v = Pel( rng() & 1023 );

  for( int log2W = 2; log2W <= 7; log2W++ )
  {
    const int w = 1 << log2W, h = std::max( w, 8 );
    DistParam dp;
    dp.org.buf    = org.data();
    dp.org.stride = stride;
    dp.cur.buf    = cur.data();
    dp.cur.stride = stride;
    dp.org.width  = w;
    dp.org.height = h;
    dp.bitDepth   = 10;
    const FpDistFunc f = rdCost.m_afpDistortFunc[0][DF_SSE + log2W];

    const int calls = std::max( 2000, 4000000 / ( w * h ) );
    double best = 1e30;
    unsigned long long check = 0;
    for( int rep = 0; rep < 5; rep++ )
    {
      const auto t0 = std::chrono::steady_clock::now();
      for( int i = 0; i < calls; i++ )
      {
        check += f( dp );
      }
      best = std::min( best, std::chrono::duration<double, std::nano>( std::chrono::steady_clock::now() - t0 ).count() / calls );
    }
    printf( "DF_SSE%-3d %3dx%-3d %8.1f ns  (check %llu)\n", w, w, h, best, check );
  }
  return 0;
}
