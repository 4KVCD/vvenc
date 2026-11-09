// N19 per-call bench (fork-only SVE check, not for upstream), argv[1] = SIMD level: DF_HAD8 .. DF_HAD128 and the
// generic DF_HAD (non-fast) on square blocks of random residuals (10-bit samples and the
// bi-pred ME range +-2046 via org = 2046 / cur = 0 extremes mixed in), best of 7, with a checksum of the results so two
// libraries (N17, N19) and levels can be compared for equal outputs.
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

  std::mt19937     rng( 17 );
  const int        stride = 144, rows = 128 * 4;
  std::vector<Pel> org( stride * rows ), cur( stride * rows );
  for( size_t i = 0; i < org.size(); i++ )
  {
    const unsigned r = rng();
    if( ( r & 63 ) == 0 )
    {
      org[i] = ( r & 64 ) ? 2046 : -2046;
      cur[i] = 0;
    }
    else
    {
      org[i] = Pel( r & 1023 );
      cur[i] = Pel( ( r >> 10 ) & 1023 );
    }
  }

  for( int generic = 0; generic <= 1; generic++ )
  for( int log2W = 3; log2W <= 7; log2W++ )
  {
    const int  w = 1 << log2W;
    const int  blocks = rows / w;
    DistParam  dp;
    dp.org.stride = stride;
    dp.cur.stride = stride;
    dp.org.width  = w;
    dp.org.height = w;
    dp.bitDepth   = 10;
    const FpDistFunc f = rdCost.m_afpDistortFunc[0][generic ? DF_HAD : DF_HAD + log2W];

    const int          calls = std::max( 4000, 8000000 / ( w * w ) );
    double             best  = 1e30;
    unsigned long long check = 0;
    for( int rep = 0; rep < 7; rep++ )
    {
      const auto t0 = std::chrono::steady_clock::now();
      for( int i = 0; i < calls; i++ )
      {
        const int b    = i % blocks;
        dp.org.buf     = org.data() + b * w * stride;
        dp.cur.buf     = cur.data() + b * w * stride;
        check         += f( dp );
      }
      best = std::min( best, std::chrono::duration<double, std::nano>( std::chrono::steady_clock::now() - t0 ).count() / calls );
    }
    printf( "%s %3dx%-3d %9.1f ns  (check %llu)\n", generic ? "generic" : "DF_HADw", w, w, best, check );
  }
  return 0;
}
