// Per-call bench (fork-only SVE check, not for upstream), argv[1] = SIMD level (NEON, SVE): PelBufferOps::
// sumSquaredError at the SBT sub-part sizes, square blocks and a picture (1920 and 1916 x 1080), 10-bit random samples,
// best of 5, with a checksum of the results.
#include "CommonLib/CommonDef.h"
#include "CommonLib/Unit.h"
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
  PelBufferOps ops;
  ops.initPelBufOps( /*enableOpt=*/true );

  std::mt19937     rng( 5 );
  const int        stride = 1936;
  std::vector<Pel> a( stride * 1088 ), b( stride * 1088 );
  for( auto& v : a ) v = Pel( rng() & 1023 );
  for( auto& v : b ) v = Pel( rng() & 1023 );

  const int sizes[][2] = { { 2, 2 }, { 2, 4 }, { 4, 4 }, { 4, 8 }, { 8, 8 }, { 8, 16 }, { 16, 16 }, { 32, 32 },
                           { 64, 64 }, { 1920, 1080 }, { 1916, 1080 } };
  for( const auto& s : sizes )
  {
    const int w = s[0], h = s[1];
    const int calls = std::max( 20, 4000000 / ( w * h ) );
    double best = 1e30;
    unsigned long long check = 0;
    for( int rep = 0; rep < 5; rep++ )
    {
      const auto t0 = std::chrono::steady_clock::now();
      for( int i = 0; i < calls; i++ )
      {
        check += ops.sumSquaredError( a.data() + ( i & 7 ), stride, b.data(), stride, w, h );
      }
      best = std::min( best, std::chrono::duration<double, std::nano>( std::chrono::steady_clock::now() - t0 ).count() / calls );
    }
    printf( "sumSquaredError %4dx%-4d %10.1f ns  (check %llu)\n", w, h, best, check );
  }
  return 0;
}
