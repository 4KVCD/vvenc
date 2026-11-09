// N33 through the library's own pointer: InterpolationFilter::m_filterN2_2D after initInterpolationFilter( true ) of
// the library this is linked against (master or the tree). Prints a hash of every output (bit depths 8 / 10 / 12,
// every fraction pair, DMVR shapes and larger, random and extreme inputs; the block only), then the best of 5 times
// per call at the DMVR shapes (16 or 8, plus 4).
#include "CommonLib/InterpolationFilter.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using namespace vvenc;
static volatile int sink = 0;

int main()
{
  InterpolationFilter flt;
  flt.initInterpolationFilter( true );
  const int srcStride = 160, dstStride = 136;
  std::vector<Pel> src( srcStride * 140 ), dst( dstStride * 136 );
  std::mt19937 rng( 33 );
  const int shapes[][2] = { { 12, 12 }, { 12, 20 }, { 20, 12 }, { 20, 20 }, { 36, 36 }, { 132, 132 }, { 4, 2 }, { 12, 2 } };
  uint64_t hash = 1469598103934665603ull;
  long cases = 0;
  for( int bd : { 8, 10, 12 } )
    for( int kind = 0; kind < 3; kind++ )
    {
      for( auto& v : src ) v = Pel( kind == 0 ? rng() % ( 1 << bd ) : kind == 1 ? ( rng() & 1 ? ( 1 << bd ) - 1 : 0 ) : ( 1 << bd ) - 1 );
      for( auto& sh : shapes )
        for( int fx = 1; fx < 16; fx++ )
          for( int fy = 1; fy < 16; fy++ )
          {
            const ClpRng c{ bd };
            flt.m_filterN2_2D( c, src.data() + 2, srcStride, dst.data(), dstStride, sh[0], sh[1], InterpolationFilter::m_bilinearFilterPrec4[fx],
                               InterpolationFilter::m_bilinearFilterPrec4[fy] );
            for( int y = 0; y < sh[1]; y++ )
              for( int x = 0; x < sh[0]; x++ ) hash = ( hash ^ uint16_t( dst[y * dstStride + x] ) ) * 1099511628211ull;
            cases++;
          }
    }
  printf( "output hash %016llx (%ld cases)\n", (unsigned long long) hash, cases );

  for( auto& v : src ) v = Pel( rng() % 1024 );
  const ClpRng c{ 10 };
  for( int s = 0; s < 4; s++ )
  {
    const int w = shapes[s][0], h = shapes[s][1];
    const int n = int( 400000000 / ( w * h ) );
    auto run = [&]( int k ) {
      const auto t0 = std::chrono::steady_clock::now();
      for( int i = 0; i < k; i++ )
      {
        flt.m_filterN2_2D( c, src.data() + 2 + ( i & 3 ), srcStride, dst.data(), dstStride, w, h, InterpolationFilter::m_bilinearFilterPrec4[1 + ( i & 7 )],
                           InterpolationFilter::m_bilinearFilterPrec4[9 - ( i & 7 )] );
        sink = sink + dst[i & 15];
      }
      return std::chrono::duration<double, std::nano>( std::chrono::steady_clock::now() - t0 ).count() / k; };
    run( n / 4 );
    double best = 1e30;
    for( int r = 0; r < 5; r++ ) best = std::min( best, run( n ) );
    printf( "%3dx%-3d %7.2f ns\n", w, h, best );
  }
  return 0;
}
