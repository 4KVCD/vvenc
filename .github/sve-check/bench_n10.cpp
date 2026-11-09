// Bench for N10 (fused 2-D 8-tap luma filters skip a zero end tap): InterpolationFilter::filter8xH / filter16xH for
// luma through the library it is linked with (run once with the base's, once with the tree's), at phase pairs (x, y)
// with zero end taps in one or both dimensions (4, 12) and none (8). Prints a checksum of every output (must match
// between the two libraries) and the best of 5 per-call times.
#include "CommonLib/CommonDef.h"
#include "CommonLib/InterpolationFilter.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using namespace vvenc;

static volatile int sink = 0;

int main()
{
  InterpolationFilter filter;
  filter.initInterpolationFilter( true );

  const ClpRng clpRng{ 10 };
  const int S = 48, T = 32;   // source stride (16 + 7 taps + margin), destination stride
  std::mt19937 rng( 9 );
  std::vector<Pel> src( S * 40 ), dst( T * 20 );
  for( auto& v : src ) v = Pel( rng() & 1023 );
  const Pel* org = src.data() + 3 * S + 3;   // three samples / rows of margin before the block for the taps

  uint64_t checksum = 0;
  const int pairs[][2] = { { 4, 4 }, { 12, 12 }, { 4, 8 }, { 8, 8 } };
  for( const auto& p : pairs )
    for( int w : { 8, 16 } )
      for( int h : { 8, 16 } )
        for( int isLast = 0; isLast < 2; isLast++ )
        {
          auto call = [&]( int i )
          {
            const Pel* s = org + ( i & 3 );
            if( w == 8 )
              filter.filter8xH( COMP_Y, s, S, dst.data(), T, w, h, p[0], p[1], isLast, CHROMA_420, clpRng, false );
            else
              filter.filter16xH( COMP_Y, s, S, dst.data(), T, w, h, p[0], p[1], isLast, CHROMA_420, clpRng, false );
          };
          call( 0 );
          for( int y = 0; y < h; y++ )
            for( int x = 0; x < w; x++ )
              checksum = checksum * 31 + uint16_t( dst[y * T + x] );
          const int n = 4000000 / ( w * h ) * 16;
          auto run = [&]( int k )
          {
            auto t0 = std::chrono::steady_clock::now();
            for( int i = 0; i < k; i++ )
            {
              call( i );
              sink = sink + dst[3];
            }
            return std::chrono::duration<double, std::nano>( std::chrono::steady_clock::now() - t0 ).count() / k;
          };
          run( n / 4 );
          double best = 1e30;
          for( int r = 0; r < 5; r++ )
          {
            const double t = run( n );
            best = t < best ? t : best;
          }
          printf( "phases %2d,%-2d %2dx%-2d last %d %8.1f ns\n", p[0], p[1], w, h, isLast, best );
        }
  printf( "checksum %016llx\n", (unsigned long long) checksum );
  return 0;
}
