// Bench for N9 (8-tap luma filters skip a zero end tap): InterpolationFilter::filterHor / filterVer for luma through
// the library it is linked with (run once with master's, once with the tree's), at phases 4 and 12 (one zero end tap)
// and 8 (none), as MC runs them: horizontal only, vertical only, and the 2-D pair (horizontal first pass into an
// intermediate buffer, vertical second pass). Prints a checksum of every output (must match between the two libraries)
// and the best of 5 per-call times.
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

  const int bd = 10;
  const ClpRng clpRng{ bd };
  const int S = 160, T = 136;   // source stride (128 + 7 taps + margin), intermediate / destination stride
  std::mt19937 rng( 9 );
  std::vector<Pel> src( S * 150 ), tmp( T * 140 ), dst( T * 140 );
  for( auto& v : src ) v = Pel( rng() & 1023 );
  const Pel* org = src.data() + 3 * S + 3;   // three samples / rows of margin before the block for the taps

  uint64_t checksum = 0;
  auto sum = [&]( const std::vector<Pel>& b, int w, int h )
  {
    for( int y = 0; y < h; y++ )
      for( int x = 0; x < w; x++ )
        checksum = checksum * 31 + uint16_t( b[y * T + x] );
  };

  enum { HOR, VER, BOTH };
  const char* modeName[3] = { "hor", "ver", "2-D" };
  for( int frac : { 4, 12, 8 } )
    for( int mode : { HOR, VER, BOTH } )
      for( int w : { 8, 16, 32, 64, 128 } )
      {
        const int h = w;
        auto call = [&]( int i )
        {
          const Pel* s = org + ( i & 3 );
          if( mode == HOR )
            filter.filterHor( COMP_Y, s, S, dst.data(), T, w, h, frac, true, CHROMA_420, clpRng );
          else if( mode == VER )
            filter.filterVer( COMP_Y, s, S, dst.data(), T, w, h, frac, true, true, CHROMA_420, clpRng );
          else
          {
            filter.filterHor( COMP_Y, s - 3 * S, S, tmp.data(), T, w, h + 7, frac, false, CHROMA_420, clpRng );
            filter.filterVer( COMP_Y, tmp.data() + 3 * T, T, dst.data(), T, w, h, frac, false, true, CHROMA_420, clpRng );
          }
        };
        call( 0 );
        sum( dst, w, h );
        const int n = 20000000 / ( w * h );
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
        printf( "phase %2d %s %3dx%-3d %9.1f ns\n", frac, modeName[mode], w, h, best );
      }
  printf( "checksum %016llx\n", ( unsigned long long ) checksum );
  return 0;
}
