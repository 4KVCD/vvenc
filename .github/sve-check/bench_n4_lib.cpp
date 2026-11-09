// N24 audit of #903 (N4) through the library's own pointer: g_tCoeffOps.fastInvCore[] of the library this is linked
// against (master or #903). Prints a hash of every output (dst[0, reducedLines * trSize), the area both versions write)
// over trSize 4..64, lines 4..64, reducedLines 1..lines, rows 4..trSize, three input kinds, so the two libraries'
// outputs can be compared; then the time per call of typical second-stage calls.
#include "CommonLib/CommonDef.h"
#include "CommonLib/TrQuant_EMT.h"
#include "CommonLib/Rom.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using namespace vvenc;
typedef void ( *CoreFn )( const TMatrixCoeff*, const TCoeff*, TCoeff*, unsigned, unsigned, unsigned );
static volatile int sink = 0;

int main()
{
  g_tCoeffOps.initTCoeffOps( true );
  std::mt19937              rng( 4 );
  std::vector<TMatrixCoeff> it( 64 * 64 );
  std::vector<TCoeff>       src( 64 * 64 ), dst( 64 * 64 );
  uint64_t hash  = 1469598103934665603ull;
  int      cases = 0;
  for( int s = 0; s < 5; s++ )
  {
    const unsigned trSize = 4u << s;
    for( int kind = 0; kind < 3; kind++ )
    {
      for( auto& v : it ) v = TMatrixCoeff( kind == 2 ? ( rng() & 1 ? 127 : -128 ) : int( rng() % 256 ) - 128 );
      for( auto& v : src ) v = TCoeff( kind == 0 ? int( rng() % 512 ) - 256 : kind == 1 ? int( rng() % 65536 ) - 32768 : ( rng() & 1 ? 32767 : -32768 ) );
      for( unsigned lines = 4; lines <= 64; lines += ( lines < 8 ? 1 : 4 ) )
        for( unsigned red = 1; red <= lines; red++ )
          for( unsigned rows = 4; rows <= trSize; rows += 4 )
          {
            for( size_t i = 0; i < dst.size(); i++ ) dst[i] = TCoeff( 0x5A5A5A5A + int( i ) );   // dirty
            g_tCoeffOps.fastInvCore[s]( it.data(), src.data(), dst.data(), lines, red, rows );
            for( size_t i = 0; i < size_t( red ) * trSize; i++ ) hash = ( hash ^ uint32_t( dst[i] ) ) * 1099511628211ull;
            cases++;
          }
    }
  }
  printf( "output hash %016llx (%d cases)\n", (unsigned long long) hash, cases );

  for( auto& v : it ) v = TMatrixCoeff( int( rng() % 256 ) - 128 );
  for( auto& v : src ) v = TCoeff( int( rng() % 512 ) - 256 );
  // ( trSize, lines, reducedLines, rows ): typical second-stage calls (lines = the other dimension, rows = its cut-off)
  const unsigned cfg[][4] = { { 4, 4, 4, 4 }, { 4, 8, 8, 4 }, { 8, 8, 8, 8 }, { 8, 16, 16, 8 }, { 16, 16, 16, 16 }, { 16, 16, 16, 8 }, { 32, 32, 32, 16 }, { 32, 32, 16, 32 }, { 64, 64, 32, 32 } };
  for( auto& c : cfg )
  {
    const int s = floorLog2( c[0] ) - 2;
    const int n = int( 200000000 / ( c[0] * c[2] * c[3] ) );
    const CoreFn f = g_tCoeffOps.fastInvCore[s];
    auto run = [&]( int k ) {
      const auto t0 = std::chrono::steady_clock::now();
      for( int i = 0; i < k; i++ ) { f( it.data(), src.data() + ( i & 7 ), dst.data(), c[1], c[2], c[3] ); sink = sink + dst[i & 63]; }
      return std::chrono::duration<double, std::nano>( std::chrono::steady_clock::now() - t0 ).count() / k; };
    run( n / 4 );
    double best = 1e30;
    for( int r = 0; r < 5; r++ ) best = std::min( best, run( n ) );
    printf( "trSize %2u lines %2u reduced %2u rows %2u: %7.2f ns\n", c[0], c[1], c[2], c[3], best );
  }
  return 0;
}
