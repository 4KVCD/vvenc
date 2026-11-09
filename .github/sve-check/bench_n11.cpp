// Bench for N11 (NEON forward DCT-II 8 with the even/odd split): fastForwardDCT2_B8 through the library it is linked
// with (master's, then the tree's), as both stages of a forward 8-point transform call it: the first stage on 10-bit
// residuals (shift 4), the second on the first stage's range (shift 9); 4 to 64 lines, no skipped lines or columns.
// Prints a checksum of every output (must match between the two libraries) and the best of 5 per-call times.
#include "CommonLib/CommonDef.h"
#include "CommonLib/TrQuant_EMT.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using namespace vvenc;

static volatile int sink = 0;

int main()
{
  g_tCoeffOps.initTCoeffOps( true );
  std::mt19937 rng( 11 );
  std::vector<TCoeff> src( 8 * 64 * 4 ), dst( 8 * 64 );
  uint64_t checksum = 0;

  for( int stage = 0; stage < 2; stage++ )
  {
    const int shift = stage == 0 ? 4 : 9;
    const int range = stage == 0 ? 1023 : 8191;
    for( auto& v : src ) v = int( rng() % ( 2 * range + 1 ) ) - range;
    for( int line : { 4, 8, 16, 32, 64 } )
    {
      auto call = [&]( int i ) { fastForwardDCT2_B8( src.data() + ( i & 3 ) * 8 * 64, dst.data(), shift, line, 0, 0 ); };
      for( int i = 0; i < 4; i++ )
      {
        call( i );
        for( int n = 0; n < 8 * line; n++ ) checksum = checksum * 31 + uint32_t( dst[n] );
      }
      const int n = 40000000 / ( 64 * line );
      auto run = [&]( int k )
      {
        auto t0 = std::chrono::steady_clock::now();
        for( int i = 0; i < k; i++ )
        {
          call( i );
          sink = sink + dst[1];
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
      printf( "stage %d lines %2d %8.1f ns\n", stage + 1, line, best );
    }
  }
  printf( "checksum %016llx\n", ( unsigned long long ) checksum );
  return 0;
}
