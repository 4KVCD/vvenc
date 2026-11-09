// N35 through the library's own pointers: TCoeffOps::fastFwdDCT2[2..4] (16 / 32 / 64) of the library this is linked
// against (#911 or the tree). Prints a hash of every output (lines 4..64, every iSkipLine / iSkipLine2, residual-range,
// +-8192-range and full-int16 inputs), then the best of 5 times per call.
#include "CommonLib/CommonDef.h"
#include "CommonLib/TrQuant_EMT.h"
#include "CommonLib/Rom.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace vvenc;
static volatile int sink = 0;

int main()
{
  g_tCoeffOps.initTCoeffOps( true );
  std::mt19937 rng( 35 );
  std::vector<TCoeff> src( 64 * 64 ), dst( 64 * 64 );
  uint64_t hash = 1469598103934665603ull;
  long cases = 0;
  for( int s = 2; s <= 4; s++ )
  {
    const int N = 4 << s;
    for( int kind = 0; kind < 3; kind++ )
      for( int line = 4; line <= 64; line += 4 )
        for( int skip = 0; skip < line; skip += 4 )
          for( int skip2 = N == 64 ? 32 : 0; skip2 < N - 4; skip2 += 4 )
          {
            for( auto& v : src ) v = TCoeff( kind == 0 ? int( rng() % 1024 ) - 512 : kind == 1 ? int( rng() % 16384 ) - 8192 : int16_t( rng() ) );
            memset( dst.data(), 0x5a, dst.size() * 4 );
            g_tCoeffOps.fastFwdDCT2[s]( src.data(), dst.data(), 2 + ( cases & 7 ), line, skip, skip2 );
            for( int i = 0; i < N * line; i++ ) hash = ( hash ^ uint32_t( dst[i] ) ) * 1099511628211ull;
            cases++;
          }
  }
  printf( "output hash %016llx (%ld cases)\n", (unsigned long long) hash, cases );
  for( auto& v : src ) v = TCoeff( int( rng() % 1024 ) - 512 );
  const int cfg[][3] = { { 2, 16, 0 }, { 2, 8, 0 }, { 2, 16, 8 }, { 3, 32, 0 }, { 3, 16, 0 }, { 3, 32, 16 }, { 4, 64, 32 } };   // ( slot, lines, iSkipLine2 )
  for( auto& c : cfg )
  {
    const int N = 4 << c[0], line = c[1], skip2 = c[2];
    const int n = int( 100000000 / ( N * line ) );
    auto run = [&]( int k ) {
      const auto t0 = std::chrono::steady_clock::now();
      for( int i = 0; i < k; i++ ) { g_tCoeffOps.fastFwdDCT2[c[0]]( src.data() + ( i & 3 ), dst.data(), 9, line, 0, skip2 ); sink = sink + dst[i & 63]; }
      return std::chrono::duration<double, std::nano>( std::chrono::steady_clock::now() - t0 ).count() / k; };
    run( n / 4 );
    double best = 1e30;
    for( int r = 0; r < 5; r++ ) best = std::min( best, run( n ) );
    printf( "N %2d lines %2d cut-off %2d: %8.2f ns\n", N, line, N - skip2, best );
  }
  return 0;
}
