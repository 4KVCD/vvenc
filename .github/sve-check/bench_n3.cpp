// Bench for N3: the NEON 16x8 / 8x16 / 8x4 / 4x8 SATDs of whichever libvvenc.a it is linked with (bench_n3.sh links
// master's and the tree's into two binaries). Prints a hash of every result over random and extreme residuals (the two
// binaries must print the same hashes), then the time per call of each kernel.
#include "CommonLib/CommonDef.h"
#include "CommonLib/arm/neon/RdCost_neon.h"

#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

using namespace vvenc;
static volatile Distortion sink = 0;

typedef Distortion ( *HadFn )( const Pel*, const Pel*, const ptrdiff_t, const ptrdiff_t );

int main( int argc, char** argv )
{
  const bool timeOnly = argc > 1;
  const struct { const char* name; HadFn fn; } kernels[] = { { "16x8", xCalcHAD16x8_neon }, { "8x16", xCalcHAD8x16_neon },
                                                            { "8x4", xCalcHAD8x4_neon }, { "4x8", xCalcHAD4x8_neon } };
  const int        S = 64;
  std::vector<Pel> org( S * 64 ), cur( S * 64 );
  if( !timeOnly )
  {
    for( auto& k : kernels )
    {
      std::mt19937 rng( 7 );
      uint64_t     hash = 1469598103934665603ull;
      for( int r = 0; r < 200000; r++ )
      {
        const int kind = r % 5;   // random 10-bit, random bi-pred range, +2046 / -2046 everywhere, sparse
        for( size_t i = 0; i < org.size(); i++ )
        {
          const unsigned v = rng();
          org[i] = Pel( kind == 0 ? int( v % 1024 ) : kind == 1 ? int( v % 3070 ) - 1023 : kind == 2 ? 2046 : kind == 3 ? -1023 : ( v & 15 ) ? 512 : int( v % 1024 ) );
          cur[i] = Pel( kind == 0 ? int( ( v >> 12 ) % 1024 ) : kind == 1 ? int( ( v >> 12 ) % 1024 ) : kind == 2 ? 0 : kind == 3 ? 1023 : 512 );
        }
        const Distortion d = k.fn( org.data(), cur.data(), S, S );
        hash = ( hash ^ uint64_t( d ) ) * 1099511628211ull;
      }
      printf( "%-4s hash %016llx\n", k.name, (unsigned long long) hash );
    }
  }
  std::mt19937 rng( 11 );
  for( auto& v : org ) v = Pel( rng() % 1024 );
  for( auto& v : cur ) v = Pel( rng() % 1024 );
  for( auto& k : kernels )
  {
    const int n = 20000000;
    for( int i = 0; i < n / 4; i++ ) sink = sink + k.fn( org.data() + ( i & 15 ), cur.data() + ( i & 7 ), S, S );
    const auto t0 = std::chrono::steady_clock::now();
    for( int i = 0; i < n; i++ ) sink = sink + k.fn( org.data() + ( i & 15 ), cur.data() + ( i & 7 ), S, S );
    printf( "%-4s %.2f ns\n", k.name, std::chrono::duration<double, std::nano>( std::chrono::steady_clock::now() - t0 ).count() / n );
  }
  return 0;
}
