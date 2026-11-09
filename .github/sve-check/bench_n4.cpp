// Bench for N4: the tree's fastInvCore_neon (writes dst, no clear) against master's g_tCoeffOps.fastInvCore[] (clears,
// then accumulates). Equality of every output over trSize 4..64, rows 4..trSize, lines up to 64 with reducedLines 1..lines
// (all % 4 tails), random and extreme inputs, dst started dirty; then A B B A timing per call.
#include "CommonLib/CommonDef.h"
#include "CommonLib/TrQuant_EMT.h"
#include "CommonLib/Rom.h"

#include <arm_neon.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace vvenc
{
// the tree's kernel, extracted from its Trafo_neon.cpp by bench_n4.sh
#include "n4_kernel.inc"
}   // namespace vvenc

using namespace vvenc;
typedef void ( *CoreFn )( const TMatrixCoeff*, const TCoeff*, TCoeff*, unsigned, unsigned, unsigned );
static volatile int sink = 0;

int main()
{
  g_tCoeffOps.initTCoeffOps( true );
  const CoreFn tree[5] = { fastInvCore_neon<4>, fastInvCore_neon<8>, fastInvCore_neon<16>, fastInvCore_neon<32>, fastInvCore_neon<64> };
  std::mt19937             rng( 4 );
  std::vector<TMatrixCoeff> it( 64 * 64 );
  std::vector<TCoeff>       src( 64 * 64 ), d0( 64 * 64 ), d1( 64 * 64 );
  bool same  = true;
  int  cases = 0;
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
            for( size_t i = 0; i < d0.size(); i++ ) d0[i] = d1[i] = TCoeff( 0x5A5A5A5A + int( i ) );
            g_tCoeffOps.fastInvCore[s]( it.data(), src.data(), d0.data(), lines, red, rows );
            tree[s]( it.data(), src.data(), d1.data(), lines, red, rows );
            // master clears dst[0, red * trSize) and adds into it; the tree must write exactly that area
            if( memcmp( d0.data(), d1.data(), d0.size() * sizeof( TCoeff ) ) )
            {
              if( same ) printf( "DIFFERENT trSize %u lines %u reduced %u rows %u kind %d\n", trSize, lines, red, rows, kind );
              same = false;
            }
            cases++;
          }
    }
  }
  printf( "equality: %s (%d cases)\n", same ? "all equal" : "DIFFERENT", cases );

  for( auto& v : it ) v = TMatrixCoeff( int( rng() % 256 ) - 128 );
  for( auto& v : src ) v = TCoeff( int( rng() % 512 ) - 256 );
  // ( trSize, lines, reducedLines, rows ): typical second-stage calls (lines = the other dimension, rows = its cut-off)
  const unsigned cfg[][4] = { { 4, 4, 4, 4 }, { 4, 8, 8, 4 }, { 8, 8, 8, 8 }, { 8, 16, 16, 8 }, { 16, 16, 16, 16 }, { 16, 16, 16, 8 }, { 32, 32, 32, 16 }, { 32, 32, 16, 32 }, { 64, 64, 32, 32 } };
  for( auto& c : cfg )
  {
    const int    s = floorLog2( c[0] ) - 2;
    const int    n = int( 200000000 / ( c[0] * c[2] * c[3] ) );
    auto run = [&]( CoreFn f, int k ) {
      const auto t0 = std::chrono::steady_clock::now();
      for( int i = 0; i < k; i++ ) { f( it.data(), src.data() + ( i & 7 ), d0.data(), c[1], c[2], c[3] ); sink = sink + d0[i & 63]; }
      return std::chrono::duration<double, std::nano>( std::chrono::steady_clock::now() - t0 ).count() / k; };
    run( g_tCoeffOps.fastInvCore[s], n / 4 ); run( tree[s], n / 4 );
    const double a1 = run( g_tCoeffOps.fastInvCore[s], n ), b1 = run( tree[s], n ), b2 = run( tree[s], n ), a2 = run( g_tCoeffOps.fastInvCore[s], n );
    const double ta = ( a1 + a2 ) / 2, tb = ( b1 + b2 ) / 2;
    printf( "trSize %2u lines %2u reduced %2u rows %2u: master %7.1f ns  tree %7.1f ns  %6.1f %% faster\n", c[0], c[1], c[2], c[3], ta, tb, ( ta / tb - 1 ) * 100 );
  }
  return same ? 0 : 1;
}
