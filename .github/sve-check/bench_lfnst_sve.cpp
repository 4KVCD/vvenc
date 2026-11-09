// LFNST check and per-call bench (fork-only SVE check, not for upstream), argv[1] = SIMD level (SCALAR, NEON, SVE):
// the TrQuant LFNST pointers at that level on every mode, index, size and zero-out size with inputs of 9, 16 and 18
// bits; prints a hash of all outputs (must equal SCALAR's) and the time per call on residual-like inputs, best of 5.
#define private public
#include "CommonLib/TrQuant.h"
#undef private
#include "vvenc/vvenc.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using namespace vvenc;

int main( int argc, char** argv )
{
  const char* simd = argc > 1 ? argv[1] : "NEON";
  printf( "SIMD %s -> %s\n", simd, vvenc_set_SIMD_extension( simd ) );
  TrQuant tq( /*enableOpt=*/true );

  std::mt19937     rng( 9 );
  std::vector<int> src( 48 ), s2( 48 ), out( 48 );
  uint64_t         hash  = 1469598103934665603ull;
  int              cases = 0;
  for( int inv = 0; inv < 2; inv++ )
    for( uint32_t mode = 0; mode < 4; mode++ )
      for( uint32_t idx = 0; idx < 2; idx++ )
        for( uint32_t size : { 4u, 8u } )
          for( int zo : { 8, 16 } )
            for( int r = 0; r < 300; r++ )
            {
              const int range = r % 3 == 0 ? ( 1 << 17 ) : r % 3 == 1 ? ( 1 << 15 ) : 512;
              for( auto& v : src ) v = int( rng() % ( 2 * range ) ) - range;
              s2 = src;
              std::fill( out.begin(), out.end(), 0x5a5a5a5a );
              if( inv ) tq.m_invLfnstNxN( s2.data(), out.data(), mode, idx, size, zo );
              else      tq.m_fwdLfnstNxN( s2.data(), out.data(), mode, idx, size, zo );
              for( int v : out ) hash = ( hash ^ uint32_t( v ) ) * 1099511628211ull;
              cases++;
            }
  printf( "output hash %016llx (%d cases)\n", (unsigned long long) hash, cases );

  std::vector<std::vector<int>> in( 64, std::vector<int>( 48 ) );
  for( auto& v : in ) for( auto& x : v ) x = int( rng() % 801 ) - 400;
  struct C { const char* name; bool inv; uint32_t size; int zo; };
  const C cs[] = { { "inv 4x4 zo  8", true, 4, 8 },   { "inv 4x4 zo 16", true, 4, 16 },  { "inv 8x8 zo  8", true, 8, 8 },
                   { "inv 8x8 zo 16", true, 8, 16 },  { "fwd 4x4 zo  8", false, 4, 8 }, { "fwd 4x4 zo 16", false, 4, 16 },
                   { "fwd 8x8 zo  8", false, 8, 8 },  { "fwd 8x8 zo 16", false, 8, 16 } };
  for( const auto& c : cs )
  {
    const int n    = 1000000;
    double    best = 1e30;
    for( int rep = 0; rep < 5; rep++ )
    {
      const auto t0 = std::chrono::steady_clock::now();
      for( int i = 0; i < n; i++ )
      {
        if( c.inv ) tq.m_invLfnstNxN( in[i & 63].data(), out.data(), i & 3, 1, c.size, c.zo );
        else        tq.m_fwdLfnstNxN( in[i & 63].data(), out.data(), i & 3, 1, c.size, c.zo );
      }
      best = std::min( best, std::chrono::duration<double, std::nano>( std::chrono::steady_clock::now() - t0 ).count() / n );
    }
    printf( "%s %7.1f ns  (%d)\n", c.name, best, out[0] );
  }
  return 0;
}
