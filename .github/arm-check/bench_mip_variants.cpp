// #879 review: the tree's NEON MIP kernels (vld4q_u8, namespace ld4) against two variants without LD4 (uzp: plain loads
// + vuzpq tree; mull: vmull/vmlal per output + pairwise reductions), all checked equal to master's pointers on random
// inputs (10-bit range, weights < 128), then timed: master, ld4, uzp, mull, mull, uzp, ld4, master.
#include "CommonLib/CommonDef.h"
#include "CommonLib/Unit.h"
#include "transpose_neon.h"
#include "sum_neon.h"

#include <arm_neon.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>

using namespace vvenc;
static volatile int sink = 0;

namespace vvenc
{
namespace ld4
{
#include "mip_kernel.inc"
}
#include "mip_variants.inc"
}

typedef void ( *MipFunc )( Pel*, const Pel*, const uint8_t*, const int, const int, bool );

int main()
{
  g_pelBufOP.initPelBufOps( true );
  std::mt19937 rng( 1 );
  alignas( 32 ) uint8_t weight[16][512];
  alignas( 32 ) Pel     input[16][8];
  for( int s = 0; s < 16; s++ )
  {
    for( int i = 0; i < 512; i++ ) weight[s][i] = uint8_t( rng() & 127 );
    for( int i = 0; i < 8; i++ )   input[s][i]  = Pel( int( rng() % 2047 ) - 1023 );
  }
  struct Case { const char* name; MipFunc f[4]; int outputs; };
  const Case cases[] = {
    { "4_4", { g_pelBufOP.mipMatrixMul_4_4, ld4::mipMatrixMul_4_4_neon, uzp::mipMatrixMul_4_4_neon, mull::mipMatrixMul_4_4_neon }, 16 },
    { "8_4", { g_pelBufOP.mipMatrixMul_8_4, ld4::mipMatrixMul_8_4_neon, uzp::mipMatrixMul_8_4_neon, mull::mipMatrixMul_8_4_neon }, 16 },
    { "8_8", { g_pelBufOP.mipMatrixMul_8_8, ld4::mipMatrixMul_8_8_neon, uzp::mipMatrixMul_8_8_neon, mull::mipMatrixMul_8_8_neon }, 64 } };
  const char* names[4] = { "master", "ld4", "uzp", "mull" };
  for( const Case& c : cases )
  {
    for( int t = 0; t < 2; t++ )
    {
      alignas( 32 ) Pel a[64], b[64];
      for( int v = 1; v < 4; v++ )
        for( int s = 0; s < 16; s++ )
        {
          c.f[0]( a, input[s], weight[s], 1023, 512, t );
          c.f[v]( b, input[s], weight[s], 1023, 512, t );
          if( memcmp( a, b, c.outputs * sizeof( Pel ) ) ) { printf( "DIFFERENT %s %s transpose %d set %d\n", names[v], c.name, t, s ); return 1; }
        }
      const int calls = 4000000;
      auto run = [&]( MipFunc f )
      {
        const auto t0 = std::chrono::steady_clock::now();
        for( int i = 0; i < calls; i++ )
        {
          f( a, input[i & 15], weight[i & 15], 1023, 512, t );
          sink = sink + a[i & 15];
        }
        return std::chrono::duration<double, std::nano>( std::chrono::steady_clock::now() - t0 ).count() / calls;
      };
      double ns[4] = {};
      for( int v = 0; v < 4; v++ ) run( c.f[v] );
      for( int v = 0; v < 4; v++ ) ns[v] += run( c.f[v] );
      for( int v = 3; v >= 0; v-- ) ns[v] += run( c.f[v] );
      printf( "mipMatrixMul_%s transpose %d  master %6.2f  ld4 %6.2f  uzp %6.2f (%+5.1f %%)  mull %6.2f (%+5.1f %%) ns; %% = faster than ld4\n",
              c.name, t, ns[0] / 2, ns[1] / 2, ns[2] / 2, 100 * ( ns[1] / ns[2] - 1 ), ns[3] / 2, 100 * ( ns[1] / ns[3] - 1 ) );
    }
  }
  return 0;
}
