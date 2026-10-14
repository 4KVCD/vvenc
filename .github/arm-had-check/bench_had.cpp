// Micro-benchmark of VVenC's Hadamard (SATD) kernels, called through RdCost's function table like the encoder does.
// usage: bench_had <SIMD extension> [calls per kernel]   Prints nanoseconds per call for each block size and a
// checksum of the results (equal checksums mean the kernels return the same values on the same input).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "vvenc/vvenc.h"
#include "CommonLib/RdCost.h"

using namespace vvenc;

int main( int argc, char** argv )
{
  const char* simd = vvenc_set_SIMD_extension( argc > 1 ? argv[1] : "" );
  if( !simd )
  {
    printf( "%s is not supported\n", argv[1] );
    return 1;
  }
  const long calls = argc > 2 ? atol( argv[2] ) : 2000000;

  RdCost rdCost;
  rdCost.create( /*enableOpt=*/true );

  // two 256x256 10-bit planes of random samples, 64-sample stride padding like a picture buffer
  const int stride = 256 + 64;
  std::vector<Pel> org( stride * 256 ), cur( stride * 256 );
  srand( 1 );
  for( auto& v : org ) v = Pel( rand() & 1023 );
  for( size_t i = 0; i < cur.size(); i++ ) cur[i] = Pel( std::min( 1023, std::max( 0, org[i] + ( rand() % 64 ) - 32 ) ) );

  printf( "SIMD=%s\n", simd );
  const int sizes[][3] = { { 8, 4, 0 }, { 4, 8, 0 }, { 8, 8, 0 }, { 16, 8, 0 }, { 8, 16, 0 }, { 16, 16, 0 }, { 32, 32, 0 }, { 32, 32, 1 }, { 64, 64, 1 } };   // width, height, fast
  for( auto& wh : sizes )
  {
    const int width = wh[0], height = wh[1], fast = wh[2];
    DistParam dp;
    dp.org.stride  = stride;
    dp.cur.stride  = stride;
    dp.org.width   = width;
    dp.org.height  = height;
    dp.bitDepth    = 10;
    dp.applyWeight = 0;
    const FpDistFunc func = rdCost.m_afpDistortFunc[0][( fast ? DF_HAD_fast : DF_HAD ) + floorLog2( width )];

    uint64_t checksum = 0;
    const auto start = std::chrono::steady_clock::now();
    for( long i = 0; i < calls; i++ )
    {
      // blocks at changing positions in the planes, so the input stays cached but differs from call to call
      const int x = ( i * 37 ) % ( 256 - width ), y = ( i * 11 ) % ( 256 - height );
      dp.org.buf = &org[y * stride + x];
      dp.cur.buf = &cur[( ( y + 3 ) % ( 256 - height ) ) * stride + x];
      checksum += func( dp );
    }
    const double ns = std::chrono::duration<double, std::nano>( std::chrono::steady_clock::now() - start ).count() / calls;
    printf( "HAD%s %2dx%-2d  %7.2f ns/call  checksum %llu\n", fast ? "_fast" : "     ", width, height, ns, (unsigned long long) checksum );
  }
  return 0;
}
