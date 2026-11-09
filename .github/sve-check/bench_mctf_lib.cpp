// MCTF integer motion error through the LINKED LIBRARY's own kernel pointer (MCTF(true).m_motionErrorLumaInt8), so the
// kernel runs with the library's code layout as shipped. Link once per library (master, #909, N8b) and compare.
// Input: two 1920x1080 8-bit luma planes on stdin (four pictures apart), scaled to 10 bit, borders replicated.
// An emulated integer search per block (raster over +-range, running best, only error < best replaces it), as in
// MCTF::motionEstimationLuma. Prints the best of 5 search times per block size and a checksum of every block's chosen
// vector and error (must match between the libraries).
#include "CommonLib/CommonDef.h"
#include "CommonLib/MCTF.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

using namespace vvenc;

int main()
{
  const int W = 1920, H = 1080, M = 80, S = W + 2 * M;
  std::vector<uint8_t> y8( W * H );
  std::vector<Pel> pic[2];
  for( int p = 0; p < 2; p++ )
  {
    if( fread( y8.data(), 1, W * H, stdin ) != size_t( W * H ) ) { printf( "short input\n" ); return 1; }
    pic[p].assign( S * ( H + 2 * M ), 0 );
    for( int y = -M; y < H + M; y++ )
      for( int x = -M; x < W + M; x++ )
      {
        const int cy = y < 0 ? 0 : y >= H ? H - 1 : y, cx = x < 0 ? 0 : x >= W ? W - 1 : x;
        pic[p][( y + M ) * S + x + M] = Pel( y8[cy * W + cx] << 2 );
      }
  }
  const Pel* org = pic[1].data() + M * S + M;
  const Pel* ref = pic[0].data() + M * S + M;

  MCTF mctf( true );
  const auto fn = mctf.m_motionErrorLumaInt8;
  uint64_t checksum = 0;
  for( int bs : { 8, 16, 32 } )
  {
    const int range = bs == 32 ? 8 : 6;
    double best = 1e30;
    for( int round = 0; round < 5; round++ )
    {
      uint64_t sum = 0;
      auto t0 = std::chrono::steady_clock::now();
      for( int by = 0; by + bs <= H; by += bs )
        for( int bx = 0; bx + bs <= W; bx += bs )
        {
          int be = 0x7fffffff, bv = 0;
          for( int dy = -range; dy <= range; dy++ )
            for( int dx = -range; dx <= range; dx++ )
            {
              const int e = fn( org + by * S + bx, S, ref + ( by + dy ) * S + bx + dx, S, bs, bs, be );
              if( e < be ) { be = e; bv = dy * 64 + dx; }
            }
          sum = sum * 31 + uint64_t( be ) * 4099 + uint64_t( bv + 1000 );
        }
      const double ms = std::chrono::duration<double, std::milli>( std::chrono::steady_clock::now() - t0 ).count();
      best = ms < best ? ms : best;
      if( round == 0 ) checksum = checksum * 1000003 + sum;
    }
    printf( "%2dx%-2d %8.2f ms\n", bs, bs, best );
  }
  printf( "checksum %016llx\n", ( unsigned long long ) checksum );
  return 0;
}
