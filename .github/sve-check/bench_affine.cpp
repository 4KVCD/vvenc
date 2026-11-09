// Bench for B10 (affine luma CU on Arm, no PROF): xPredAffineBlk's per-sub-block loop through m_if's 4x4 / Hor / Ver
// filter pointers (what master runs on Arm) against xFpAffineLumaCU of the tree. Equality first, then A B B A timing
// per CU at realistic affine MV spreads.
#include "CommonLib/CommonDef.h"
#include "CommonLib/Unit.h"
#include "CommonLib/InterPrediction.h"

#include "CommonLib/arm/CommonDefARM.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace vvenc;

static volatile int sink = 0;

// As vvenc_unit_test's affinePROF4x4_unfused: border into a 6x6 window, then the gradient filter and PROF pointers.
static void prof4x4( InterPredInterpolation& ip, Pel* dst, int dstStride, const Pel* pred, int predStride, const Pel* ref,
                     int refStride, const InterPredInterpolation::AffineLumaCU& cu, bool bi )
{
  const int extStride = 8, gradStride = 6;
  Pel ext[6 * extStride], gradX[6 * gradStride], gradY[6 * gradStride];
  int dMvX[16], dMvY[16];
  for( int y = -1; y < 5; y++ )
    for( int x = -1; x < 5; x++ )
    {
      const bool inside = x >= 0 && x < 4 && y >= 0 && y < 4;
      ext[( y + 1 ) * extStride + x + 1] = inside ? pred[y * predStride + x] : Pel( ( ref[y * refStride + x] << cu.shiftNum ) - IF_INTERNAL_OFFS );
    }
  for( int i = 0; i < 16; i++ ) { dMvX[i] = cu.dMvXY[2 * i]; dMvY[i] = cu.dMvXY[2 * i + 1]; }
  ip.xFpProfGradFilter( ext, extStride, 6, 6, gradStride, gradX, gradY, cu.clpRng.bd );
  ip.xFpApplyPROF( dst, dstStride, ext + extStride + 1, extStride, 4, 4, gradX + gradStride + 1, gradY + gradStride + 1,
                   gradStride, dMvX, dMvY, 4, bi, cu.shiftNum, cu.offset, cu.clpRng );
}

// As vvenc_unit_test's affineLumaCU_perSubblock: the loop of xPredAffineBlk for luma.
static void perSubblock( InterPredInterpolation& ip, InterpolationFilter& filter, const InterPredInterpolation::AffineLumaCU& cu, bool prof, bool bi )
{
  const bool isLast = !prof && !bi;
  Pel ext[8 * 6];
  for( int h = 0; h < cu.height; h += 4 )
  {
    for( int w = 0; w < cu.width; w += 4 )
    {
      const int x = 2 + w, y = 2 + h;
      int mvHor = cu.mvScaleHor + cu.dMvHorX * x + cu.dMvVerX * y;
      int mvVer = cu.mvScaleVer + cu.dMvHorY * x + cu.dMvVerY * y;
      roundAffineMv( mvHor, mvVer, cu.roundShift );
      Mv mv( mvHor, mvVer );
      mv.clipToStorageBitDepth();
      cu.storedMv[( h / 4 ) * cu.storedMvStride + w / 4] = mv;
      mvHor = std::min( cu.horMax, std::max( cu.horMin, mv.hor ) );
      mvVer = std::min( cu.verMax, std::max( cu.verMin, mv.ver ) );
      const int  xFrac = mvHor & 15, yFrac = mvVer & 15;
      const Pel* ref   = cu.ref + ( h + ( mvVer >> 4 ) ) * cu.refStride + w + ( mvHor >> 4 );
      Pel*       dst   = prof ? ext + 8 + 1 : cu.dst + h * cu.dstStride + w;
      const int  dstStride = prof ? 8 : cu.dstStride;
      if( xFrac && yFrac )
        filter.filter4x4( COMP_Y, ref, cu.refStride, dst, dstStride, 4, 4, xFrac, yFrac, isLast, CHROMA_420, cu.clpRng );
      else if( !yFrac )
        filter.filterHor( COMP_Y, ref, cu.refStride, dst, dstStride, 4, 4, xFrac, isLast, CHROMA_420, cu.clpRng );
      else
        filter.filterVer( COMP_Y, ref, cu.refStride, dst, dstStride, 4, 4, yFrac, true, isLast, CHROMA_420, cu.clpRng );
      if( prof )
        prof4x4( ip, cu.dst + h * cu.dstStride + w, cu.dstStride, dst, 8, ref + ( yFrac >> 3 ) * cu.refStride + ( xFrac >> 3 ), cu.refStride, cu, bi );
    }
  }
}

int main()
{
  InterPredInterpolation ip;
  ip.init( /*enableOpt=*/true );
  InterpolationFilter filter;
  filter.initInterpolationFilter( true );
  printf( "simd level: %s\n", read_arm_extension_name().c_str() );
  if( !ip.xFpAffineLumaCU[0][0] ) { printf( "no CU kernel\n" ); return 1; }
  int16_t dMvXY[32];
  for( auto& d : dMvXY ) d = int16_t( rand() % 63 - 31 );

  const int picSize = 256, margin = 144, stride = picSize + 2 * margin;
  std::vector<Pel> pic( stride * ( picSize + 2 * margin ) );
  std::vector<Pel> dA( 128 * 128 ), dB( 128 * 128 );
  std::vector<Mv>  mA( 32 * 32 ), mB( 32 * 32 );
  srand( 5 );
  bool same = true;
  int  cases = 0;
  const int sizes[][2] = { { 8, 8 }, { 16, 8 }, { 16, 16 }, { 32, 16 }, { 32, 32 }, { 64, 64 }, { 128, 128 } };
  for( int bd : { 8, 10 } )
  {
    for( auto& p : pic ) p = Pel( rand() & ( ( 1 << bd ) - 1 ) );
    for( int prof = 0; prof < 2; prof++ )
    for( bool bi : { false, true } )
    {
      if( !ip.xFpAffineLumaCU[prof][bi] ) continue;
      for( auto& s : sizes )
      {
        const int width = s[0], height = s[1];
        const int log2W = floorLog2( width ), log2H = floorLog2( height );
        // 64 CUs with random affine models: corner MVs within 16 samples, control-point differences within 2 samples
        std::vector<InterPredInterpolation::AffineLumaCU> cus( 64 );
        for( auto& cu : cus )
        {
          const int x0 = ( rand() % ( picSize - width ) ) & ~7, y0 = ( rand() % ( picSize - height ) ) & ~7;
          const Mv mvLT( rand() % 513 - 256, rand() % 513 - 256 );
          const Mv d1( rand() % 65 - 32, rand() % 65 - 32 ), d2( rand() % 65 - 32, rand() % 65 - 32 );
          cu.ref = pic.data() + ( margin + y0 ) * stride + margin + x0;
          cu.refStride = stride; cu.dstStride = 128; cu.width = width; cu.height = height;
          cu.dMvHorX = d1.hor * ( 1 << ( MAX_CU_DEPTH - log2W ) );
          cu.dMvHorY = d1.ver * ( 1 << ( MAX_CU_DEPTH - log2W ) );
          cu.dMvVerX = d2.hor * ( 1 << ( MAX_CU_DEPTH - log2H ) );
          cu.dMvVerY = d2.ver * ( 1 << ( MAX_CU_DEPTH - log2H ) );
          cu.mvScaleHor = mvLT.hor * ( 1 << MAX_CU_DEPTH );
          cu.mvScaleVer = mvLT.ver * ( 1 << MAX_CU_DEPTH );
          cu.roundShift = MAX_CU_DEPTH - 4 + MV_FRACTIONAL_BITS_INTERNAL;
          cu.horMin = ( -128 - 8 - x0 + 1 ) * 16; cu.horMax = ( picSize + 8 - x0 - 1 ) * 16;
          cu.verMin = ( -128 - 8 - y0 + 1 ) * 16; cu.verMax = ( picSize + 8 - y0 - 1 ) * 16;
          cu.storedMvStride = 32; cu.dMvXY = dMvXY; cu.shiftNum = std::max( 2, IF_INTERNAL_PREC - bd ); cu.offset = ( 1 << ( cu.shiftNum - 1 ) ) + IF_INTERNAL_OFFS;
          cu.clpRng = ClpRng{ bd };
        }
        for( auto cu : cus )
        {
          InterPredInterpolation::AffineLumaCU a = cu, b = cu;
          a.dst = dA.data(); a.storedMv = mA.data(); b.dst = dB.data(); b.storedMv = mB.data();
          perSubblock( ip, filter, a, prof, bi );
          ip.xFpAffineLumaCU[prof][bi]( b );
          for( int y = 0; y < height; y++ ) same &= !memcmp( &dA[y * 128], &dB[y * 128], width * sizeof( Pel ) );
          for( int y = 0; y < height / 4; y++ ) same &= !memcmp( &mA[y * 32], &mB[y * 32], width / 4 * sizeof( Mv ) );
          cases++;
        }
        auto runA = [&]( int n ) {
          const auto t0 = std::chrono::steady_clock::now();
          for( int i = 0; i < n; i++ ) { auto c = cus[i & 63]; c.dst = dA.data(); c.storedMv = mA.data(); perSubblock( ip, filter, c, prof, bi ); sink = sink + dA[0]; }
          return std::chrono::duration<double, std::nano>( std::chrono::steady_clock::now() - t0 ).count() / n;
        };
        auto runB = [&]( int n ) {
          const auto t0 = std::chrono::steady_clock::now();
          for( int i = 0; i < n; i++ ) { auto c = cus[i & 63]; c.dst = dB.data(); c.storedMv = mB.data(); ip.xFpAffineLumaCU[prof][bi]( c ); sink = sink + dB[0]; }
          return std::chrono::duration<double, std::nano>( std::chrono::steady_clock::now() - t0 ).count() / n;
        };
        const int n = std::max( 2000, 4000000 / ( width * height ) );
        runA( n / 4 ); runB( n / 4 );
        const double a1 = runA( n ), b1 = runB( n ), b2 = runB( n ), a2 = runA( n );
        const double a = ( a1 + a2 ) / 2, b = ( b1 + b2 ) / 2;
        printf( "bd %2d prof %d bi %d %3dx%-3d  per-sub-block %8.1f ns  CU kernel %8.1f ns  %5.1f %% faster\n", bd, prof, bi, width, height, a, b, ( a / b - 1 ) * 100 );
      }
    }
  }
  printf( "%s on %d CUs\n", same ? "equal" : "DIFFERENT", cases );
  return same ? 0 : 1;
}
