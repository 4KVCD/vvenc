// N32 real-search check of #864: MCTF's motion search on real pictures through one library's own code, with the column
// kernel (m_motionErrorLumaFracColumn as the build sets it) and without it (nullptr: master's per-vector path with its
// early exits). Pictures: crowd_run 1080p (8-bit scaled to 10-bit), current pictures 10 / 30 / 50, references at -2, -1,
// +1, +2. The search levels run as motionEstimationMCTF runs them at 1080p (extra 1/8 level); only the last level
// (doubleRes, the one with the subpel search) is timed. Every block's vector and error must be equal with and without
// the column kernel. Rounds A B B A, per MCTFSpeed 2 / 3 / 4 (the presets that use the 4-tap low-res search).
#include "CommonLib/MCTF.h"
#include "EncoderLib/EncCfg.h"

#include <chrono>
#include <exception>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace vvenc;

struct Pair
{
  PelStorage orig, ref, origSub2, origSub4, origSub8;
};

// The luma plane of an 8-bit picture as 10-bit samples, with the frame margin MCTF reads
static void toLuma( const std::vector<uint8_t>& y8, int W, int H, PelStorage& p )
{
  p.create( CHROMA_400, Area( 0, 0, W, H ), 0, MCTF_FRAME_MARGIN );
  for( int y = 0; y < H; y++ )
    for( int x = 0; x < W; x++ )
      p.Y().at( x, y ) = Pel( y8[y * W + x] << 2 );
  p.extendBorderPel( MCTF_FRAME_MARGIN );
}

// MCTF::subsampleLuma (private): 2x2 average, MCTF_PADDING border
static void subsample( const PelStorage& in, PelStorage& out )
{
  const int w = in.Y().width / 2, h = in.Y().height / 2;
  out.create( CHROMA_400, Area( 0, 0, w, h ), 0, MCTF_PADDING );
  g_pelBufOP.downsample2x2( in.Y().buf, in.Y().stride, out.Y().buf, out.Y().stride, w, h );
  out.extendBorderPel( MCTF_PADDING, MCTF_PADDING );
}

static int benchMain();

int main()
{
  try
  {
    return benchMain();
  }
  catch( const std::exception& e )
  {
    printf( "exception: %s\n", e.what() );
    return 1;
  }
}

static int benchMain()
{
  // a 1920x1080 4:2:0 8-bit y4m stream on stdin, read up to the last picture used
  char hdr[256];
  if( !fgets( hdr, sizeof( hdr ), stdin ) || !strstr( hdr, " W1920 H1080" ) ) { printf( "need a 1080p y4m on stdin\n" ); return 1; }
  const int W = 1920, H = 1080;
  std::vector<std::vector<uint8_t>> luma( 53 );
  std::vector<uint8_t> chroma( W * H / 2 );
  for( int n = 0; n < 53; n++ )
  {
    char line[64];
    luma[n].resize( W * H );
    if( !fgets( line, sizeof( line ), stdin ) || fread( luma[n].data(), 1, W * H, stdin ) != size_t( W * H ) || fread( chroma.data(), 1, chroma.size(), stdin ) != chroma.size() )
    {
      printf( "short stream at picture %d\n", n );
      return 1;
    }
  }

  std::vector<Pair> pairs;
  pairs.reserve( 12 );   // PelStorage must not be copied: no reallocation
  for( int cur : { 10, 30, 50 } )
    for( int d : { -2, -1, 1, 2 } )
    {
      pairs.emplace_back();
      toLuma( luma[cur], W, H, pairs.back().orig );
      toLuma( luma[cur + d], W, H, pairs.back().ref );
    }

  for( int speed : { 2, 3, 4 } )
  {
    VVEncCfg cfg;
    vvenc_init_default( &cfg, W, H, 50, 0, 32, VVENC_MEDIUM );
    cfg.m_internalBitDepth[0] = cfg.m_internalBitDepth[1] = 10;
    vvenc_init_config_parameter( &cfg );
    cfg.m_vvencMCTF.MCTFSpeed = speed;
    MCTF mctf;
    mctf.init( cfg, true, nullptr );
    const int us = cfg.m_vvencMCTF.MCTFUnitSize;
    auto column = mctf.m_motionErrorLumaFracColumn;
    if( speed == 2 ) printf( "simd column kernel %s, unit size %d, %zu picture pairs\n", column ? "set" : "null", us, pairs.size() );

    // the levels before the subpel level, as motionEstimationMCTF (addLevel at 1080p), once per pair
    std::vector<Array2D<MotionVector>> mv2( pairs.size() );
    for( size_t i = 0; i < pairs.size(); i++ )
    {
      Pair& p = pairs[i];
      p.origSub2.destroy();
      p.origSub4.destroy();
      p.origSub8.destroy();
      subsample( p.orig, p.origSub2 );
      subsample( p.origSub2, p.origSub4 );
      subsample( p.origSub4, p.origSub8 );
      PelStorage r2, r4, r8;
      subsample( p.ref, r2 );
      subsample( r2, r4 );
      subsample( r4, r8 );
      Array2D<MotionVector> mvm( W / ( us * 16 ) + 1, H / ( us * 16 ) + 1 ), mv0( W / ( us * 8 ) + 1, H / ( us * 8 ) + 1 ),
                            mv1( W / ( us * 4 ) + 1, H / ( us * 4 ) + 1 );
      mv2[i].allocate( W / ( us * 2 ) + 1, H / ( us * 2 ) + 1 );
      mctf.motionEstimationLuma( mvm, p.origSub8, r8, MCTF_PADDING, 2 * us );
      mctf.motionEstimationLuma( mv0, p.origSub4, r4, MCTF_PADDING, 2 * us, &mvm, 2 );
      mctf.motionEstimationLuma( mv1, p.origSub2, r2, MCTF_PADDING, 2 * us, &mv0, 2 );
      mctf.motionEstimationLuma( mv2[i], p.orig, p.ref, MCTF_FRAME_MARGIN, 2 * us, &mv1, 2 );
    }

    const int wb = ( W + us - 1 ) / us, hb = ( H + us - 1 ) / us;
    auto run = [&]( bool withColumn, std::vector<Array2D<MotionVector>>& out )
    {
      mctf.m_motionErrorLumaFracColumn = withColumn ? column : nullptr;
      const auto t0 = std::chrono::steady_clock::now();
      for( size_t i = 0; i < pairs.size(); i++ )
      {
        out[i].allocate( wb, hb );
        mctf.motionEstimationLuma( out[i], pairs[i].orig, pairs[i].ref, MCTF_FRAME_MARGIN, us, &mv2[i], 1, true );
      }
      return std::chrono::duration<double, std::milli>( std::chrono::steady_clock::now() - t0 ).count();
    };

    std::vector<Array2D<MotionVector>> a( pairs.size() ), b( pairs.size() );
    run( true, a );   // warm-up
    double ta = 0, tb = 0;
    const int rounds = 3;
    for( int r = 0; r < rounds; r++ )
    {
      ta += run( true, a );
      tb += run( false, b );
      tb += run( false, b );
      ta += run( true, a );
    }
    ta /= 2 * rounds;
    tb /= 2 * rounds;

    long blocks = 0, frac = 0;
    for( size_t i = 0; i < pairs.size(); i++ )
      for( int y = 0; y < hb; y++ )
        for( int x = 0; x < wb; x++ )
        {
          const MotionVector &u = a[i].get( x, y ), &v = b[i].get( x, y );
          if( u.x != v.x || u.y != v.y || u.error != v.error )
          {
            printf( "MISMATCH speed %d pair %zu block %d,%d: %d,%d,%d vs %d,%d,%d\n", speed, i, x, y, u.x, u.y, u.error, v.x, v.y, v.error );
            return 1;
          }
          blocks++;
          frac += ( u.x & 15 ) || ( u.y & 15 );
        }
    printf( "MCTFSpeed %d: last level (integer + subpel search) %8.2f ms without column, %8.2f ms with column per %zu pairs: %+.1f %% (%ld blocks equal, %.0f %% fractional)\n",
            speed, tb, ta, pairs.size(), 100 * ( tb / ta - 1 ), blocks, 100.0 * frac / blocks );
  }
  return 0;
}
