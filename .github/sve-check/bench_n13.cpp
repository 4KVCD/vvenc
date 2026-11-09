// Bench for N13 (NEON equal-bins ALF statistics kernel), linked with the N13 tree's library. Per 4-bin block, as
// EncAdaptiveLoopFilter::getPreBlkStats dispatches it:
//   before (master on Arm): m_getPreBlkStatsAccum (the NEON 16-bin-pair kernel) for every block;
//   after: getEqualBinsMask (copied from EncAdaptiveLoopFilter.cpp), then m_getPreBlkStatsAccumEqualBins when at most
//          two distinct bins are left (popcount(mask) >= 2: masks 3, 5, 6, 7), else m_getPreBlkStatsAccum.
// For each mask class: best-of-5 ns per block, and a checksum of the statistics of before and after (must match).
#include "CommonLib/CommonDef.h"
#include "EncoderLib/EncAdaptiveLoopFilter.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace vvenc;

#define NL_COVAR_bstride ( MAX_NUM_ALF_LUMA_COEFF << 4 )
#define NL_COVAR_kstride ( 1 << 4 )
#define NL_COVAR_xstride ( 1 )
#define GET_ALF_COVAR( elocal, b, k, x ) elocal[( b ) * NL_COVAR_bstride + ( k ) * NL_COVAR_kstride + ( x ) * NL_COVAR_xstride]

// copied verbatim from EncAdaptiveLoopFilter.cpp (static there)
static int getEqualBinsMask( const Pel* ELocal, const int numCoeff )
{
  constexpr int numBins         = AdaptiveLoopFilter::AlfNumClippingValues;
  const size_t  clippedRowsSize = size_t( numCoeff - 1 ) * NL_COVAR_kstride * sizeof( Pel );
  const auto    binRows         = [ELocal]( int b ) { return &GET_ALF_COVAR( ELocal, b, 0, 0 ); };

  if( memcmp( binRows( 0 ), binRows( numBins - 1 ), clippedRowsSize ) == 0 )
  {
    return ( 1 << ( numBins - 1 ) ) - 1;
  }
  int equalBinsMask = 0;
  for( int b = 0; b + 1 < numBins; b++ )
  {
    if( memcmp( binRows( b ), binRows( b + 1 ), clippedRowsSize ) == 0 )
    {
      equalBinsMask |= 1 << b;
    }
  }
  return equalBinsMask;
}

static volatile float sink = 0;

int main()
{
  constexpr int numBins = AdaptiveLoopFilter::AlfNumClippingValues;
  constexpr int binSize = MAX_NUM_ALF_LUMA_COEFF << 4;
  EncAdaptiveLoopFilter alf( /*enableOpt=*/true );
  std::mt19937 rng( 13 );

  for( int filterSize : { 7, 5 } )
  {
    AlfFilterShape shape( filterSize );
    AlfCovariance covA, covB;
    covA.create( shape.numCoeff, numBins );
    covB.create( shape.numCoeff, numBins );

    for( int mask : { 7, 3, 6, 5, 0, 1, 2, 4 } )
    {
      // 64 blocks with this mask: random 10-bit differences, bins copied where the mask says equal; neighbouring bins
      // that should differ get a different last clipped sample
      constexpr int numBlocks = 64;
      std::vector<Pel> e( numBlocks * numBins * binSize );
      std::vector<Pel> y( numBlocks * 16 );
      for( auto& v : e ) v = Pel( int( rng() % 2047 ) - 1023 );
      for( auto& v : y ) v = Pel( int( rng() % 511 ) - 255 );
      bool maskOk = true;
      for( int n = 0; n < numBlocks; n++ )
      {
        Pel* blk = &e[n * numBins * binSize];
        for( int b = 0; b + 1 < numBins; b++ )
        {
          if( ( mask >> b ) & 1 )
            std::memcpy( blk + ( b + 1 ) * binSize, blk + b * binSize, binSize * sizeof( Pel ) );
          else
            blk[( b + 1 ) * binSize + ( shape.numCoeff - 2 ) * 16 + 15] = Pel( blk[b * binSize + ( shape.numCoeff - 2 ) * 16 + 15] + 1 );
        }
        maskOk = maskOk && getEqualBinsMask( blk, shape.numCoeff ) == mask;
      }

      auto before = [&]( AlfCovariance& cov, int n )
      {
        alf.m_getPreBlkStatsAccum( cov, shape, &e[n * numBins * binSize], ( const Pel( * )[4] ) & y[n * 16], numBins );
      };
      auto after = [&]( AlfCovariance& cov, int n )
      {
        const Pel* blk  = &e[n * numBins * binSize];
        const int  m    = getEqualBinsMask( blk, shape.numCoeff );
        const int  ones = ( m & 1 ) + ( ( m >> 1 ) & 1 ) + ( ( m >> 2 ) & 1 );
        if( ones >= 2 )
          alf.m_getPreBlkStatsAccumEqualBins( cov, shape, blk, ( const Pel( * )[4] ) & y[n * 16], m );
        else
          alf.m_getPreBlkStatsAccum( cov, shape, blk, ( const Pel( * )[4] ) & y[n * 16], numBins );
      };

      covA.reset();
      covB.reset();
      for( int n = 0; n < numBlocks; n++ ) { before( covA, n ); after( covB, n ); }
      bool equal = covA.pixAcc == covB.pixAcc && !std::memcmp( covA.y, covB.y, sizeof( AlfCovariance::Ty ) * numBins );
      for( int b = 0; b < numBins; b++ )
        equal = equal && !std::memcmp( covA.E[b], covB.E[b], sizeof( AlfCovariance::TE ) * numBins );

      auto time = [&]( auto fn, AlfCovariance& cov )
      {
        const int reps = 4000;
        auto t0 = std::chrono::steady_clock::now();
        for( int r = 0; r < reps; r++ )
        {
          cov.reset();
          for( int n = 0; n < numBlocks; n++ ) fn( cov, n );
          sink = sink + cov.E[1][2][0][1];
        }
        return std::chrono::duration<double, std::nano>( std::chrono::steady_clock::now() - t0 ).count() / ( reps * numBlocks );
      };
      time( before, covA ); time( after, covB );
      double tb = 1e30, ta = 1e30;
      for( int r = 0; r < 5; r++ )
      {
        tb = std::min( tb, time( before, covA ) );
        ta = std::min( ta, time( after, covB ) );
      }
      printf( "coeff %2d mask %d (%s) maskcheck %s equal %s  before %6.1f ns  after %6.1f ns  %6.1f %% faster\n", shape.numCoeff, mask,
              mask == 7 ? "all equal " : mask == 3 ? "runs 3+1  " : mask == 6 ? "runs 1+3  " : mask == 5 ? "runs 2+2  " : "full+check",
              maskOk ? "ok" : "BAD", equal ? "yes" : "NO", tb, ta, ( tb / ta - 1 ) * 100 );
    }
    covA.destroy();
    covB.destroy();
  }
  return 0;
}
