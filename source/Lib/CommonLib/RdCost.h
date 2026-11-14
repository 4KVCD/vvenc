/* -----------------------------------------------------------------------------
The copyright in this software is being made available under the Clear BSD
License, included below. No patent rights, trademark rights and/or 
other Intellectual Property Rights other than the copyrights concerning 
the Software are granted under this license.

The Clear BSD License

Copyright (c) 2019-2026, Fraunhofer-Gesellschaft zur Förderung der angewandten Forschung e.V. & The VVenC Authors.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted (subject to the limitations in the disclaimer below) provided that
the following conditions are met:

     * Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.

     * Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.

     * Neither the name of the copyright holder nor the names of its
     contributors may be used to endorse or promote products derived from this
     software without specific prior written permission.

NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.


------------------------------------------------------------------------------------------- */
/** \file     RdCost.h
    \brief    RD cost computation classes (header)
*/

#pragma once

#include "CommonDef.h"
#include "Mv.h"
#include "Unit.h"
#include "Slice.h"

#include <math.h>

//! \ingroup CommonLib
//! \{

namespace vvenc {

struct Picture;
  
#if defined(TARGET_SIMD_X86)  && ENABLE_SIMD_OPT_DIST
using namespace x86_simd;
#endif
#if defined(TARGET_SIMD_ARM)  && ENABLE_SIMD_OPT_DIST
using namespace arm_simd;
#endif

class DistParam;

// ====================================================================================================================
// Type definition
// ====================================================================================================================

// for function pointer
typedef Distortion( *FpDistFunc   )( const DistParam& );
typedef void      ( *FpDistFuncX5 )( const DistParam&, Distortion*, bool );
// The SADs of four reference positions. A result above maximumDistortionForEarlyExit may be a partial sum.
typedef void      ( *FpSadFourPositions )( const DistParam&, const Pel* const*, Distortion* );

// psy-rd measures a block's texture energy on the FINE band of its unnormalized Walsh-Hadamard transform: in an 8x8 block
// the coefficients (u, v) with seq(u) + seq(v) >= 4, in a 4x4 block >= 2, where seq is the number of sign changes along the
// transform's row. Grain and fine texture live there. Against psy on every AC coefficient at the same strength (PsyRD 20,
// preset fast) the band needs 26 to 52 % fewer bytes for the same finest-octave grain energy on film (nocturne_grain,
// wrath_of_man) and keeps the flat inter QP's steadiness. The higher edge 6 (seq sum >= 6, at PsyRD 12) kept 1.3 to 1.7 dB more
// luma PSNR at equal size, but it needs 37 % to 3.3x the bytes of psy on every AC coefficient for the same grain energy and
// pulses with the GOP. The butterflies give the natural (Sylvester) order, not the sequency order: row r's
// sign at column c is the parity of popcount( r & c ), so rows 0..7 of the 8x8 transform change sign 0, 7, 3, 4, 1, 6, 2, 5
// times and rows 0..3 of the 4x4 one 0, 3, 1, 2.
constexpr int hadamardRowSequency( int row, int n )
{
  int changes = 0, prevParity = 0;   // column 0 is + in every row
  for( int col = 1; col < n; col++ )
  {
    int parity = 0;
    for( int bits = row & col; bits; bits &= bits - 1 )   // a loop: MSVC has no constexpr popcount
    {
      parity ^= 1;
    }
    changes += parity != prevParity;
    prevParity = parity;
  }
  return changes;
}

// the band as a mask over the transform's coefficients: keep[v * n + u] is -1 (all bits) inside it, 0 outside (DC always)
struct PsyBandMask
{
  alignas( 32 ) int32_t keep[64];
};

constexpr PsyBandMask psyBandMask( int n, int minSequency )
{
  PsyBandMask mask{};
  for( int v = 0; v < n; v++ )
    for( int u = 0; u < n; u++ )
      mask.keep[v * n + u] = hadamardRowSequency( u, n ) + hadamardRowSequency( v, n ) >= minSequency ? -1 : 0;
  return mask;
}

static constexpr PsyBandMask PSY_BAND_8x8 = psyBandMask( 8, 4 );

// The band in groups: an orientation (the sequency along one axis at least twice the other's: 0 or 1; else mixed: 2) and,
// for the 8x8 transform, a fineness ring (sequency sum under 8: + 0, from 8: + 3). Film grain spreads its energy over all
// groups; a stripe (a few long cosines of a large transform, a directional prediction) fills one orientation. A block
// whose band energy equals the source's can therefore still differ from it group by group: psy-rd counts that difference
// with weight PSY_GROUP_WEIGHT beside the band-energy difference. On film grain this keeps what the energy difference
// alone keeps (gangster_chase: strokes, patchwork, pulsing and missing grain unchanged at equal size); on clean video it
// removes a quarter of the stripes the energy difference alone rewards (wrath_of_man: -24 % at equal size). Higher
// weights keep less grain on film (0.5, 0.75 and 1 measured); comparing coefficient by coefficient, or choosing the
// measure by the source block's own band, loses the grain of flat areas.
// keep[g][v * n + u] is -1 inside group g, 0 outside.
static constexpr int    PSY_NUM_GROUPS   = 6;
static constexpr double PSY_GROUP_WEIGHT = 0.25;

// Grain patchwork ("some blocks have grain, their neighbours do not"): a large block coded with little residual is a
// cheap choice that keeps half the grain while the blocks around it split and keep it; later pictures inherit both. A
// block that lost a fraction f of the source's band energy therefore pays ( 1 + PSY_DEFICIT_CONVEXITY x f ) times its
// band-energy difference: one block at half the grain costs more than two blocks at three quarters, which evens the
// grain out between neighbours. Energy above the source's keeps its plain weight. Where the term acts, psy-rd's base
// strength goes down to PSY_GRAIN_BASE_STRENGTH of its value: the same patchwork, missing grain and pulsing as
// convexity 2 at full strength with fewer block and structure artifacts and half the growth in bytes per QP
// (gangster_chase at equal size, 4 rate points).
static constexpr double PSY_DEFICIT_CONVEXITY   = 4.0;
static constexpr double PSY_GRAIN_BASE_STRENGTH = 0.59375;   // 19 / 32: a value --PsyRD can take
// On clean video the same term buys grain-like energy the source does not have (wrath_of_man at equal size: brush
// strokes, patchwork and flicker about doubled). So it scales with the picture's grain floor (the 10th percentile of
// the source's 8x8 band energies, RdCost::buildPsyOrgEnergies): none up to PSY_GRAIN_FLOOR_NONE, full from
// PSY_GRAIN_FLOOR_FULL.
static constexpr int    PSY_GRAIN_FLOOR_NONE  = 2;   // wrath_of_man: 2 on every picture
static constexpr int    PSY_GRAIN_FLOOR_FULL  = 5;   // gangster_chase: 6 to 24
// ... and with how much of the picture's fine band changes from picture to picture (Picture::psyStableShare):
// animation, graphics and repeated pictures are stable (57-100 % of the blocks in the test clips; the term cost
// detective_conan 14-43 % BD-rate), camera sources are not (0-9 %). Full up to PSY_STABLE_SHARE_FULL, none from _NONE.
static constexpr double PSY_STABLE_SHARE_FULL = 0.15;
static constexpr double PSY_STABLE_SHARE_NONE = 0.40;
// A picture without grain (no term) codes with psy-rd and psy-rdoq at PSY_CLEAN_STRENGTH of their strength at low
// rates: there the tune's kept energy on clean video is partly brush strokes (wrath_of_man at equal size, half
// strength: strokes -48 % and -43 % at the two low rates, down to the default encoder's level, flicker and structure
// about a third lower, for up to 7 points of fine energy, still well above the default encoder's). A quarter adds
// little over a half. At high rates the half strength is a loss (3.2 MB: strokes +40 %, fine energy -4 points), so
// the strength is full up to slice QP PSY_CLEAN_QP_FULL and falls to PSY_CLEAN_STRENGTH at PSY_CLEAN_QP_HALF.
static constexpr double PSY_CLEAN_STRENGTH = 0.5;
static constexpr int    PSY_CLEAN_QP_FULL  = 24;
static constexpr int    PSY_CLEAN_QP_HALF  = 34;

struct PsyGroupMasks
{
  alignas( 32 ) int32_t keep[PSY_NUM_GROUPS][64];
};

constexpr PsyGroupMasks psyGroupMasks( int n, int minSequency )
{
  PsyGroupMasks masks{};
  for( int v = 0; v < n; v++ )
    for( int u = 0; u < n; u++ )
    {
      const int su = hadamardRowSequency( u, n ), sv = hadamardRowSequency( v, n );
      if( su + sv >= minSequency )
      {
        const int g = ( su >= 2 * sv ? 0 : sv >= 2 * su ? 1 : 2 ) + ( n == 8 && su + sv >= 8 ? 3 : 0 );
        masks.keep[g][v * n + u] = -1;
      }
    }
  return masks;
}

static constexpr PsyGroupMasks PSY_GROUPS_8x8 = psyGroupMasks( 8, 4 );
static constexpr PsyGroupMasks PSY_GROUPS_4x4 = psyGroupMasks( 4, 2 );   // three groups: a 4x4 band has one ring
static constexpr int PSY_NUM_GROUPS_4x4 = 3;
static constexpr PsyBandMask PSY_BAND_4x4 = psyBandMask( 4, 2 );   // the same frequencies: a 4x4 row spans half the samples

// ====================================================================================================================
// Class definition
// ====================================================================================================================

/// distortion parameter class
class DistParam
{
public:
  CPelBuf               org;
  CPelBuf               cur;
  FpDistFunc            distFunc  = nullptr;
  FpDistFuncX5          dmvrSadX5 = nullptr;
  FpSadFourPositions    sadFourPositions = nullptr; // SADs of four reference positions; null where no kernel exists
#if ENABLE_MEASURE_SEARCH_SPACE
  FpDistFunc            xDistFunc = nullptr;
#endif
  int                   bitDepth    = 0;
  int                   subShift    = 0;
  ComponentID           compID      = MAX_NUM_COMP;
  bool                  applyWeight = false;     // whether weighted prediction is used or not
  Distortion            maximumDistortionForEarlyExit = MAX_DISTORTION; /// During cost calculations, if distortion exceeds this value, cost calculations may early-terminate.
  const WPScalingParam* wpCur   = nullptr;           // weighted prediction scaling parameters for current ref
  const CPelBuf*        orgLuma = nullptr;

  const Pel*            mask        = nullptr;
  int                   maskStride  = 0;
  int                   stepX       = 0;
  int                   maskStride2 = 0;

  DistParam() = default;

  // org and cur have the same size and the kernels only read org's, so cur's width and height stay 0
  DistParam( const Pel* _org, int _orgStride, const Pel* _cur, int _curStride, int _width, int _height, FpDistFunc _distFunc, int _bitDepth, int _subShift, ComponentID _compID )
    : distFunc(_distFunc), bitDepth(_bitDepth), subShift(_subShift), compID(_compID)
  {
    org.buf    = _org;
    org.stride = _orgStride;
    org.width  = _width;
    org.height = _height;
    cur.buf    = _cur;
    cur.stride = _curStride;
  }
};

/// RD cost computation class
class RdCost
{
public:
  Distortion ( *m_fxdWtdPredPtr )( const DistParam& dp, uint32_t fixedWeight );
  // the bits of getBitsMultiplePredsIBC( x0 + i * xStep, y ) for i = 0 ... count - 1, with the given predictors
  void       ( *m_bvBitsLineIBC )( const Mv* predictors, const int x0, const int xStep, const int count, const int y, const bool useIMV, unsigned* bits );
  // psy-rd: the L1 norm of an 8x8 block's unnormalized Walsh-Hadamard transform over the fine band PSY_BAND_8x8
  int64_t    ( *m_psyBand8x8 )( const Pel* src, ptrdiff_t stride );
  // the same of a 4x4 block, over PSY_BAND_4x4
  int64_t    ( *m_psyBand4x4 )( const Pel* src, ptrdiff_t stride );
  // per group of PSY_GROUPS_8x8: the L1 of the transform coefficients of the 8x8 block a, or a + b summed as Pel when b
  // is given (prediction + residual, unclipped). The six sums add up to m_psyBand8x8 of the same block.
  void       ( *m_psyGroupSums8x8 )( const Pel* a, ptrdiff_t aStride, const Pel* b, ptrdiff_t bStride, int64_t sums[PSY_NUM_GROUPS] );
  // the same of a 4x4 block: the PSY_NUM_GROUPS_4x4 sums of PSY_GROUPS_4x4, which add up to m_psyBand4x4
  void       ( *m_psyGroupSums4x4 )( const Pel* a, ptrdiff_t aStride, const Pel* b, ptrdiff_t bStride, int64_t sums[PSY_NUM_GROUPS_4x4] );

  // for distortion
  FpDistFunc              m_afpDistortFunc[2][DF_TOTAL_FUNCTIONS]; // [eDFunc]
  FpDistFuncX5            m_afpDistortFuncX5[2]; // [eDFunc]
  FpSadFourPositions      m_afpSadFourPositions[8] = {}; // [log2 of the width], SAD at bit depths up to 10 without weights

private:
  vvencCostMode           m_costMode;
  double                  m_distortionWeight[MAX_NUM_COMP]; // only chroma values are used.
  double                  m_dLambda;
  double                  m_dLambda_unadjusted; // TODO: check is necessary
  double                  m_DistScaleUnadjusted;

  std::vector<uint32_t>   m_lumaLevelToWeightPLUT;

  double                  m_chromaWeight;
  ChromaFormat            m_cf;
  double                  m_DistScale;
  double                  m_dLambdaMotionSAD;

  // for motion cost
  Mv                      m_mvPredictor;
  Mv                      m_bvPredictors[2];
  double                  m_motionLambda;
  // The motion search asks for the cost of a bit count at every position it tests. A multiplication and a conversion back take longer than this lookup.
  static const uint32_t   MOTION_COST_TABLE_SIZE = 128;
  Distortion              m_motionCost[MOTION_COST_TABLE_SIZE];
  int                     m_iCostScale;
  double                  m_dCostIBC;
  double                  m_psyStrength = 0;          // psy-rd strength, 0: off
  double                  m_psyRecCredit = 1.0;       // share of the reconstruction's energy psy-rd credits
public:
  RdCost();
  virtual ~RdCost();

  void          create( bool enableOpt = true );

#if defined(TARGET_SIMD_X86)  && ENABLE_SIMD_OPT_DIST
  void          initRdCostX86();
  template <X86_VEXT vext>
  void          _initRdCostX86();
#endif

#if defined(TARGET_SIMD_ARM)  && ENABLE_SIMD_OPT_DIST
  void initRdCostARM();
  template<ARM_VEXT vext>
  void _initRdCostARM();
#endif   // TARGET_SIMD_ARM

  void          setDistortionWeight ( const ComponentID compID, const double distortionWeight ) { m_distortionWeight[compID] = distortionWeight; }
  void          setLambda           ( double dLambda, const BitDepths &bitDepths );
  void          setCostMode         ( vvencCostMode m )                      { m_costMode = m; }

  double        getLambda           ( bool unadj = false )              { return unadj ? m_dLambda_unadjusted : m_dLambda; }
  double        getChromaWeight     ()                                  { return ((m_distortionWeight[COMP_Cb] + m_distortionWeight[COMP_Cr]) / 2.0); }
  double        calcRdCost          ( uint64_t fracBits, Distortion distortion, bool useUnadjustedLambda = true ) const
  {
    return ( useUnadjustedLambda ? m_DistScaleUnadjusted : m_DistScale ) * double( distortion ) + double( fracBits );
  }

  // psy-rd (x264's psy_rd): the texture energy of a block is the L1 norm of its AC Walsh-Hadamard coefficients in 8-bit
  // sample units; the psy distortion of an area is strength * sqrt( lambda ) * the sum over its 8x8 blocks (n x n with n the
  // area's smaller side when that is under 8: 4x4, or 2x2 / 1x1 for ISP slivers) of | energy( source ) -
  // energy( reconstruction ) |. Luma only; strength 0 is off.
  void          setPsyStrength      ( double strength )                 { m_psyStrength = strength; }
  bool          usePsy              ()                            const { return m_psyStrength > 0; }
  // the source energies and the source-edge gate (Picture::psyEdgeGate) of a picture, once before its CTUs are coded
  void          buildPsyOrgEnergies ( Picture& pic, int bitDepth ) const;
  // only this share of the reconstruction's energy counts against the source's (1: all; IntraSearch's angular modes in grain)
  void          setPsyRecCredit     ( double credit )                   { m_psyRecCredit = credit; }
  // of rec, or of rec + resi when resi is given (the transform decisions have the residual but no reconstruction yet)
  Distortion    getPsyDistortion    ( const Picture& pic, const CompArea& area, const CPelBuf& rec, const CPelBuf& resi, int bitDepth ) const;
  Distortion    getPsyDistortion    ( const Picture& pic, const CompArea& area, const CPelBuf& rec, int bitDepth ) const { return getPsyDistortion( pic, area, rec, CPelBuf(), bitDepth ); }
  // the grouped band difference of pred + resi (resi may be empty) to the source over the area's n x n blocks (no
  // strength, lambda or credit): psy-rdoq's per-TU choice compares two quantisations with it
  int64_t       getPsyGroupGap      ( const Picture& pic, const CompArea& area, const CPelBuf& pred, const CPelBuf& resi ) const;
  // the sum over the area's whole 8x8 blocks of | E_org - E_rec | (no strength, lambda or credit): SAO weighs its changes
  int64_t       getPsyEnergyGap     ( const Picture& pic, const CompArea& area, const CPelBuf& rec, int bitDepth ) const;

  void          setDistParam        ( DistParam &rcDP, const CPelBuf& org, const Pel* piRefY , int iRefStride, int bitDepth, ComponentID compID, int subShiftMode = 0, int useHadamard = 0 );
  DistParam     setDistParam        ( const CPelBuf& org, const CPelBuf& cur, int bitDepth, DFunc dfunc );
  DistParam     setDistParam        ( const Pel* pOrg, const Pel* piRefY, int iOrgStride, int iRefStride, int bitDepth, ComponentID compID, int width, int height, int subShift, bool isDMVR = false );
  void          setDistParamGeo     ( DistParam &rcDP, const CPelBuf& org, const Pel *piRefY, int iRefStride, const Pel *mask, int iMaskStride, int stepX, int iMaskStride2, int bitDepth, ComponentID compID );

  double        getMotionLambda     ()                      const { return m_dLambdaMotionSAD; }
  void          selectMotionLambda  ()                            { if( m_motionLambda != getMotionLambda() ) { setMotionLambda( getMotionLambda() ); } }
  void          setPredictor        ( const Mv& rcMv )            { m_mvPredictor = rcMv; }
  void          setCostScale        ( int iCostScale )            { m_iCostScale = iCostScale; }
  Distortion    getCost             ( uint32_t b )          const { return b < MOTION_COST_TABLE_SIZE ? m_motionCost[b] : Distortion( m_motionLambda * b ); }
  // for motion cost
  static uint32_t    xGetExpGolombNumberOfBits( int iVal )
  {
    CHECKD( iVal == std::numeric_limits<int>::min(), "Wrong value" );

#if ENABLE_SIMD_OPT && defined( TARGET_SIMD_X86 )
    // the proper Log2 is not restricted to 0...MAX_CU_SIZE
    return 1 + ( Log2( iVal <= 0 ? ( unsigned( -iVal ) << 1 ) + 1 : unsigned( iVal << 1 ) ) << 1 );
#else
    unsigned uiLength2 = 1, uiTemp2 = ( iVal <= 0 ) ? ( unsigned( -iVal ) << 1 ) + 1 : unsigned( iVal << 1 );

    while( uiTemp2 > MAX_CU_SIZE )
    {
      uiLength2 += ( MAX_CU_DEPTH << 1 );
      uiTemp2  >>=   MAX_CU_DEPTH;
    }

    return uiLength2 + ( Log2(uiTemp2) << 1 );
#endif
  }
  Distortion     getCostOfVectorWithPredictor( const int x, const int y, const unsigned imvShift )  { return getCost( getBitsOfVectorWithPredictor( x, y, imvShift ) ); }
  uint32_t       getBitsOfVectorWithPredictor( const int x, const int y, const unsigned imvShift )  { return getBitsOfVectorHor( x, imvShift ) + getBitsOfVectorVer( y, imvShift ); }
  // the two halves of getBitsOfVectorWithPredictor: each depends on one component only
  uint32_t       getBitsOfVectorHor( const int x, const unsigned imvShift ) const { return xGetExpGolombNumberOfBits( ( ( x * ( 1 << m_iCostScale ) ) - m_mvPredictor.hor ) >> imvShift ); }
  uint32_t       getBitsOfVectorVer( const int y, const unsigned imvShift ) const { return xGetExpGolombNumberOfBits( ( ( y * ( 1 << m_iCostScale ) ) - m_mvPredictor.ver ) >> imvShift ); }

  void           saveUnadjustedLambda ();
  void           setMotionLambda      ( double motionLambda );
  void           setChromaFormat      ( ChromaFormat cf )   { m_cf = cf; }
  void           setPredictorsIBC     (Mv* pcMv)
  {
    for (int i = 0; i < 2; i++)
    {
      m_bvPredictors[i] = pcMv[i];
    }
  }
  void           getMotionCostIBC(int add) { m_dCostIBC = m_dLambdaMotionSAD + add; }
  Distortion     getBvCostMultiplePredsIBC(int x, int y, bool useIMV);
  Distortion     getBvCostIBC         ( const unsigned bits ) const { return Distortion( m_dCostIBC * bits ); }
  // the bits of the block vectors ( x0 + i * xStep, y ), i = 0 ... count - 1, as getBitsMultiplePredsIBC
  void           getBvBitsRowIBC      ( const int x0, const int xStep, const int count, const int y, const bool useIMV, unsigned* bits ) const
  {
    m_bvBitsLineIBC( m_bvPredictors, x0, xStep, count, y, useIMV, bits );
  }
  // the bits of the block vectors ( x, y0 + i * yStep ): the estimate is symmetric in the two components, so this is
  // the row of the transposed vectors
  void           getBvBitsColumnIBC   ( const int x, const int y0, const int yStep, const int count, const bool useIMV, unsigned* bits ) const
  {
    const Mv transposed[2] = { Mv( m_bvPredictors[0].ver, m_bvPredictors[0].hor ), Mv( m_bvPredictors[1].ver, m_bvPredictors[1].hor ) };
    m_bvBitsLineIBC( transposed, y0, yStep, count, x, useIMV, bits );
  }
	
  static Distortion xGetSAD8          ( const DistParam& pcDtParam );
  static Distortion xGetSAD16         ( const DistParam& pcDtParam ); // needs to be public for xGetSAD_MxN_SIMD ( NOTE: they are all public in vvenc )
  static void       xGetSAD16X5       ( const DistParam& pcDtParam, Distortion* cost, bool isCalCentrePos ); // needs to be public for xGetSADX5_16xN_SIMD ( NOTE: they are all public in vvenc )

  static Distortion xGetSSE           ( const DistParam& pcDtParam );
  static Distortion xGetSSE4          ( const DistParam& pcDtParam );
  static Distortion xGetSSE8          ( const DistParam& pcDtParam );
  static Distortion xGetSSE16         ( const DistParam& pcDtParam );
  static Distortion xGetSSE32         ( const DistParam& pcDtParam );
  static Distortion xGetSSE64         ( const DistParam& pcDtParam );
  static Distortion xGetSSE128        ( const DistParam& pcDtParam );


  static Distortion xGetSAD           ( const DistParam& pcDtParam );
  static Distortion xGetSAD4          ( const DistParam& pcDtParam );
  static Distortion xGetSAD32         ( const DistParam& pcDtParam );
  static Distortion xGetSAD64         ( const DistParam& pcDtParam );
  static Distortion xGetSAD128        ( const DistParam& pcDtParam );
  static Distortion xGetSADwMask      ( const DistParam &pcDtParam );
  
  static void       xGetSAD8X5        ( const DistParam& pcDtParam, Distortion* cost, bool isCalCentrePos );
  
  static Distortion xCalcHADs2x2      ( const Pel* piOrg, const Pel* piCur, int iStrideOrg, int iStrideCur );
  static Distortion xGetHAD2SADs      ( const DistParam& pcDtParam );
  static int64_t    xPsyBandL1        ( const Pel* src, ptrdiff_t stride, int n );
  static int64_t    xPsyBand8x8       ( const Pel* src, ptrdiff_t stride ) { return xPsyBandL1( src, stride, 8 ); }
  static int64_t    xPsyBand4x4       ( const Pel* src, ptrdiff_t stride ) { return xPsyBandL1( src, stride, 4 ); }
  static void       xPsyGroupSums8x8  ( const Pel* a, ptrdiff_t aStride, const Pel* b, ptrdiff_t bStride, int64_t sums[PSY_NUM_GROUPS] );
  static void       xPsyGroupSums4x4  ( const Pel* a, ptrdiff_t aStride, const Pel* b, ptrdiff_t bStride, int64_t sums[PSY_NUM_GROUPS_4x4] );
  void              xPsyBlock         ( const Picture& pic, const CPelBuf& org, int x, int y, int xInArea, int yInArea, int n, const CPelBuf& rec, const CPelBuf& resi, double credit, int64_t& recL1, int64_t& groupDiff ) const;
  static int64_t    xPsySliverDiff    ( const Pel* org, ptrdiff_t orgStride, const Pel* rec, ptrdiff_t recStride, const Pel* resi, ptrdiff_t resiStride, int n, double credit );
  int64_t           xPsyEnergy        ( const Pel* src, ptrdiff_t stride, int n, int shift ) const;
  int64_t           xPsyOrgEnergy     ( const Picture& pic, const CPelBuf& org, int x, int y, int xInArea, int yInArea, int n, int shift ) const;
  template<bool fastHad>
  static Distortion xGetHADs          ( const DistParam& pcDtParam );

#if defined(TARGET_SIMD_X86)  && ENABLE_SIMD_OPT_DIST

  template<X86_VEXT vext>
  static Distortion xGetSSE_SIMD    ( const DistParam& pcDtParam );
  template<int iWidth, X86_VEXT vext>
  static Distortion xGetSSE_NxN_SIMD( const DistParam& pcDtParam );

  template<X86_VEXT vext>
  static Distortion xGetSAD_SIMD    ( const DistParam& pcDtParam );
  template<int iWidth, X86_VEXT vext>
  static Distortion xGetSAD_NxN_SIMD( const DistParam& pcDtParam );

  template <X86_VEXT vext>
  static void xGetSADX5_8xN_SIMD    ( const DistParam& rcDtParam, Distortion* cost, bool isCalCentrePos );
  template <X86_VEXT vext>
  static void xGetSADX5_16xN_SIMD_X86   ( const DistParam& rcDtParam, Distortion* cost, bool isCalCentrePos );

  template<X86_VEXT vext, bool fastHad>
  static Distortion xGetHADs_SIMD   ( const DistParam& pcDtParam );
  template<X86_VEXT vext>
  static Distortion xGetHAD2SADs_SIMD( const DistParam &rcDtParam );

  template<X86_VEXT vext> 
  static Distortion xGetSADwMask_SIMD( const DistParam &pcDtParam );
#endif

  unsigned int   getBitsMultiplePredsIBC(int x, int y, bool useIMV);

  Distortion   getDistPart( const CPelBuf& org, const CPelBuf& cur, int bitDepth, const ComponentID compId, DFunc eDFunc );

  void initLumaLevelToWeightTable( int lumaBD );
};// END CLASS DEFINITION RdCost

} // namespace vvenc

//! \}

