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


/** \file     EncModeCtrl.cpp
    \brief    Encoder controller for trying out specific modes
*/

#include <cstdio>
#include "EncCu.h"

#include "EncModeCtrl.h"
#include "EncStage.h"
#include "CommonLib/RdCost.h"
#include "CommonLib/CodingStructure.h"
#include "CommonLib/Picture.h"
#include "CommonLib/UnitTools.h"
#include "CommonLib/dtrace_next.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <string>

//! \ingroup EncoderLib
//! \{

namespace vvenc {

void CacheBlkInfoCtrl::create(int ctuSize)
{
  const unsigned numPos = MAX_CU_SIZE >> MIN_CU_LOG2;
  const int maxSizeIdx  = MAX_CU_SIZE_IDX - MIN_CU_LOG2;

  //static constexpr size_t numCu = 7921;

  size_t numCu = 0;
  
  for( int wIdx = 0; wIdx < maxSizeIdx; wIdx++ )
  {
    for( int hIdx = 0; hIdx < maxSizeIdx; hIdx++ )
    {
      for( unsigned y = 0; y < numPos; y++ )
      {
        for( unsigned x = 0; x < numPos; x++ )
        {
          // a block of width W might be offset of N * W + 1/2 W (bcs of TT), same for H
          // W = 1 << ( wIdx + 2 )
          // 1/2 W = 1 << ( wIdx + 1 )
          // remainder of (N+1/2)*W -> x & ( ( 1 << ( wIdx + 1 ) ) - 1 )
  
          if( (x + (1 << (wIdx)) <= (ctuSize >> MIN_CU_LOG2))
              && (y + (1 << (hIdx)) <= (ctuSize >> MIN_CU_LOG2))
              && (((x << MIN_CU_LOG2) & ((1 << (wIdx + MIN_CU_LOG2 - 1)) - 1)) == 0)
              && (((y << MIN_CU_LOG2) & ((1 << (hIdx + MIN_CU_LOG2 - 1)) - 1)) == 0) )
          {
            numCu++;
          }
        }
      }
    }
  }

  m_codedCUInfoBuf = new CodedCUInfo[numCu];
  CodedCUInfo* cuInfo = m_codedCUInfoBuf;

  for( int wIdx = 0; wIdx < maxSizeIdx; wIdx++ )
  {
    for( int hIdx = 0; hIdx < maxSizeIdx; hIdx++ )
    {
      for( unsigned y = 0; y < numPos; y++ )
      {
        for( unsigned x = 0; x < numPos; x++ )
        {
          // a block of width W might be offset of N * W + 1/2 W (bcs of TT), same for H
          // W = 1 << ( wIdx + 2 )
          // 1/2 W = 1 << ( wIdx + 1 )
          // remainder of (N+1/2)*W -> x & ( ( 1 << ( wIdx + 1 ) ) - 1 )

          if(( x + (1<<(wIdx)) <= ( ctuSize >> MIN_CU_LOG2 ) )
            && ( y + (1<<(hIdx)) <= ( ctuSize >> MIN_CU_LOG2 ) )
            && ( ( ( x << MIN_CU_LOG2 ) & ((1 << (wIdx + MIN_CU_LOG2 - 1)) - 1) ) == 0 )
            && ( ( ( y << MIN_CU_LOG2 ) & ((1 << (hIdx + MIN_CU_LOG2 - 1)) - 1) ) == 0 ) )
          {
            m_codedCUInfo[wIdx][hIdx][x][y] = cuInfo++;
            m_codedCUInfo[wIdx][hIdx][x][y]->poc       = -1;
            m_codedCUInfo[wIdx][hIdx][x][y]->ctuRsAddr = -1;
          }
          else
          {
            m_codedCUInfo[wIdx][hIdx][x][y] = nullptr;
          }
        }
      }
    }
  }
}

void CacheBlkInfoCtrl::destroy()
{
  delete[] m_codedCUInfoBuf;
  m_codedCUInfoBuf = nullptr;
}

void CacheBlkInfoCtrl::init( const Slice &slice )
{
  m_pcv = slice.pps->pcv;
}

void CacheBlkInfoCtrl::initBlk( const UnitArea& area, int poc )
{
  unsigned idx1, idx2, idx3, idx4;
  getAreaIdxNew( area.Y(), *m_pcv, idx1, idx2, idx3, idx4 );

  const int ctuRsAddr = getCtuAddr( area.lumaPos(), *m_pcv );
  CodedCUInfo* cuInfo = m_codedCUInfo[idx1][idx2][idx3][idx4];

  if( cuInfo->poc != poc || cuInfo->ctuRsAddr != ctuRsAddr )
  {
    GCC_WARNING_DISABLE_class_memaccess
    memset( cuInfo, 0, sizeof( CodedCUInfo ) );
    GCC_WARNING_RESET

    cuInfo->poc       = poc;
    cuInfo->ctuRsAddr = ctuRsAddr;
  }
}

CodedCUInfo& CacheBlkInfoCtrl::getBlkInfo( const UnitArea& area )
{
  unsigned idx1, idx2, idx3, idx4;
  getAreaIdxNew( area.Y(), *m_pcv, idx1, idx2, idx3, idx4 );
//  DTRACE( g_trace_ctx, D_TMP, "%d loc %d %d %d %d\n", g_trace_ctx->getChannelCounter(D_TMP), idx1, idx2, idx3, idx4);
  return *m_codedCUInfo[idx1][idx2][idx3][idx4];
}

void CodedCUInfo::setMv( const RefPicList refPicList, const int iRefIdx, const Mv& rMv )
{
  if( iRefIdx >= MAX_STORED_CU_INFO_REFS ) return;

  saveMv [refPicList][iRefIdx] = rMv;
  validMv[refPicList][iRefIdx] = true;
}

bool CodedCUInfo::getMv( const RefPicList refPicList, const int iRefIdx, Mv& rMv ) const
{
  if( iRefIdx >= MAX_STORED_CU_INFO_REFS )
  {
    rMv = saveMv[refPicList][0];
    return false;
  }

  rMv = saveMv[refPicList][iRefIdx];
  return validMv[refPicList][iRefIdx];
}

uint8_t CacheBlkInfoCtrl::findBestSbt( const UnitArea& area, const uint32_t curPuSse )
{
  unsigned idx1, idx2, idx3, idx4;
  getAreaIdxNew( area.Y(), *m_pcv, idx1, idx2, idx3, idx4 );
  CodedCUInfo* pSbtSave = m_codedCUInfo[idx1][idx2][idx3][idx4];

  for( int i = 0; i < pSbtSave->numPuInfoStored; i++ )
  {
    if( curPuSse == pSbtSave->puSse[i] )
    {
      return pSbtSave->puSbt[i];
    }
  }

  return MAX_UCHAR;
}

bool CacheBlkInfoCtrl::saveBestSbt( const UnitArea& area, const uint32_t curPuSse, const uint8_t curPuSbt )
{
  unsigned idx1, idx2, idx3, idx4;
  getAreaIdxNew( area.Y(), *m_pcv, idx1, idx2, idx3, idx4 );
  CodedCUInfo* pSbtSave = m_codedCUInfo[idx1][idx2][idx3][idx4];

  if( pSbtSave->numPuInfoStored == SBT_NUM_SL )
  {
    return false;
  }

  pSbtSave->puSse[pSbtSave->numPuInfoStored] = curPuSse;
  pSbtSave->puSbt[pSbtSave->numPuInfoStored] = curPuSbt;
  pSbtSave->numPuInfoStored++;

  return true;
}

// sharedLevels: the number of leading levels of the split path (the CTU counts as one) whose nodes have been searched
// only once since the result was stored; INT_MAX when that holds for the whole path, as it does without the DEV split
// refinement (BestEncInfoCache::xSharedLevels)
static bool isTheSameNbHood( const CodingUnit &cu, const CodingStructure& cs, const Partitioner &partitioner, int picW, int picH, const int sharedLevels )
{
  if( cu.chType != partitioner.chType )
  {
    return false;
  }

  const PartitioningStack &ps = partitioner.getPartStack();

  int i = 1;

  for( ; i < ps.size(); i++ )
  {
    if( ps[i].split != CU::getSplitAtDepth( cu, i - 1 ) )
    {
      break;
    }
  }

  // A node that is searched a second time may by now have other blocks coded before it, also where the split path
  // to the stored result is the same. Only the neighbours outside the deepest node searched once are still the same.
  i = std::min( i, sharedLevels );

  const UnitArea& cmnAnc = ps[i - 1].parts[ps[i - 1].idx];
  const UnitArea cuArea  = CS::getArea( cs, cu, partitioner.chType, partitioner.treeType );

  for( int i = 0; i < cmnAnc.blocks.size(); i++ )
  {
    if( i < cuArea.blocks.size() && cuArea.blocks[i].valid() && cuArea.blocks[i].pos() != cmnAnc.blocks[i].pos() )
    {
      return false;
    }
  }

  return true;
}

void BestEncInfoCache::create( const bool reuseCuResults, const ChromaFormat chFmt, const int ctuSize )
{
  m_reuseCuResults = reuseCuResults;

  if( !m_reuseCuResults ) return;

  m_chFmt = chFmt;
  std::fill_n( &m_bestEncInfo[0][0][0][0], sizeof( m_bestEncInfo ) / sizeof( m_bestEncInfo[0][0][0][0] ), nullptr );
}

void BestEncInfoCache::destroy()
{
  if( !m_reuseCuResults ) return;

  std::fill_n( &m_bestEncInfo[0][0][0][0], sizeof( m_bestEncInfo ) / sizeof( m_bestEncInfo[0][0][0][0] ), nullptr );
  m_filledSlots.clear();
  m_entryArena.release();
  m_coeffArena.release();
  m_dmvrMvArena.release();

  m_pcv = nullptr;
}

void BestEncInfoCache::init( const Slice &slice )
{
  if( !m_reuseCuResults ) return;

  // A new CTU starts, and no result of the previous one can match its CUs.
  for( BestEncodingInfo** slot : m_filledSlots )
  {
    *slot = nullptr;
  }
  m_filledSlots.clear();
  m_entryArena.reset();
  m_coeffArena.reset();
  m_dmvrMvArena.reset();
  m_storeCount        = 0;
  m_revisitFirstStore = 0;

  bool isInitialized = m_pcv;

  m_pcv = slice.pps->pcv;

  if( isInitialized ) return;

  m_dummyCS.pcv = m_pcv;
}

// Takes storage for the result of a CU with size indices wIdx and hIdx, set up the way the earlier
// fixed table of results was set up when it was created.
BestEncodingInfo* BestEncInfoCache::xNewEntry( const int wIdx, const int hIdx )
{
  BestEncodingInfo* encInfo = m_entryArena.take( 1 );
  encInfo->~BestEncodingInfo();
  new( encInfo ) BestEncodingInfo();

  const UnitArea area( m_chFmt, Area( 0, 0, 1 << ( wIdx + MIN_CU_LOG2 ), 1 << ( hIdx + MIN_CU_LOG2 ) ) );

  encInfo->cu.UnitArea::operator=( area );
  encInfo->tu.UnitArea::operator=( area );

  encInfo->cu.chType   = CH_L;
  encInfo->cu.treeType = TREE_D;
  encInfo->cu.modeType = MODE_TYPE_ALL;
  encInfo->cu.qp       = MAX_SCHAR;
  encInfo->tu.chType   = CH_L;

  if( hIdx >= 1 && wIdx >= 1 && ( wIdx + hIdx ) >= 3 )
  {
    const int dmvrSize = ( 1 << std::max( 0, ( wIdx + MIN_CU_LOG2 - DMVR_SUBCU_SIZE_LOG2 ) ) ) * ( 1 << std::max( 0, ( hIdx + MIN_CU_LOG2 - DMVR_SUBCU_SIZE_LOG2 ) ) );
    encInfo->cu.mvdL0SubPu = m_dmvrMvArena.take( dmvrSize );
  }

  TCoeffSig* coeff[MAX_NUM_TBLOCKS] = { 0, };
  for( int i = 0; i < area.blocks.size(); i++ )
  {
    coeff[i] = m_coeffArena.take( area.blocks[i].area() );
  }

  encInfo->tu.cs = &m_dummyCS;
  encInfo->tu.init( coeff );

  encInfo->poc      = -1;
  encInfo->testMode = EncTestMode();
  return encInfo;
}

bool BestEncInfoCache::setFromCs( const CodingStructure& cs, const EncTestMode& testMode, const Partitioner& partitioner )
{
  if( !m_reuseCuResults ) return false;

  if( cs.cus.size() != 1 || cs.tus.size() != 1 || partitioner.maxBTD <= 1 )
  {
    return false;
  }

  unsigned idx1, idx2, idx3, idx4;
  getAreaIdxNew( cs.area.Y(), *m_pcv, idx1, idx2, idx3, idx4 );

  BestEncodingInfo*& slot = m_bestEncInfo[idx1][idx2][idx3][idx4];
  if( !slot )
  {
    slot = xNewEntry( idx1, idx2 );
    m_filledSlots.push_back( &slot );
  }
  BestEncodingInfo& encInfo = *slot;

  encInfo.poc            =  cs.picture->poc;
  encInfo.cu.UnitArea::operator= ( *cs.cus.front() );
  encInfo.tu.UnitArea::operator= ( *cs.tus.front() );
  encInfo.cu             = *cs.cus.front();
  for( auto &blk : cs.tus.front()->blocks )
  {
    if( blk.valid() ) encInfo.tu.copyComponentFrom( *cs.tus.front(), blk.compID );
  }
  encInfo.testMode       = testMode;
  encInfo.dist           = cs.dist;
  encInfo.costEDO        = cs.costDbOffset;
  encInfo.storeNumber    = ++m_storeCount;

  return true;
}

// How far down the split path a stored result can share its neighbourhood with the CU being coded now (isTheSameNbHood)
int BestEncInfoCache::xSharedLevels( const BestEncodingInfo& encInfo ) const
{
  const bool storedBeforeSecondSearch = encInfo.storeNumber <= m_revisitFirstStore;   // never true while m_revisitFirstStore is 0: store numbers start at 1
  return storedBeforeSecondSearch ? m_revisitSharedLevels : INT_MAX;
}

// The DEV split refinement searches some CUs as a picture with a lower MTT depth limit would (mttDepthLimit, see
// EncModeCtrl::xRuleMttDepthLimit), and such a picture reuses nothing if that limit is 1. Results are still stored
// with the picture's own limit (setFromCs): storing changes nothing for a CU that reuses nothing, and the refinement
// pass, which searches with the picture's limit, can then reuse what the capped pass found.
bool BestEncInfoCache::isReusingCuValid( const CodingStructure& cs, const Partitioner& partitioner, int qp, const unsigned mttDepthLimit )
{
  if( !m_reuseCuResults || partitioner.treeType == TREE_C || mttDepthLimit <= 1 || partitioner.currQgEnable() )
  {
    return false; //if save & load is allowed for chroma CUs, we should check whether luma info (pred, recon, etc) is the same, which is quite complex
  }

  unsigned idx1, idx2, idx3, idx4;
  getAreaIdxNew( cs.area.Y(), *m_pcv, idx1, idx2, idx3, idx4 );

  const BestEncodingInfo* stored = m_bestEncInfo[idx1][idx2][idx3][idx4];
  if( !stored )
  {
    return false;
  }
  const BestEncodingInfo& encInfo = *stored;

  if( encInfo.cu.treeType != partitioner.treeType || encInfo.cu.modeType != partitioner.modeType )
  {
    return false;
  }
  if( encInfo.cu.qp != qp )
    return false;
  // the cheap tests first, the area and neighbourhood comparisons last
  if( cs.picture->poc != encInfo.poc
    || cs.currQP[partitioner.chType] != encInfo.cu.qp
    || CU::isIBC(encInfo.cu)
    || CS::getArea( cs, cs.area, partitioner.chType, partitioner.treeType ) != CS::getArea( cs, encInfo.cu, partitioner.chType, partitioner.treeType )
    || !isTheSameNbHood( encInfo.cu, cs, partitioner, (cs.picture->Y().width), (cs.picture->Y().height), xSharedLevels( encInfo ) )
    )
  {
    return false;
  }
  else
  {
    return true;
  }
}

bool BestEncInfoCache::setCsFrom( CodingStructure& cs, EncTestMode& testMode, const Partitioner& partitioner ) const
{
  if( !m_reuseCuResults ) return false;

  unsigned idx1, idx2, idx3, idx4;
  getAreaIdxNew( cs.area.Y(), *m_pcv, idx1, idx2, idx3, idx4 );

  const BestEncodingInfo* stored = m_bestEncInfo[idx1][idx2][idx3][idx4];
  if( !stored )
  {
    return false;
  }
  const BestEncodingInfo& encInfo = *stored;

  if( cs.picture->poc != encInfo.poc 
    || CS::getArea( cs, cs.area, partitioner.chType, partitioner.treeType ) != CS::getArea( cs, encInfo.cu, partitioner.chType, partitioner.treeType ) 
    || !isTheSameNbHood( encInfo.cu, cs, partitioner, (cs.picture->Y().width), (cs.picture->Y().height), xSharedLevels( encInfo ) )
    || partitioner.currQgEnable() || cs.currQP[partitioner.chType] != encInfo.cu.qp
    )
  {
    return false;
  }

  const UnitArea ua = CS::getArea( cs, cs.area, partitioner.chType, partitioner.treeType );
  CodingUnit     &cu = cs.addCU( ua, partitioner.chType );
  cu.treeType = partitioner.treeType;
  cu.modeType = partitioner.modeType;
  cu.initPuData();
  TransformUnit  &tu = cs.addTU( ua, partitioner.chType, &cu );

  cu          .repositionTo( encInfo.cu );
  cu          .repositionTo( encInfo.cu );
  tu          .repositionTo( encInfo.tu );

  cu          = encInfo.cu;
  cu          = encInfo.cu;
  for( auto &blk : tu.blocks )
  {
    if( blk.valid() ) tu.copyComponentFrom( encInfo.tu, blk.compID );
  }

  testMode    = encInfo.testMode;
  cs.dist     = encInfo.dist;
  cs.costDbOffset = encInfo.costEDO;
  return true;
}


//////////////////////////////////////////////////////////////////////////
// EncModeCtrl
//////////////////////////////////////////////////////////////////////////

// DEV MTT entry gate and census, nearby-use gate and search census (EncModeCtrl.h)

static int devEnvInt( const char* name, const int defaultValue )
{
  const char* value = getenv( name );
  return value ? ( int ) strtol( value, nullptr, 0 ) : defaultValue;   // base 0: tool masks may be given as 0x...
}

// Opens the census file of one encoder thread, <prefix>.<n>.txt: one file per thread needs no lock and mixes no lines
static FILE* openCensusFile( const char* prefix )
{
  static std::atomic<int> censusFileCount{ 0 };
  const std::string fileName = std::string( prefix ) + "." + std::to_string( censusFileCount++ ) + ".txt";
  FILE* const file = fopen( fileName.c_str(), "w" );
  if( file )
  {
    setvbuf( file, nullptr, _IOFBF, 1 << 20 );   // few, large writes: the lines are written inside the timed searches
  }
  return file;
}

static double devEnvDouble( const char* name, const double defaultValue )
{
  const char* value = getenv( name );
  return value ? std::max( 0.0, atof( value ) ) : defaultValue;
}

// Index of a BT / TT split in the census arrays, -1 for any other split
static int mttSplitIndex( const PartSplit split )
{
  switch( split )
  {
  case CU_HORZ_SPLIT: return 0;
  case CU_VERT_SPLIT: return 1;
  case CU_TRIH_SPLIT: return 2;
  case CU_TRIV_SPLIT: return 3;
  default:            return -1;
  }
}

static bool isHorizontalMttSplit( const PartSplit split )
{
  return split == CU_HORZ_SPLIT || split == CU_TRIH_SPLIT;
}

// How far apart the motion of two CUs is: the largest difference of an MV component (MV units), or -1 where one CU
// could not take over the other's motion (different prediction mode, direction or reference picture, or motion that
// is not one vector per list: affine, GEO, sub-block merge). Two intra CUs count as alike (0).
static int motionGap( const CodingUnit& a, const CodingUnit& b )
{
  if( a.predMode != b.predMode )
  {
    return -1;
  }
  if( CU::isIntra( a ) )
  {
    return 0;
  }
  const bool oneVectorPerList = !a.affine && !b.affine && !a.geo && !b.geo && a.mergeType == MRG_TYPE_DEFAULT_N && b.mergeType == MRG_TYPE_DEFAULT_N;
  if( !CU::isInter( a ) || !oneVectorPerList || a.interDir != b.interDir )
  {
    return -1;
  }
  int gap = 0;
  for( int list = 0; list < NUM_REF_PIC_LIST_01; list++ )
  {
    if( a.interDir & ( 1 << list ) )
    {
      if( a.refIdx[list] != b.refIdx[list] )
      {
        return -1;
      }
      gap = std::max( { gap, abs( a.mv[list][0].hor - b.mv[list][0].hor ), abs( a.mv[list][0].ver - b.mv[list][0].ver ) } );
    }
  }
  return gap;
}

// The DevTool bits of a coded CU
static uint32_t devToolBits( const CodingUnit& cu )
{
  const auto bit = []( const bool used, const DevTool tool ) { return used ? 1u << tool : 0u; };
  const uint32_t mtt = bit( cu.mtDepth > 0, DEV_TOOL_MTT );
  if( CU::isIntra( cu ) )
  {
    return mtt | bit( true, DEV_TOOL_INTRA ) | bit( cu.ispMode != 0, DEV_TOOL_ISP ) | bit( cu.mipFlag, DEV_TOOL_MIP );
  }
  if( !CU::isInter( cu ) )
  {
    return mtt;   // IBC
  }
  const bool merge = cu.mergeFlag;
  return mtt | bit( merge, DEV_TOOL_MERGE ) | bit( cu.mmvdMergeFlag, DEV_TOOL_MMVD ) | bit( cu.geo, DEV_TOOL_GEO ) | bit( cu.ciip, DEV_TOOL_CIIP )
             | bit( cu.affine && merge, DEV_TOOL_AFFINE_MERGE ) | bit( cu.affine && !merge, DEV_TOOL_AFFINE )
             | bit( cu.imv != 0 && !merge, DEV_TOOL_IMV ) | bit( cu.smvdMode != 0 && !merge, DEV_TOOL_SMVD )
             | bit( cu.BcwIdx != BCW_DEFAULT && !merge, DEV_TOOL_BCW ) | bit( cu.sbtInfo != 0, DEV_TOOL_SBT );
}

// Index of a search in DevNodeStats, -1 for a mode the search census does not keep
static int devSearchIndex( const EncTestModeType type )
{
  switch( type )
  {
  case ETM_MERGE_SKIP: return DEV_SEARCH_MERGE;
  case ETM_INTER_ME:   return DEV_SEARCH_INTER_ME;
  case ETM_INTER_IMV:  return DEV_SEARCH_IMV;
  case ETM_IBC:
  case ETM_IBC_MERGE:  return DEV_SEARCH_IBC;
  case ETM_INTRA:      return DEV_SEARCH_INTRA;
  case ETM_SPLIT_QT:   return DEV_SEARCH_QT;
  case ETM_SPLIT_BT_H: return DEV_SEARCH_BT_H;
  case ETM_SPLIT_BT_V: return DEV_SEARCH_BT_V;
  case ETM_SPLIT_TT_H: return DEV_SEARCH_TT_H;
  case ETM_SPLIT_TT_V: return DEV_SEARCH_TT_V;
  default:             return -1;
  }
}

static DevNeighbourCu describeNeighbour( const CodingUnit* cu )
{
  DevNeighbourCu neighbour;
  if( cu )
  {
    neighbour.width   = cu->lwidth();
    neighbour.height  = cu->lheight();
    neighbour.qtDepth = cu->qtDepth;
    neighbour.mtDepth = cu->mtDepth;
  }
  return neighbour;
}

// Sums the SAD between a block's source and prediction into four strips from top to bottom (rows) and four from left
// to right (cols)
static void predErrorStrips( const CPelBuf& org, const CPelBuf& pred, int rows[NUM_ERROR_STRIPS], int cols[NUM_ERROR_STRIPS] )
{
  std::fill_n( rows, NUM_ERROR_STRIPS, 0 );
  std::fill_n( cols, NUM_ERROR_STRIPS, 0 );
  for( unsigned y = 0; y < org.height; y++ )
  {
    const Pel* orgRow  = org.bufAt( 0, y );
    const Pel* predRow = pred.bufAt( 0, y );
    int rowSum = 0;
    for( unsigned x = 0; x < org.width; x++ )
    {
      const int diff = abs( orgRow[x] - predRow[x] );
      rowSum                                  += diff;
      cols[x * NUM_ERROR_STRIPS / org.width ] += diff;
    }
    rows[y * NUM_ERROR_STRIPS / org.height] += rowSum;
  }
}

// How much the prediction error differs between the parts that the BT / TT splits of one direction would make, as a
// share of the block's error: the larger of the BT contrast (first half against second half) and the TT contrast
// (middle half against the outer quarters). strips: the error in four strips across the split (rows for the
// horizontal splits, which cut the block into rows; cols for the vertical ones). 0 for a block without error.
static double splitErrorContrast( const int strips[NUM_ERROR_STRIPS] )
{
  const int total = strips[0] + strips[1] + strips[2] + strips[3];
  if( total <= 0 )
  {
    return 0.0;
  }
  const int btContrast = abs( strips[0] + strips[1] - strips[2] - strips[3] );
  const int ttContrast = abs( strips[1] + strips[2] - strips[0] - strips[3] );
  return ( double ) std::max( btContrast, ttContrast ) / total;
}

// DEV motion estimate of the splits (EncModeCtrl.h): a candidate motion of a node, as the reference picture and the
// place in it that predicts the node (whole samples, kept inside the picture)
struct MotionCand
{
  const Picture* refPic;
  int            x, y;
  bool operator==( const MotionCand& other ) const { return refPic == other.refPic && x == other.x && y == other.y; }
};

static constexpr int MAX_MOTION_CANDS = 12;
static constexpr int NUM_MOTION_CELLS = 4;   // per direction; a multiple of 4, so that BT, TT and QT parts are whole cells

typedef int MotionCellSad[NUM_MOTION_CELLS][NUM_MOTION_CELLS];

// SAD between a block's source and one candidate's prediction, for each cell of a 4 x 4 grid over the block
static void motionCellSad( const CPelBuf& org, const CPelBuf& ref, const MotionCand& cand, MotionCellSad sad )
{
  const int cellWidth  = org.width  / NUM_MOTION_CELLS;
  const int cellHeight = org.height / NUM_MOTION_CELLS;
  for( int cellY = 0; cellY < NUM_MOTION_CELLS; cellY++ )
  {
    std::fill_n( sad[cellY], NUM_MOTION_CELLS, 0 );
    for( int y = cellY * cellHeight; y < ( cellY + 1 ) * cellHeight; y++ )
    {
      const Pel* orgRow = org.bufAt( 0, y );
      const Pel* refRow = ref.bufAt( cand.x, cand.y + y );
      for( int cellX = 0; cellX < NUM_MOTION_CELLS; cellX++ )
      {
        int sum = 0;
        for( int x = cellX * cellWidth; x < ( cellX + 1 ) * cellWidth; x++ )
        {
          sum += abs( orgRow[x] - refRow[x] );
        }
        sad[cellY][cellX] += sum;
      }
    }
  }
}

// SAD of the cells [y0, y1) x [x0, x1) with the candidate that predicts that part best
static int bestPartSad( const MotionCellSad* sadOfCand, const int numCands, const int y0, const int y1, const int x0, const int x1 )
{
  int best = INT_MAX;
  for( int cand = 0; cand < numCands; cand++ )
  {
    int sum = 0;
    for( int y = y0; y < y1; y++ )
    {
      for( int x = x0; x < x1; x++ )
      {
        sum += sadOfCand[cand][y][x];
      }
    }
    best = std::min( best, sum );
  }
  return best;
}

// DEV skipped-parent gate: the searches it has a rule for (DevParentSearch bits)
static constexpr uint32_t PARENT_GATE_SEARCHES =
    1u << DEV_TOOL_IMV | 1u << DEV_TOOL_SBT | 1u << DEV_TOOL_INTRA | 1u << DEV_TOOL_AFFINE | 1u << DEV_TOOL_GEO | 1u << DEV_TOOL_CIIP | 1u << DEV_TOOL_BCW
  | 1u << DEV_TOOL_SMVD | 1u << DEV_TOOL_MTT | 1u << DEV_TOOL_AFFINE_MERGE | 1u << DEV_TOOL_MMVD | 1u << DEV_PARENT_ME | 1u << DEV_PARENT_QT;

// The DEV split gates (EncModeCtrl.h) can act where the picture's MTT depth limit is in this range. Below 2 an MTT
// root would itself be the node whose splits are left out; above 3 the capped pass has levels of nodes that the
// refinement pass has no rule for.
static constexpr unsigned MIN_GATED_MTT_DEPTH_LIMIT = 2;
static constexpr unsigned MAX_GATED_MTT_DEPTH_LIMIT = 3;
static constexpr int      CTU_SIZE_IN_MIN_CUS       = MAX_CU_SIZE >> MIN_CU_LOG2;

static bool devEnvIsOn( const char* name )
{
  const char* value = getenv( name );
  return value && atoi( value ) != 0;
}

// A cost margin of the DEV split refinement: a ratio to the cost it is compared with
static double devEnvMargin( const char* name, const double defaultMargin = 1.0 )
{
  const char* value = getenv( name );
  return value ? std::max( 0.0, atof( value ) ) : defaultMargin;   // TR / TP below 1 refine less than the best split; 0: no refinement pass
}

// MTT depth below the splits the picture border forces, which are not counted against the depth limit
static int freeMttDepth( const Partitioner& partitioner )
{
  return ( int ) partitioner.currMtDepth - ( int ) partitioner.currImplicitBtDepth;
}

static bool isBtOrTt( const PartSplit split )
{
  return split == CU_HORZ_SPLIT || split == CU_VERT_SPLIT || split == CU_TRIH_SPLIT || split == CU_TRIV_SPLIT;
}

// Calls visit( index ) for every 4x4 luma block of the area, with the block's index in a map of the CTU
template<typename Visit>
static void forEachMinCu( const Area& lumaArea, const PreCalcValues& pcv, Visit visit )
{
  const int firstColumn = ( lumaArea.x & pcv.maxCUSizeMask ) >> MIN_CU_LOG2;
  const int firstRow    = ( lumaArea.y & pcv.maxCUSizeMask ) >> MIN_CU_LOG2;
  const int columns     = lumaArea.width  >> MIN_CU_LOG2;
  const int rows        = lumaArea.height >> MIN_CU_LOG2;
  for( int row = firstRow; row < firstRow + rows; row++ )
  {
    for( int column = firstColumn; column < firstColumn + columns; column++ )
    {
      visit( row * CTU_SIZE_IN_MIN_CUS + column );
    }
  }
}

void EncModeCtrl::init( const VVEncCfg& encCfg, RdCost* pRdCost )
{
  m_pcEncCfg = &encCfg;
  m_pcRdCost = pRdCost;
  comprCUCtx = nullptr;

  CacheBlkInfoCtrl::create( encCfg.m_CTUSize );
  BestEncInfoCache::create( encCfg.m_reuseCuResults, encCfg.m_internChromaFormat, encCfg.m_CTUSize );

  // DEV MTT entry gate and census: the environment is read here, once per encoder thread
  m_devMttGate       = devEnvInt   ( "VVENC_DEV_MTT_GATE", 0 ) != 0;
  m_devGateSkip      = devEnvInt   ( "VVENC_DEV_MTT_GATE_SKIP", 1 );
  m_devGateQtRatio   = devEnvDouble( "VVENC_DEV_MTT_GATE_QT", 1.02 );
  m_devGateDensity   = devEnvDouble( "VVENC_DEV_MTT_GATE_DENSITY", 0.0 );
  m_devGatePairMvTol = devEnvInt   ( "VVENC_DEV_MTT_GATE_PAIR", -1 );
  m_devGateShape     = devEnvDouble( "VVENC_DEV_MTT_GATE_SHAPE", 0.0 );
  m_devGateShapeDir  = devEnvDouble( "VVENC_DEV_MTT_GATE_SHAPE_DIR", 0.0 );
  m_devGateMaxLimit  = std::max( 1, devEnvInt( "VVENC_DEV_MTT_GATE_LIMIT", 1 ) );
  m_devGateMtDepths  = devEnvInt   ( "VVENC_DEV_MTT_GATE_MTDEPTHS", 1 );
  m_devGateMotion     = devEnvDouble( "VVENC_DEV_MTT_GATE_MOTION", 0.0 );
  m_devGateMotionBits = devEnvDouble( "VVENC_DEV_MTT_GATE_MOTION_BITS", 0.0 );
  m_devGateAsTried    = devEnvInt   ( "VVENC_DEV_MTT_GATE_AS_TRIED", 0 ) != 0;
  m_devMeGate         = devEnvInt   ( "VVENC_DEV_ME_GATE", 0 );
  m_devStopT          = devEnvDouble( "VVENC_DEV_SPLIT_STOP", 0.0 );
  m_devStopRest       = devEnvDouble( "VVENC_DEV_SPLIT_STOP_REST", 0.0 );
  m_devStopShare      = devEnvInt   ( "VVENC_DEV_SPLIT_STOP_SHARE", DEV_REST_BY_ERROR );
  m_devStopBase       = devEnvInt   ( "VVENC_DEV_SPLIT_STOP_BASE", 0 );
  m_devStopSplits     = devEnvInt   ( "VVENC_DEV_SPLIT_STOP_SPLITS", 3 );
  m_devStopMtDepths   = devEnvInt   ( "VVENC_DEV_SPLIT_STOP_MTDEPTHS", -1 );
  m_devStopTtFollows  = devEnvInt   ( "VVENC_DEV_SPLIT_STOP_TT", 0 ) != 0;
  m_devStopDir2       = devEnvDouble( "VVENC_DEV_SPLIT_STOP_DIR2", 0.0 );
  m_devIntraGateGood  = devEnvDouble( "VVENC_DEV_INTRA_GATE_GOOD", 0.0 );
  m_devParentGate     = devEnvInt   ( "VVENC_DEV_PARENT_GATE", 0 ) & PARENT_GATE_SEARCHES;
  m_devParentLevel    = devEnvInt   ( "VVENC_DEV_PARENT_GATE_LEVEL", 1 );
  m_devParentDepth    = std::max( 0, devEnvInt( "VVENC_DEV_PARENT_GATE_DEPTH", 1 ) );
  m_devParentOwn      = devEnvInt   ( "VVENC_DEV_PARENT_GATE_OWN", 0 ) != 0;
  m_devAmvrGateSkip    = devEnvInt   ( "VVENC_DEV_AMVR_GATE_SKIP", 0 ) != 0;
  m_devAmvrGateBehind  = devEnvDouble( "VVENC_DEV_AMVR_GATE_BEHIND", 0.0 );
  m_devAmvrGateMvd     = devEnvInt   ( "VVENC_DEV_AMVR_GATE_MVD", 0 );
  m_devAmvrGate4PelMvd = devEnvInt   ( "VVENC_DEV_AMVR_GATE_4PEL", 0 );
  m_devSbtGate         = devEnvDouble( "VVENC_DEV_SBT_GATE", 0.0 );
  const char* censusPrefix = getenv( "VVENC_DEV_MTT_CENSUS" );
  if( censusPrefix && !m_censusFile )
  {
    m_censusFile = openCensusFile( censusPrefix );
  }

  // DEV nearby-use gate and search census
  m_devToolGate        = devEnvInt( "VVENC_DEV_TOOL_GATE", 0 ) & ( ( 1u << NUM_GATED_DEV_TOOLS ) - 1 );
  m_devToolGateSources = devEnvInt( "VVENC_DEV_TOOL_GATE_SRC", 7 );
  m_devToolGateProbe   = devEnvInt( "VVENC_DEV_TOOL_GATE_PROBE", 4 );
  m_searchCensusStep   = std::max( 1, devEnvInt( "VVENC_DEV_SEARCH_CENSUS_STEP", 1 ) );
  const char* searchCensusPrefix = getenv( "VVENC_DEV_SEARCH_CENSUS" );
  if( searchCensusPrefix && !m_searchCensusFile )
  {
    m_searchCensusFile = openCensusFile( searchCensusPrefix );
  }
  m_devFinalCensus = m_searchCensusFile && devEnvInt( "VVENC_DEV_FINAL_CENSUS", 0 ) != 0;
  m_searchCensusPoc = devEnvInt( "VVENC_DEV_SEARCH_CENSUS_POC", -1 );

  // DEV split gates: the environment is read here, once per encoder thread
  m_devSplitRefine      = devEnvIsOn  ( "VVENC_DEV_SPLIT_REFINE" );
  m_devRefineRootMargin = devEnvMargin( "VVENC_DEV_SPLIT_REFINE_TR" );
  m_devRefineNodeMargin = devEnvMargin( "VVENC_DEV_SPLIT_REFINE_TP" );
  m_devRefineAfterQt    = m_devSplitRefine && devEnvIsOn( "VVENC_DEV_SPLIT_REFINE_AFTERQT" );
  m_devRefineQtMargin   = devEnvMargin( "VVENC_DEV_SPLIT_REFINE_QTM", 1.05 );
  m_devCappedPassCheck  = m_devSplitRefine && devEnvIsOn( "VVENC_DEV_SPLIT_REFINE_CHECK" );
  m_devRootRules        = m_devSplitRefine && !m_devCappedPassCheck ? ( unsigned ) devEnvInt( "VVENC_DEV_SPLIT_REFINE_RULES", 0 ) : 0;
  // not with the check: it raises the search limit of every inter picture, which only a capped pass takes back
  m_devRefineMinTLayer  = m_devSplitRefine && !m_devCappedPassCheck ? ( unsigned ) std::max( 0, devEnvInt( "VVENC_DEV_SPLIT_REFINE_MINTL", 0 ) ) : 0;
  m_refinesThisCtu      = false;
  m_devGateAllLayers    = !m_devCappedPassCheck && devEnvIsOn( "VVENC_DEV_SPLIT_ALL_LAYERS" );
  // The gated pictures are those with the highest MTT depth limit of the encode's layers, which is also the id of
  // the encode's split-cost table. The check of the capped pass searches them with a limit one higher, so there
  // the gates act on that limit.
  const unsigned highestLayerLimit = ( unsigned ) std::max( 0, encCfg.m_splitCostThrParamId );
  const unsigned highestGatedLimit = highestLayerLimit + ( m_devCappedPassCheck ? 1 : 0 );
  const bool     limitCanBeGated   = highestGatedLimit >= MIN_GATED_MTT_DEPTH_LIMIT && highestGatedLimit <= MAX_GATED_MTT_DEPTH_LIMIT;
  m_highestGatedMttLimit = limitCanBeGated ? highestGatedLimit : 0;
  m_lowestGatedMttLimit  = !limitCanBeGated ? 1 : m_devGateAllLayers ? MIN_GATED_MTT_DEPTH_LIMIT : highestGatedLimit;
  m_searchDepthRaise    = 0;
  m_devSplitCover       = devEnvIsOn  ( "VVENC_DEV_SPLIT_COVER" );
  m_deepSplitPass       = DSP_ALL;
  if( m_devSplitCover )
  {
    m_deepSeenByRoot.assign( CTU_SIZE_IN_MIN_CUS * CTU_SIZE_IN_MIN_CUS, 0 );
    m_deepWonByRoot .assign( CTU_SIZE_IN_MIN_CUS * CTU_SIZE_IN_MIN_CUS, 0 );
  }
}

void EncModeCtrl::destroy()
{
  for( FILE** censusFile : { &m_censusFile, &m_searchCensusFile } )
  {
    if( *censusFile )
    {
      fclose( *censusFile );
      *censusFile = nullptr;
    }
  }
  if( m_devCappedPassCheck )
  {
    // positive control of the check: proves the raise and the capped passes ran (a run where neither fires matches trivially)
    fprintf( stderr, "DEV SPLIT CHECK: %u CTUs raised, %u capped passes\n", m_devCheckRaisedCtus, m_devCappedPasses );
  }
  CacheBlkInfoCtrl::destroy();
  BestEncInfoCache::destroy();
}

void EncModeCtrl::initCTUEncoding( const Slice &slice )
{
  CacheBlkInfoCtrl::init( slice );
  BestEncInfoCache::init( slice );

  CHECK( !m_ComprCUCtxList.empty(), "Mode list is not empty at the beginning of a CTU" );

  // v10: the split refinement acts on the CTU only if its picture's temporal id is high enough
  m_refinesThisCtu = m_devSplitRefine && slice.TLayer >= m_devRefineMinTLayer;

  if( m_devSplitCover )
  {
    // root numbers start again at 1, so the marks of the previous CTU must go
    m_mttRootCount = 0;
    std::fill( m_deepSeenByRoot.begin(), m_deepSeenByRoot.end(), 0 );
    std::fill( m_deepWonByRoot .begin(), m_deepWonByRoot .end(), 0 );
  }

  if( m_pcEncCfg->m_fastQtBtEnc )
  {
    m_skipThresholdE0023FastEnc = ((slice.getMinPictureDistance() <= PICTURE_DISTANCE_TH) ? FAST_SKIP_DEPTH : SKIP_DEPTH);
  }
  else
  {
    m_skipThresholdE0023FastEnc = SKIP_DEPTH;
  }
}

// KeepHighBand: a 64-point transform zeroes every coefficient above position 32, which is the finest octave of the block,
// whatever the rate. The finest octave of the source is measured per 8x8 cell as the mean energy of the three detail
// coefficients of an orthonormal 2x2 Haar transform; the block keeps it only if split, when the median cell's energy is
// above K^2 quantisation steps^2. K = .1: at .2 film keeps half the gain at QP 20 (detail tests 2026-11-06/07). The
// step uses the slice QP, not the CU's QP under QPA.
static bool finestOctaveNeedsSplit( const CodingStructure& cs, const Partitioner& partitioner )
{
  static constexpr double K        = 0.1;
  static constexpr int    cellSize = 8;
  static constexpr int    maxCells = ( MAX_CU_SIZE / cellSize ) * ( MAX_CU_SIZE / cellSize );
  const CPelBuf src     = cs.getOrgBuf( partitioner.currArea().block( COMP_Y ) );
  const int     columns = src.width / cellSize, rows = src.height / cellSize;
  CHECKD( columns * rows > maxCells, "more 8x8 cells than a CU can hold" );
  double cells[maxCells];
  int    cellCount = 0;
  for( int row = 0; row < rows; row++ )
  {
    for( int column = 0; column < columns; column++ )
    {
      int64_t energy = 0;
      for( int y = 0; y < cellSize; y += 2 )
      {
        const Pel* top    = src.bufAt( column * cellSize, row * cellSize + y );
        const Pel* bottom = src.bufAt( column * cellSize, row * cellSize + y + 1 );
        for( int x = 0; x < cellSize; x += 2 )
        {
          const int a = top[x], b = top[x + 1], c = bottom[x], d = bottom[x + 1];
          const int horizontal = a - b + c - d, vertical = a + b - c - d, diagonal = a - b - c + d;
          energy += horizontal * horizontal + vertical * vertical + diagonal * diagonal;
        }
      }
      cells[cellCount++] = energy / 4.0 / ( 3 * cellSize * cellSize / 4 );   // the division by 4 makes the transform orthonormal
    }
  }
  if( !cellCount )
  {
    return false;
  }
  std::nth_element( cells, cells + cellCount / 2, cells + cellCount );
  const double median = cells[cellCount / 2];
  const double qStep  = pow( 2.0, ( cs.slice->sliceQp - 4 ) / 6.0 ) * ( 1 << ( cs.sps->bitDepths[CH_L] - 8 ) );
  return median > K * K * qStep * qStep;
}

void EncModeCtrl::initCULevel( Partitioner &partitioner, const CodingStructure& cs, int  MergeSimpleFlag, const CodingUnit* cuLeft, const CodingUnit* cuAbove )
{
  // Min/max depth
  unsigned minDepth = 0;
  unsigned maxDepth = cs.pcv->getMaxDepth( cs.slice->sliceType, partitioner.chType );
  if( m_pcEncCfg->m_useFastLCTU )
  {
    partitioner.setMaxMinDepth(minDepth, maxDepth, cs, cs.picture->useQtbttSpeedUpMode, MergeSimpleFlag, cuLeft, cuAbove);
  }

  minDepth = std::max<unsigned>( minDepth, cs.pcv->getMinDepth( cs.slice->sliceType, partitioner.chType ) );
  // KeepHighBand: a QT split where a 64-point transform would zero the source's finest octave (only square CUs can split by QT)
  if( m_pcEncCfg->m_keepHighBand && m_pcEncCfg->m_DepQuantEnabled && partitioner.chType == CH_L && cs.slice->sliceQp <= KEEP_HIGH_BAND_MAX_SLICE_QP
      && partitioner.currArea().lwidth() == partitioner.currArea().lheight() && partitioner.currArea().lwidth() > 32 && partitioner.canSplit( CU_QUAD_SPLIT, cs ) && finestOctaveNeedsSplit( cs, partitioner ) )
  {
    minDepth = std::max<unsigned>( minDepth, partitioner.currQtDepth + 1 );
  }
  maxDepth = std::min<unsigned>( maxDepth, cs.pcv->getMaxDepth( cs.slice->sliceType, partitioner.chType ) );

  m_ComprCUCtxList.push_back( ComprCUCtx( cs, minDepth, maxDepth ) );
  comprCUCtx = &m_ComprCUCtxList.back();

  const unsigned mttDepthLimit = xRuleMttDepthLimit( cs, partitioner );
  const unsigned qtOrderLimit  = xRootRuleLimit( DRR_QT_ORDER, partitioner, mttDepthLimit );   // = mttDepthLimit without the DEV root rules

  const bool qtFirstByNeighbours = (  cuLeft  &&  cuAbove  && cuLeft ->qtDepth > partitioner.currQtDepth && cuAbove->qtDepth > partitioner.currQtDepth )
                                || (  cuLeft  && !cuAbove  && cuLeft ->qtDepth > partitioner.currQtDepth )
                                || ( !cuLeft  &&  cuAbove  && cuAbove->qtDepth > partitioner.currQtDepth )
                                || ( !cuAbove && !cuLeft   && cs.area.lwidth() >= ( 32 << cs.slice->TLayer ) );
  const auto qtFirstByLimit      = [&]( const unsigned limit ) { return m_pcEncCfg->m_qtbttSpeedUp > 1 && limit < ( ( cs.slice->isIntra() && !cs.sps->IBC ) ? 3u : 2u ); };
  const bool qtBeforeBt          = ( qtFirstByNeighbours || qtFirstByLimit( qtOrderLimit ) )
                                && ( cs.area.lwidth() > ( cs.pcv->getMinQtSize( *cs.slice, partitioner.chType ) << 1 ) );

  // set features
  ComprCUCtx &cuECtx    = *comprCUCtx;
  cuECtx.mttDepthLimit  = mttDepthLimit;
  cuECtx.qtBeforeBt     = qtBeforeBt;
  cuECtx.qtFirstByLowLimit = qtBeforeBt && !qtFirstByNeighbours && !qtFirstByLimit( codedMttDepthLimit( partitioner ) );   // DEV root rule DRR_QT_DEEPER
  cuECtx.doTriHorzSplit = true;
  cuECtx.doTriVertSplit = true;
  cuECtx.doMoreSplits   = 3;
  cuECtx.isReusingCu    = isReusingCuValid( cs, partitioner, cs.baseQP, mttDepthLimit );
  cuECtx.didHorzSplit   = partitioner.canSplit( CU_HORZ_SPLIT, cs );
  cuECtx.didVertSplit   = partitioner.canSplit( CU_VERT_SPLIT, cs );
  cuECtx.doHorChromaSplit = true;
  cuECtx.doVerChromaSplit = true;
  cuECtx.doQtChromaSplit  = true;
  cuECtx.censusNodeStart  = censusClock();   // DEV MTT census and search census; 0 without them
  if( m_devToolGate || m_searchCensusFile )
  {
    // DEV nearby-use gate and search census
    DevNodeStats& dev  = cuECtx.dev;
    dev.id             = m_devNodeCount++;
    const ComprCUCtx* parent = m_ComprCUCtxList.size() > 1 ? &*( m_ComprCUCtxList.end() - 2 ) : nullptr;
    dev.parentId       = parent ? parent->dev.id : -1;
    dev.toolsOfParent  = parent && parent->bestCostBeforeSplit != MAX_DOUBLE ? parent->dev.toolsOfBestNoSplit : ~0u;
    dev.toolsOfNearCus = ( cuLeft ? devToolBits( *cuLeft ) : 0 ) | ( cuAbove ? devToolBits( *cuAbove ) : 0 );
    dev.left           = describeNeighbour( cuLeft );
    dev.above          = describeNeighbour( cuAbove );
  }

  if( m_devSplitCover && xGatesDeepSplits( cs, partitioner ) )
  {
    // a luma node below an MTT root has a luma parent
    cuECtx.mttRootId = freeMttDepth( partitioner ) == 0 ? ++m_mttRootCount : ( m_ComprCUCtxList.end() - 2 )->mttRootId;
  }

  if( m_pcEncCfg->m_contentBasedFastQtbt && mttDepthLimit )
  {
    const CompArea& currArea = partitioner.currArea().Y();
    int cuHeight  = currArea.height;
    int cuWidth   = currArea.width;

    const bool condIntraInter = m_pcEncCfg->m_IntraPeriod == 1 ? ( partitioner.currBtDepth == 0 ) : ( cuHeight > 32 && cuWidth > 32 );

    if( cuWidth == cuHeight && condIntraInter )
    {
      const CPelBuf bufCurrArea = cs.getOrgBuf( partitioner.currArea().block( COMP_Y ) );

      int grad[4];
      g_pelBufOP.absGradientSums( bufCurrArea.buf, bufCurrArea.stride, cuWidth, cuHeight, grad );

      cuECtx.grad_horVal = (double)grad[0];
      cuECtx.grad_verVal = (double)grad[1];
      cuECtx.grad_dowVal = (double)grad[2];
      cuECtx.grad_dupVal = (double)grad[3];
    }
  }
}

// True at the nodes the DEV MTT entry gate decides for: a luma node in an inter picture, away from the picture border,
// with a result for the search without a split. By default a node of the QT tree where the MTT depth limit is 1, so
// that its BT / TT splits make leaf CUs; VVENC_DEV_MTT_GATE_LIMIT and _MTDEPTHS widen that (EncModeCtrl.h).
bool EncModeCtrl::xIsMttGateNode( const CodingStructure& cs, const Partitioner& partitioner ) const
{
  const ComprCUCtx& cuECtx   = *comprCUCtx;
  const unsigned    mttLimit = cs.pcv->getMaxMTTDepth( *cs.slice, CH_L );
  return !cs.slice->isIntra() && partitioner.chType == CH_L && partitioner.modeType == MODE_TYPE_ALL
      && partitioner.currImplicitBtDepth == 0
      && mttLimit >= 1 && mttLimit <= m_devGateMaxLimit
      && partitioner.currMtDepth < mttLimit && ( m_devGateMtDepths >> partitioner.currMtDepth & 1 )
      && cuECtx.bestCS && cuECtx.bestCostBeforeSplit != MAX_DOUBLE;
}

// The DEV MTT entry gate's decision for a node: the BT / TT directions it may search (rules in EncModeCtrl.h)
MttGate EncModeCtrl::xDecideMttGate( const Partitioner& partitioner )
{
  ComprCUCtx&  cuECtx      = *comprCUCtx;
  const double noSplitCost = cuECtx.bestCostBeforeSplit;
  const double bestCost    = cuECtx.bestCS->cost;   // the QT split's cost where that was searched first and won
  const double lumaSamples = ( double ) partitioner.currArea().lumaSize().area();

  const bool noResidual  = m_devGateSkip != 0 && !( cuECtx.nsFlags & ( NS_RESIDUAL | NS_INTRA ) );
  const bool qtFarBehind = m_devGateQtRatio > 0.0 && cuECtx.qtSplitCost != MAX_DOUBLE && cuECtx.qtSplitCost > m_devGateQtRatio * noSplitCost;
  const bool cheapBlock  = bestCost < m_devGateDensity * m_pcRdCost->getLambda() * lumaSamples;
  if( noResidual || qtFarBehind || cheapBlock )
  {
    // Where the QT split comes after the BT splits (16x16 at medium), two BT results of skipped CUs alone would have
    // stopped it (doMoreSplits, useModeResult). Below a skipped block that is what the unsearched BT splits would
    // most likely have given, so the QT split is stopped as before instead of being searched more often.
    if( m_devGateSkip != 2 && !cuECtx.qtBeforeBt && cuECtx.isBestNoSplitSkip )
    {
      cuECtx.doMoreSplits -= 2;
    }
    return MTT_GATE_NONE;
  }

  const auto directions = []( const bool horizontal, const bool vertical )
  {
    return ( horizontal ? MTT_GATE_HOR : MTT_GATE_NONE ) | ( vertical ? MTT_GATE_VER : MTT_GATE_NONE );
  };
  int allowed = MTT_GATE_BOTH;
  if( m_devGatePairMvTol >= 0 && cuECtx.qtLeavesKnown )
  {
    // A BT split joins two pairs of QT leaves: it can only stand in for the QT split where each pair moves alike
    const auto alike = [&]( const QtPair pair ) { return cuECtx.qtPairMvGap[pair] >= 0 && cuECtx.qtPairMvGap[pair] <= m_devGatePairMvTol; };
    allowed &= directions( alike( QT_PAIR_TOP ) && alike( QT_PAIR_BOTTOM ), alike( QT_PAIR_LEFT ) && alike( QT_PAIR_RIGHT ) );
  }
  const bool errorKnown = cuECtx.dev.predErrorRows[0] >= 0;
  if( ( m_devGateShape > 0.0 || m_devGateShapeDir > 0.0 ) && errorKnown )
  {
    // even error / weaker direction: horizontal splits cut the block into rows, vertical ones into columns
    const double horContrast = splitErrorContrast( cuECtx.dev.predErrorRows );
    const double verContrast = splitErrorContrast( cuECtx.dev.predErrorCols );
    allowed &= directions( horContrast >= m_devGateShape && horContrast >= m_devGateShapeDir * verContrast,
                           verContrast >= m_devGateShape && verContrast >= m_devGateShapeDir * horContrast );
  }
  const int* motionSad = cuECtx.dev.motionSad;
  if( xMotionGateIsOn() && motionSad[DEV_MOTION_NONE] >= 0 && !( cuECtx.nsFlags & NS_INTRA ) )
  {
    // one motion: a direction is searched only where other motion for the parts of its BT or TT split lowers the SAD
    const double neededGain = std::max( m_devGateMotion * motionSad[DEV_MOTION_NONE], m_devGateMotionBits * m_pcRdCost->getMotionLambda() );
    const auto pays = [&]( const DevMotionSplit bt, const DevMotionSplit tt )
    {
      return motionSad[DEV_MOTION_NONE] - std::min( motionSad[bt], motionSad[tt] ) >= neededGain;
    };
    allowed &= directions( pays( DEV_MOTION_BT_H, DEV_MOTION_TT_H ), pays( DEV_MOTION_BT_V, DEV_MOTION_TT_V ) );
  }
  return MttGate( allowed );
}

bool EncModeCtrl::xMttGateVetoes( const PartSplit split, const CodingStructure& cs, const Partitioner& partitioner )
{
  if( !xIsMttGateNode( cs, partitioner ) )
  {
    return false;
  }
  ComprCUCtx& cuECtx = *comprCUCtx;
  if( cuECtx.mttGate == MTT_GATE_UNDECIDED )
  {
    cuECtx.mttGate = xDecideMttGate( partitioner );
  }
  return !( cuECtx.mttGate & ( isHorizontalMttSplit( split ) ? MTT_GATE_HOR : MTT_GATE_VER ) );
}

// Keeps what the DEV MTT entry gate and census read from a node's QT split: its cost and, where it gave four unsplit
// CUs, how alike the motion of each pair of neighbouring leaves is
void EncModeCtrl::xNoteQtSplitResult( const CodingStructure& qtCS )
{
  ComprCUCtx& cuECtx = *comprCUCtx;
  cuECtx.qtSplitCost = qtCS.cost;

  const Size leafSize( qtCS.area.lwidth() >> 1, qtCS.area.lheight() >> 1 );
  bool fourLeaves = qtCS.cost != MAX_DOUBLE && qtCS.cus.size() == 4;
  for( int i = 0; fourLeaves && i < 4; i++ )
  {
    fourLeaves = qtCS.cus[i]->chType == CH_L && qtCS.cus[i]->lumaSize() == leafSize;
  }
  cuECtx.qtLeavesKnown = fourLeaves;
  if( fourLeaves )
  {
    // the four leaves are in coding order: top left, top right, bottom left, bottom right
    cuECtx.qtPairMvGap[QT_PAIR_TOP]    = motionGap( *qtCS.cus[0], *qtCS.cus[1] );
    cuECtx.qtPairMvGap[QT_PAIR_BOTTOM] = motionGap( *qtCS.cus[2], *qtCS.cus[3] );
    cuECtx.qtPairMvGap[QT_PAIR_LEFT]   = motionGap( *qtCS.cus[0], *qtCS.cus[2] );
    cuECtx.qtPairMvGap[QT_PAIR_RIGHT]  = motionGap( *qtCS.cus[1], *qtCS.cus[3] );
  }
}

int64_t EncModeCtrl::censusClock() const
{
  if( !m_censusFile && !m_searchCensusFile )
  {
    return 0;
  }
  return std::chrono::duration_cast<std::chrono::nanoseconds>( std::chrono::steady_clock::now().time_since_epoch() ).count();
}

void EncModeCtrl::censusAddSplitTime( const EncTestMode& encTestmode, const int64_t startTime )
{
  const int splitIndex = mttSplitIndex( getPartSplit( encTestmode ) );
  if( m_censusFile && splitIndex >= 0 )
  {
    comprCUCtx->censusSplitTime[splitIndex] += censusClock() - startTime;
  }
  if( devSearchIndex( encTestmode.type ) >= 0 )
  {
    devAddTime( DevTimer( devSearchIndex( encTestmode.type ) ), startTime );
  }
}

void EncModeCtrl::devAddTime( const DevTimer timer, const int64_t startTime )
{
  if( m_searchCensusFile && comprCUCtx )
  {
    comprCUCtx->dev.time[timer] += censusClock() - startTime;
  }
}

static bool isBtSplit( const PartSplit split )
{
  return split == CU_HORZ_SPLIT || split == CU_VERT_SPLIT;
}

// Index of a split's direction in ComprCUCtx::devBtHalfCost / devBtStopped
static int splitDirection( const PartSplit split )
{
  return isHorizontalMttSplit( split ) ? 0 : 1;
}

// DEV split stop and split census (EncModeCtrl.h), before the first part of a split of the current node is searched
// (the partitioner still stands on the node). hasModeConstraint: the parts are searched as inter-only or intra-only.
void EncModeCtrl::devStartSplit( DevSplitTry& splitTry, const EncTestMode& encTestmode, const CodingStructure& bestCS, const Partitioner& partitioner,
                                 const int64_t splitBits, const bool hasModeConstraint )
{
  const bool inCensus = m_searchCensusFile && m_devCtuInCensus;
  if( ( m_devStopT <= 0.0 && !inCensus ) || !isLuma( partitioner.chType ) || !comprCUCtx )
  {
    return;
  }
  const PartSplit    split   = getPartSplit( encTestmode );
  const DevSplitKind kind    = split == CU_QUAD_SPLIT ? DEV_SPLIT_KIND_QT : isBtSplit( split ) ? DEV_SPLIT_KIND_BT : DEV_SPLIT_KIND_TT;
  const bool         hasBest = bestCS.cost != MAX_DOUBLE && !bestCS.cus.empty();
  splitTry.active      = true;
  splitTry.testMode    = encTestmode;
  splitTry.nodeArea    = partitioner.currArea().Y();
  splitTry.splitBits   = splitBits;
  splitTry.bestCost    = hasBest ? bestCS.cost + bestCS.costDbOffset : MAX_DOUBLE;
  // The rule needs a best cost to compare with and a split whose parts are all searched: none outside the picture,
  // no second pass with another mode constraint, no chroma search after the luma parts (which such a constraint brings)
  splitTry.stoppable   = hasBest && !bestCS.slice->isIntra() && partitioner.treeType == TREE_D && !hasModeConstraint
                      && bestCS.picture->Y().contains( splitTry.nodeArea );
  splitTry.ruleOn      = splitTry.stoppable && m_devStopT > 0.0 && ( m_devStopSplits >> kind & 1 ) && ( m_devStopMtDepths >> partitioner.currMtDepth & 1 );
  splitTry.firstNodeId = m_devNodeCount;
  splitTry.startTime   = censusClock();
}

// DEV split stop, VVENC_DEV_SPLIT_STOP_DIR2 (EncModeCtrl.h): true where the vertical BT split of the current node is
// not searched because the horizontal one, searched before it, lost by too much
bool EncModeCtrl::xSecondDirectionStops( const CodingStructure& cs, const Partitioner& partitioner ) const
{
  const ComprCUCtx& cuECtx = *comprCUCtx;
  return m_devStopDir2 > 0.0 && !cs.slice->isIntra() && isLuma( partitioner.chType ) && ( m_devStopMtDepths >> partitioner.currMtDepth & 1 )
      && cuECtx.bestCS && cuECtx.didHorzSplit && cuECtx.bestCostHorzSplit != MAX_DOUBLE
      && cuECtx.bestCostHorzSplit > m_devStopDir2 * cuECtx.bestCS->cost;
}

// DEV split stop: the estimated cost of the parts of a split that are not searched yet (EncModeCtrl.h, DevRestShare)
double EncModeCtrl::xRestOfSplitEstimate( const DevSplitTry& splitTry ) const
{
  const ComprCUCtx& cuECtx    = *comprCUCtx;
  const PartSplit   split     = getPartSplit( splitTry.testMode );
  const int         direction = splitDirection( split );
  const double*     btHalf    = cuECtx.devBtHalfCost[direction];
  if( m_devStopShare == DEV_REST_BY_BT_PARTS && ( split == CU_TRIH_SPLIT || split == CU_TRIV_SPLIT ) && btHalf[0] != MAX_DOUBLE )
  {
    // A TT split's parts are a quarter, a half and a quarter of the block: after the first, half of the BT split's
    // first half and all of its second half remain; after the second, half of its second half
    return splitTry.partsDone == 1 ? 0.5 * btHalf[0] + btHalf[1] : 0.5 * btHalf[1];
  }

  double     restShare = 1.0 - ( double ) splitTry.lumaDone / splitTry.nodeArea.area();
  const int* strips    = direction == 0 ? cuECtx.dev.predErrorRows : cuECtx.dev.predErrorCols;   // the strips a split of this direction cuts along
  if( m_devStopShare != DEV_REST_BY_AREA && split != CU_QUAD_SPLIT && strips[0] >= 0 )
  {
    const int errorOfBlock = std::accumulate( strips, strips + NUM_ERROR_STRIPS, 0 );
    const int errorDone    = std::accumulate( strips, strips + splitTry.quartersDone, 0 );
    if( errorOfBlock > 0 )
    {
      restShare = 1.0 - ( double ) errorDone / errorOfBlock;
    }
  }
  const bool useNoSplitCost = m_devStopBase == 0 && cuECtx.bestCostBeforeSplit != MAX_DOUBLE;
  return restShare * ( useNoSplitCost ? cuECtx.bestCostBeforeSplit : splitTry.bestCost );
}

// DEV split stop and split census, after a part of the split was searched and taken into splitCS. True where the
// rule ends the search of the split here.
bool EncModeCtrl::devSplitPartDone( DevSplitTry& splitTry, const CodingStructure& splitCS, const Area& partArea, const bool isLastPart )
{
  if( !splitTry.active || splitTry.partsDone == MAX_SPLIT_PARTS )
  {
    return false;
  }
  const Area& node = splitTry.nodeArea;
  const int   part = splitTry.partsDone++;
  splitTry.costAfterPart[part]  = m_pcRdCost->calcRdCost( splitCS.fracBits + splitTry.splitBits, splitCS.dist ) + splitCS.costDbOffset;
  splitTry.timeAfterPart[part]  = censusClock() - splitTry.startTime;
  splitTry.nodesAfterPart[part] = m_devNodeCount;
  splitTry.lumaDone            += partArea.area();
  splitTry.quartersDone         = isHorizontalMttSplit( getPartSplit( splitTry.testMode ) ) ? NUM_ERROR_STRIPS * ( partArea.y + partArea.height - node.y ) / node.height
                                                                                           : NUM_ERROR_STRIPS * ( partArea.x + partArea.width  - node.x ) / node.width;
  if( !splitTry.ruleOn || isLastPart )
  {
    return false;
  }
  const double estimate = splitTry.costAfterPart[part] + m_devStopRest * xRestOfSplitEstimate( splitTry );
  if( estimate <= m_devStopT * splitTry.bestCost )
  {
    return false;
  }
  splitTry.stoppedAfter = splitTry.partsDone;
  splitTry.stopEstimate = estimate;
  return true;
}

// DEV split stop and split census, at the end of the search of a split (finalCost: what the mode decision compares;
// MAX_DOUBLE for a stopped split). Keeps what the rule reads from a BT split later and writes the census line:
// P nodeId split (EncTestModeType) stoppable ruleOn partsSearched stoppedAfter (parts, 0 = not stopped) firstNodeId |
// best cost of the node before the split (-1: none), RD cost of the split flags, final cost (-1: none), the rule's
// estimate at its stop | ns in the split | for each of MAX_SPLIT_PARTS parts: RD cost of the parts up to it (with the
// split flags), ns up to it, node counter after it (-1 for a part that was not searched). The nodes below part k
// have the ids from the counter after part k - 1 (firstNodeId for the first part) up to the counter after part k.
void EncModeCtrl::devFinishSplit( const DevSplitTry& splitTry, const double finalCost )
{
  if( !splitTry.active )
  {
    return;
  }
  ComprCUCtx&     cuECtx = *comprCUCtx;
  const PartSplit split  = getPartSplit( splitTry.testMode );
  if( splitTry.stoppedAfter )
  {
    m_devCtuCounts[DEV_COUNT_SPLITS_STOPPED]++;
  }
  if( isBtSplit( split ) && splitTry.stoppedAfter )
  {
    // counts as searched for the rules of trySplit, at the cost the rule estimated
    cuECtx.devBtStopped[splitDirection( split )]              = true;
    ( split == CU_HORZ_SPLIT ? cuECtx.bestCostHorzSplit : cuECtx.bestCostVertSplit ) = splitTry.stopEstimate;
  }
  else if( isBtSplit( split ) && finalCost != MAX_DOUBLE && splitTry.partsDone == 2 )
  {
    cuECtx.devBtHalfCost[splitDirection( split )][0] = splitTry.costAfterPart[0];
    cuECtx.devBtHalfCost[splitDirection( split )][1] = splitTry.costAfterPart[1] - splitTry.costAfterPart[0];
  }

  if( m_searchCensusFile && m_devCtuInCensus )
  {
    const auto costOrNone = []( const double cost ) { return cost == MAX_DOUBLE ? -1.0 : cost; };
    fprintf( m_searchCensusFile, "P %d %d %d %d %d %d %d | %.7g %.7g %.7g %.7g | %lld |", cuECtx.dev.id, ( int ) splitTry.testMode.type,
             ( int ) splitTry.stoppable, ( int ) splitTry.ruleOn, splitTry.partsDone, splitTry.stoppedAfter, splitTry.firstNodeId,
             costOrNone( splitTry.bestCost ), m_pcRdCost->calcRdCost( splitTry.splitBits, 0 ), costOrNone( finalCost ), splitTry.stopEstimate,
             ( long long ) ( censusClock() - splitTry.startTime ) );
    for( int part = 0; part < MAX_SPLIT_PARTS; part++ )
    {
      if( part < splitTry.partsDone )
      {
        fprintf( m_searchCensusFile, " %.7g %lld %d", splitTry.costAfterPart[part], ( long long ) splitTry.timeAfterPart[part], splitTry.nodesAfterPart[part] );
      }
      else
      {
        fprintf( m_searchCensusFile, " -1 -1 -1" );
      }
    }
    fprintf( m_searchCensusFile, "\n" );
  }
}

// DEV skipped-parent gate (EncModeCtrl.h): true for a node whose best mode without a split is skip, or with
// VVENC_DEV_PARENT_GATE_LEVEL=2 any inter merge mode. False for a node without such a mode (a forced split).
bool EncModeCtrl::xIsSkippedNode( const ComprCUCtx& node ) const
{
  const bool isInterMerge = ( node.nsFlags & NS_MERGE ) && !( node.nsFlags & NS_IBC );
  return node.isBestNoSplitSkip || ( m_devParentLevel >= 2 && isInterMerge );
}

// DEV skipped-parent gate: true where it stops this search (a DevParentSearch bit other than the splits) at the
// current node, which is then a split part of a skipped node
bool EncModeCtrl::xParentGateStops( const int search )
{
  if( !( m_devParentGate >> search & 1 ) || m_ComprCUCtxList.size() < 2 || !xIsSkippedNode( *( m_ComprCUCtxList.end() - 2 ) ) )
  {
    return false;
  }
  m_devParentStops[search]++;
  comprCUCtx->dev.parentGateStopped |= 1u << search;
  return true;
}

// DEV skipped-parent gate: true where it stops this split of the current node: the node VVENC_DEV_PARENT_GATE_DEPTH
// levels above it is skipped (0: the node itself), and with VVENC_DEV_PARENT_GATE_OWN the node itself too
bool EncModeCtrl::xParentGateStopsSplit( const PartSplit split )
{
  const int    search = split == CU_QUAD_SPLIT ? ( int ) DEV_PARENT_QT : ( int ) DEV_TOOL_MTT;
  const size_t levels = m_ComprCUCtxList.size();   // the current node is the last of them
  if( !( m_devParentGate >> search & 1 ) || levels <= ( size_t ) m_devParentDepth )
  {
    return false;
  }
  const ComprCUCtx& node  = m_ComprCUCtxList.back();
  const ComprCUCtx& above = *( m_ComprCUCtxList.end() - 1 - m_devParentDepth );
  if( !xIsSkippedNode( above ) || ( m_devParentOwn && !xIsSkippedNode( node ) ) )
  {
    return false;
  }
  m_devParentStops[search]++;
  comprCUCtx->dev.parentGateStopped |= 1u << search;
  return true;
}

// DEV nearby-use gate for the intra search of an inter picture, with the v4 fallback (EncModeCtrl.h): true where the
// search is stopped at the current node, whose best mode so far is the inter CU bestCU. Keeps what the census reads.
bool EncModeCtrl::xIntraGateStops( const CodingStructure& bestCS, const CodingUnit& bestCU, const Area& lumaArea )
{
  DevNodeStats& dev = comprCUCtx->dev;
  dev.interCostBeforeIntra  = bestCS.cost;
  dev.interFlagsBeforeIntra = ( bestCU.skip ? NS_SKIP : 0 ) | ( bestCU.mergeFlag ? NS_MERGE : 0 ) | ( bestCU.rootCbf ? NS_RESIDUAL : 0 );
  if( xNearbyGateAllows( DEV_TOOL_INTRA ) )
  {
    dev.intraGateSaid = DEV_INTRA_USED_NEARBY;   // also where the gate is off
    return false;
  }
  const bool interIsGood = m_devIntraGateGood <= 0.0 || bestCS.cost < m_devIntraGateGood * m_pcRdCost->getLambda() * lumaArea.area();
  if( !interIsGood )
  {
    dev.toolsStopped &= ~( 1u << DEV_TOOL_INTRA );   // xToolUsedNearby noted a stop
  }
  dev.intraGateSaid = interIsGood ? DEV_INTRA_STOPPED : DEV_INTRA_KEPT_BY_FALLBACK;
  m_devCtuCounts[interIsGood ? DEV_COUNT_INTRA_STOPPED : DEV_COUNT_INTRA_KEPT]++;
  return interIsGood;
}

// DEV nearby-use gate: true where a CU nearby uses the tool, in a probe CTU and in pictures the gate leaves alone;
// elsewhere it notes that the tool was stopped at the current node
bool EncModeCtrl::xToolUsedNearby( const DevTool tool )
{
  if( !m_devGateInPicture || m_devCtuIsProbe || !comprCUCtx )
  {
    return true;
  }
  DevNodeStats& dev = comprCUCtx->dev;
  uint32_t toolsNearby = 0;
  for( int source = 0; source < NUM_DEV_NEARBY; source++ )
  {
    if( m_devToolGateSources >> source & 1 )
    {
      toolsNearby |= source == DEV_NEAR_CUS ? dev.toolsOfNearCus : source == DEV_NEAR_PARENT ? dev.toolsOfParent : m_devToolsNearCtu[source];
    }
  }
  // an affine merge CU nearby speaks for affine motion as much as one found by affine motion estimation
  const uint32_t signsOfTool = 1u << tool | ( tool == DEV_TOOL_AFFINE ? 1u << DEV_TOOL_AFFINE_MERGE : 0u );
  if( toolsNearby & signsOfTool )
  {
    return true;
  }
  dev.toolsStopped |= 1u << tool;
  return false;
}

// DEV nearby-use gate and search census, before the first node of a CTU: which CTUs nearby use which tools, and
// whether this CTU is a probe CTU / written to the census
void EncModeCtrl::devInitCtu( const CodingStructure& picCS, const UnitArea& ctuArea )
{
  std::fill_n( m_devCtuCounts, NUM_DEV_CTU_COUNTS, 0 );
  std::fill_n( m_devParentStops, NUM_DEV_PARENT_SEARCHES, 0 );
  std::fill_n( m_devV7Stops, NUM_DEV_V7_STOPS, 0 );
  if( !m_devToolGate && !m_searchCensusFile )
  {
    return;
  }
  const PreCalcValues& pcv   = *picCS.pcv;
  const Slice&         slice = *picCS.slice;
  const Picture&       pic   = *picCS.picture;
  const int            ctuX  = ctuArea.lx() >> pcv.maxCUSizeLog2;
  const int            ctuY  = ctuArea.ly() >> pcv.maxCUSizeLog2;

  m_devGateInPicture = !slice.isIntra();
  m_devCtuIsProbe    = m_devToolGateProbe > 0 && ( ctuX + 2 * ctuY + slice.poc ) % m_devToolGateProbe == 0;
  m_devCtuInCensus   = m_searchCensusFile && ( ctuX + 3 * ctuY + slice.poc ) % m_searchCensusStep == 0
                       && ( m_searchCensusPoc < 0 || m_searchCensusPoc == slice.poc );
  m_devCtuStart      = censusClock();

  // The CTUs left, above and above left in the same tile are finished before this one starts, whatever the threads do
  uint32_t toolsOfCtus = 0;
  const auto addCtu = [&]( const int x, const int y )
  {
    const size_t ctuRsAddr = y * pcv.widthInCtus + x;
    if( x >= 0 && y >= 0 && ctuRsAddr < pic.devCtuTools.size() && picCS.pps->getTileIdx( x, y ) == picCS.pps->getTileIdx( ctuX, ctuY ) )
    {
      toolsOfCtus |= pic.devCtuTools[ctuRsAddr];
    }
  };
  addCtu( ctuX - 1, ctuY );
  addCtu( ctuX, ctuY - 1 );
  addCtu( ctuX - 1, ctuY - 1 );

  // A reference picture is finished before this one starts, except with IFP
  uint32_t toolsOfRefs = 0;
  if( !slice.isIntra() && !m_pcEncCfg->m_ifpLines )
  {
    const size_t ctuRsAddr = ctuY * pcv.widthInCtus + ctuX;
    for( int list = 0; list < NUM_REF_PIC_LIST_01; list++ )
    {
      const Picture* refPic = slice.numRefIdx[list] > 0 ? slice.getRefPic( RefPicList( list ), 0 ) : nullptr;
      if( refPic && refPic->devCtuTools.size() == pic.devCtuTools.size() && ctuRsAddr < refPic->devCtuTools.size() )
      {
        toolsOfRefs |= refPic->devCtuTools[ctuRsAddr];
      }
    }
  }
  m_devToolsNearCtu[DEV_NEAR_CTUS] = toolsOfCtus;
  m_devToolsNearCtu[DEV_NEAR_REFS] = toolsOfRefs;
}

// DEV nearby-use gate and search census, with the final luma CUs of a CTU: keeps the tools they use for the CTUs and
// pictures coded later. Census line, areas in luma samples: C poc temporalLayer isIntra ctuX ctuY isProbe inCensus |
// tools (DevTool bits) | area of the CUs with each of the NUM_DEV_TOOLS tools | CUs | tools of the CTUs nearby,
// tools of the same CTU in the reference pictures | v3: ns spent in the CTU, lambda, the CTU's RD cost, distortion
// and bits (fractional bits), the CTU's QP offset from the look-ahead (BIM / CU-tree; 0 without one), the picture's
// MCTF motion error and visual activity
void EncModeCtrl::devFinishCtu( const CodingStructure& ctuCS, const UnitArea& ctuArea )
{
  if( !m_devToolGate && !m_searchCensusFile )
  {
    return;
  }
  uint32_t tools = 0;
  int      areaOfTool[NUM_DEV_TOOLS] = { 0 };
  for( const CodingUnit* cu : ctuCS.cus )
  {
    if( !isLuma( cu->chType ) )
    {
      continue;   // a chroma CU of a local dual tree
    }
    const uint32_t toolsOfCu = devToolBits( *cu );
    tools |= toolsOfCu;
    for( int tool = 0; tool < NUM_DEV_TOOLS; tool++ )
    {
      areaOfTool[tool] += ( toolsOfCu >> tool & 1 ) ? cu->lumaSize().area() : 0;
    }
  }

  const PreCalcValues& pcv  = *ctuCS.pcv;
  const int            ctuX = ctuArea.lx() >> pcv.maxCUSizeLog2;
  const int            ctuY = ctuArea.ly() >> pcv.maxCUSizeLog2;
  std::vector<uint32_t>& toolsOfPicture = ctuCS.picture->devCtuTools;
  const size_t           ctuRsAddr      = ctuY * pcv.widthInCtus + ctuX;
  if( ctuRsAddr < toolsOfPicture.size() )
  {
    toolsOfPicture[ctuRsAddr] = tools;   // each CTU has its own entry: no lock
  }

  if( m_searchCensusFile )
  {
    const Slice& slice = *ctuCS.slice;
    fprintf( m_searchCensusFile, "C %d %d %d %d %d %d %d | %u |", slice.poc, ( int ) slice.TLayer, ( int ) slice.isIntra(), ctuX, ctuY,
             ( int ) m_devCtuIsProbe, ( int ) m_devCtuInCensus, tools );
    for( int tool = 0; tool < NUM_DEV_TOOLS; tool++ )
    {
      fprintf( m_searchCensusFile, " %d", areaOfTool[tool] );
    }
    const PicShared&        picShared = *ctuCS.picture->m_picShared;
    const std::vector<int>& qpOffsets = picShared.m_ctuBimQpOffset;
    fprintf( m_searchCensusFile, " | %d | %u %u", ( int ) ctuCS.cus.size(), m_devToolsNearCtu[DEV_NEAR_CTUS], m_devToolsNearCtu[DEV_NEAR_REFS] );
    fprintf( m_searchCensusFile, " | %lld %.6g %.9g %llu %llu %d %d %d\n", ( long long ) ( censusClock() - m_devCtuStart ), m_pcRdCost->getLambda(),
             ctuCS.cost, ( unsigned long long ) ctuCS.dist, ( unsigned long long ) ctuCS.fracBits,
             ctuRsAddr < qpOffsets.size() ? qpOffsets[ctuRsAddr] : 0, ( int ) picShared.m_picMotEstError, ( int ) ctuCS.picture->picVA.visAct );
    // v4, its own line so that the C line stays as the v3 reader knows it: G poc isIntra ctuX ctuY | the
    // NUM_DEV_CTU_COUNTS counts of what the v4 switches did in the CTU (DevCtuCount)
    fprintf( m_searchCensusFile, "G %d %d %d %d |", slice.poc, ( int ) slice.isIntra(), ctuX, ctuY );
    for( int count = 0; count < NUM_DEV_CTU_COUNTS; count++ )
    {
      fprintf( m_searchCensusFile, " %d", m_devCtuCounts[count] );
    }
    fprintf( m_searchCensusFile, "\n" );
    // v5, the skipped-parent gate: H poc isIntra ctuX ctuY | how often it stopped each of the
    // NUM_DEV_PARENT_SEARCHES searches in the CTU (DevParentSearch)
    fprintf( m_searchCensusFile, "H %d %d %d %d |", slice.poc, ( int ) slice.isIntra(), ctuX, ctuY );
    for( int search = 0; search < NUM_DEV_PARENT_SEARCHES; search++ )
    {
      fprintf( m_searchCensusFile, " %d", m_devParentStops[search] );
    }
    fprintf( m_searchCensusFile, "\n" );
    // v7, the smart AMVR and SBT gates: J poc isIntra ctuX ctuY | what they stopped in the CTU (DevV7Stop)
    fprintf( m_searchCensusFile, "J %d %d %d %d | %d %d %d\n", slice.poc, ( int ) slice.isIntra(), ctuX, ctuY,
             m_devV7Stops[DEV_V7_AMVR_STOPPED], m_devV7Stops[DEV_V7_4PEL_STOPPED], m_devV7Stops[DEV_V7_SBT_TESTS_STOPPED] );
    if( m_devFinalCensus )
    {
      xWriteFinalCus( ctuCS );
    }
  }
}

// DEV search census, v8 (VVENC_DEV_FINAL_CENSUS=1 with the search census): one F line per luma CU of the CTU's final
// partition, so that two encodes can be compared decision by decision (work/smart_slow/mg8_final_census.py):
// F poc layer | x y width height qtDepth mtDepth | predMode skip merge DevTool-bits rootCbf interDir | luma SSE |
//   BCW index, then per list: reference POC (-1 none), MV x, MV y in 1/16 sample (the CU's first motion; the first
//   control point of an affine CU)
// The SSE is of the reconstruction before the loop filters.
void EncModeCtrl::xWriteFinalCus( const CodingStructure& ctuCS ) const
{
  const Slice& slice = *ctuCS.slice;
  for( const CodingUnit* cu : ctuCS.cus )
  {
    if( !isLuma( cu->chType ) )
    {
      continue;   // a chroma CU of a local dual tree
    }
    const CompArea&  luma = cu->Y();
    const Distortion sse  = m_pcRdCost->getDistPart( ctuCS.getOrgBuf( luma ), ctuCS.getRecoBuf( luma ), ctuCS.sps->bitDepths[CH_L], COMP_Y, DF_SSE );
    fprintf( m_searchCensusFile, "F %d %d | %d %d %d %d %d %d | %d %d %d %u %d %d | %llu | %d", slice.poc, ( int ) slice.TLayer,
             luma.x, luma.y, ( int ) luma.width, ( int ) luma.height, ( int ) cu->qtDepth, ( int ) cu->mtDepth,
             ( int ) cu->predMode, ( int ) cu->skip, ( int ) cu->mergeFlag, devToolBits( *cu ), ( int ) cu->rootCbf, ( int ) cu->interDir,
             ( unsigned long long ) sse, ( int ) cu->BcwIdx );
    for( const RefPicList list : { REF_PIC_LIST_0, REF_PIC_LIST_1 } )
    {
      const bool usesList = CU::isInter( *cu ) && cu->refIdx[list] >= 0;
      const Mv   mv       = usesList ? cu->mv[list][0] : Mv();
      fprintf( m_searchCensusFile, " %d %d %d", usesList ? slice.getRefPOC( list, cu->refIdx[list] ) : -1, mv.hor, mv.ver );
    }
    fprintf( m_searchCensusFile, "\n" );
  }
}

// DEV search census and even-error rule of the MTT entry gate: keeps what they read from the best mode without a split
void EncModeCtrl::xNoteNoSplitResult( const CodingUnit& bestCU, const Partitioner& partitioner )
{
  ComprCUCtx&            cuECtx = *comprCUCtx;
  const CodingStructure& bestCS = *cuECtx.bestCS;
  cuECtx.dev.toolsOfBestNoSplit = devToolBits( bestCU );

  // A mode taken from the cache is reconstructed without filling the CS's prediction buffer
  const bool errorStripsRead = m_searchCensusFile || m_devGateShape > 0.0 || m_devGateShapeDir > 0.0 || xSplitStopReadsError();
  if( errorStripsRead && isLuma( bestCU.chType ) && CU::isInter( bestCU ) && !cuECtx.isReusingCu )
  {
    const CPelBuf org  = bestCS.getOrgBuf ( COMP_Y );
    const CPelBuf pred = bestCS.getPredBuf( COMP_Y );
    if( org.width == pred.width && org.height == pred.height && org.width >= NUM_ERROR_STRIPS && org.height >= NUM_ERROR_STRIPS )
    {
      predErrorStrips( org, pred, cuECtx.dev.predErrorRows, cuECtx.dev.predErrorCols );
    }
  }
  if( m_searchCensusFile || ( m_devMttGate && xMotionGateIsOn() && xIsMttGateNode( bestCS, partitioner ) ) )
  {
    xEstimateSplitMotion( bestCU, partitioner );
  }
}

// DEV motion estimate of the splits (the "one motion" rule in EncModeCtrl.h), made once the best mode without a split
// is known: fills DevNodeStats::motionSad for a luma node of an inter picture. With IFP a reference picture may not
// be finished where a candidate points, so no estimate is made then.
void EncModeCtrl::xEstimateSplitMotion( const CodingUnit& bestCU, const Partitioner& partitioner )
{
  ComprCUCtx&            cuECtx  = *comprCUCtx;
  const CodingStructure& cs      = *cuECtx.bestCS;
  const Slice&           slice   = *cs.slice;
  const Picture&         picture = *cs.picture;
  const Area             area    = partitioner.currArea().Y();
  const CPelBuf          org     = cs.getOrgBuf( COMP_Y );
  if( slice.isIntra() || !isLuma( partitioner.chType ) || !isLuma( bestCU.chType ) || m_pcEncCfg->m_ifpLines
      || area.width < 2 * NUM_MOTION_CELLS || area.height < 2 * NUM_MOTION_CELLS || org.width != area.width || org.height != area.height
      || area.x + area.width > picture.lwidth() || area.y + area.height > picture.lheight() )
  {
    return;
  }

  static_vector<MotionCand, MAX_MOTION_CANDS> cands;
  const auto addCand = [&]( const RefPicList list, const int refIdx, const Mv& mv )
  {
    const Picture* refPic = refIdx >= 0 && refIdx < slice.numRefIdx[list] ? slice.getRefPic( list, refIdx ) : nullptr;
    if( !refPic || refPic->lwidth() != picture.lwidth() || refPic->lheight() != picture.lheight() || cands.size() == MAX_MOTION_CANDS )
    {
      return;   // no such picture, or one of another size (reference picture resampling)
    }
    const int roundToSample = 1 << ( MV_FRACTIONAL_BITS_INTERNAL - 1 );
    const MotionCand cand = { refPic,
                              Clip3( 0, ( int ) ( picture.lwidth()  - area.width  ), area.x + ( ( mv.hor + roundToSample ) >> MV_FRACTIONAL_BITS_INTERNAL ) ),
                              Clip3( 0, ( int ) ( picture.lheight() - area.height ), area.y + ( ( mv.ver + roundToSample ) >> MV_FRACTIONAL_BITS_INTERNAL ) ) };
    if( std::find( cands.begin(), cands.end(), cand ) == cands.end() )
    {
      cands.push_back( cand );
    }
  };
  const auto addMotionOfCu = [&]( const CodingUnit* cu )
  {
    for( int list = 0; cu && CU::isInter( *cu ) && list < NUM_REF_PIC_LIST_01; list++ )
    {
      if( cu->interDir & ( 1 << list ) )
      {
        addCand( RefPicList( list ), cu->refIdx[list], cu->mv[list][0] );
      }
    }
  };

  // the motion of the best mode without a split, then of the CUs at four places along the left and the upper edge
  addMotionOfCu( &bestCU );
  for( int i = 0; i < 4; i++ )
  {
    const Position atLeft ( area.x - 1, area.y + ( 2 * i + 1 ) * ( int ) area.height / 8 );
    const Position atAbove( area.x + ( 2 * i + 1 ) * ( int ) area.width / 8, area.y - 1 );
    if( atLeft.x >= 0 )
    {
      addMotionOfCu( cs.getCURestricted( atLeft, bestCU, CH_L ) );
    }
    if( atAbove.y >= 0 )
    {
      addMotionOfCu( cs.getCURestricted( atAbove, bestCU, CH_L ) );
    }
  }
  // the temporal candidates at the centres of the four quadrants, towards the first reference picture of each list
  for( int list = 0; cs.picHeader->enableTMVP && list < NUM_REF_PIC_LIST_01; list++ )
  {
    for( int quadrant = 0; slice.numRefIdx[list] > 0 && quadrant < 4; quadrant++ )
    {
      const Position centre( area.x + ( 2 * ( quadrant & 1 ) + 1 ) * ( int ) area.width / 4, area.y + ( 2 * ( quadrant >> 1 ) + 1 ) * ( int ) area.height / 4 );
      Mv mv;
      if( CU::getColocatedMVP( bestCU, RefPicList( list ), centre, mv, 0 ) )
      {
        addCand( RefPicList( list ), 0, mv );
      }
    }
  }
  if( cands.empty() )
  {
    return;
  }

  MotionCellSad sadOfCand[MAX_MOTION_CANDS];
  const int numCands = ( int ) cands.size();
  for( int i = 0; i < numCands; i++ )
  {
    motionCellSad( org, cands[i].refPic->getRecoBuf( COMP_Y ), cands[i], sadOfCand[i] );
  }
  // SAD of a partition whose parts are the cell ranges between the given edges, each with its best candidate
  const auto partitionSad = [&]( const std::initializer_list<int> rowEdges, const std::initializer_list<int> colEdges )
  {
    int sum = 0;
    for( auto row = rowEdges.begin(); row + 1 != rowEdges.end(); row++ )
    {
      for( auto col = colEdges.begin(); col + 1 != colEdges.end(); col++ )
      {
        sum += bestPartSad( sadOfCand, numCands, row[0], row[1], col[0], col[1] );
      }
    }
    return sum;
  };
  int* motionSad = cuECtx.dev.motionSad;
  motionSad[DEV_MOTION_NONE] = partitionSad( { 0, 4 },       { 0, 4 } );
  motionSad[DEV_MOTION_BT_H] = partitionSad( { 0, 2, 4 },    { 0, 4 } );
  motionSad[DEV_MOTION_BT_V] = partitionSad( { 0, 4 },       { 0, 2, 4 } );
  motionSad[DEV_MOTION_TT_H] = partitionSad( { 0, 1, 3, 4 }, { 0, 4 } );
  motionSad[DEV_MOTION_TT_V] = partitionSad( { 0, 4 },       { 0, 1, 3, 4 } );
  motionSad[DEV_MOTION_QT]   = partitionSad( { 0, 2, 4 },    { 0, 2, 4 } );
  cuECtx.dev.motionCands     = numCands;
}

// DEV search census: keeps the cost of an RD-tested mode for its search and, for a mode without a split, for the tools
// it uses and does not use
void EncModeCtrl::xNoteModeResult( const EncTestMode& encTestmode, const CodingStructure& tempCS, const bool useEDO )
{
  const int search = devSearchIndex( encTestmode.type );
  if( comprCUCtx )
  {
    comprCUCtx->dev.lastNotedCost = MAX_DOUBLE;
  }
  if( search < 0 || tempCS.cost == MAX_DOUBLE || !comprCUCtx )
  {
    return;
  }
  DevNodeStats& dev  = comprCUCtx->dev;
  const double  cost = tempCS.cost + ( useEDO ? tempCS.costDbOffset : 0 );   // what useModeResult compares
  dev.lastNotedCost      = cost;
  dev.searchCost[search] = std::min( dev.searchCost[search], cost );
  if( !isModeSplit( encTestmode ) && !tempCS.cus.empty() )
  {
    const CodingUnit& cu    = *tempCS.cus.front();
    const uint32_t    tools = devToolBits( cu );
    for( int tool = 0; tool < NUM_DEV_TOOLS; tool++ )
    {
      double& bestCost = ( tools >> tool & 1 ) ? dev.costWith[tool] : dev.costWithout[tool];
      bestCost         = std::min( bestCost, cost );
    }
    // v7, smart AMVR: the motion search with quarter-sample MVDs and equal weights, and AMVR by MVD precision
    if( encTestmode.type == ETM_INTER_ME && cu.imv == IMV_OFF && cu.BcwIdx == BCW_DEFAULT && cost < dev.meCost )
    {
      dev.meCost     = cost;
      dev.meFracBits = tempCS.fracBits;
    }
    if( encTestmode.type == ETM_INTER_IMV && cu.imv >= IMV_FPEL && cu.imv <= IMV_HPEL )
    {
      dev.imvCost[cu.imv - IMV_FPEL] = std::min( dev.imvCost[cu.imv - IMV_FPEL], cost );
    }
  }
}

// Bins of one MVD component as VVC codes it (abs_mvd_greater0, abs_mvd_greater1, the rest as an Exp-Golomb code of
// order 1, sign), each bin counted as one bit
static int mvdComponentBits( const int mvd )
{
  const int absMvd = abs( mvd );
  if( absMvd == 0 )
  {
    return 1;
  }
  if( absMvd == 1 )
  {
    return 3;
  }
  return 3 + 2 * floorLog2( ( unsigned ) absMvd );   // EG1 of absMvd - 2 has 2 * floor( log2( absMvd ) ) bins
}

// The MVDs a CU found by motion search codes: per list in use and per control point (one unless affine). The MVDs of
// the second list of an SMVD CU are not coded.
static DevMvdStats devMvdStats( const CodingUnit& cu )
{
  static constexpr int QUARTER_SHIFT = MV_FRACTIONAL_BITS_INTERNAL - 2;   // MVDs are kept in 1/16 samples
  static constexpr int WHOLE_SHIFT   = MV_FRACTIONAL_BITS_INTERNAL;
  DevMvdStats stats;
  stats.largest = 0;
  stats.biPred  = cu.interDir == 3;
  stats.affine  = cu.affine;
  stats.smvd    = cu.smvdMode != 0;
  const int controlPoints = !cu.affine ? 1 : cu.affineType == AFFINEMODEL_6PARAM ? 3 : 2;
  for( int list = 0; list < NUM_REF_PIC_LIST_01; list++ )
  {
    if( !( cu.interDir >> list & 1 ) || ( list == REF_PIC_LIST_1 && cu.smvdMode ) )
    {
      continue;
    }
    for( int point = 0; point < controlPoints; point++ )
    {
      for( const int mvd : { cu.mvd[list][point].hor, cu.mvd[list][point].ver } )
      {
        const int quarters = abs( mvd ) >> QUARTER_SHIFT;
        const int whole    = ( abs( mvd ) + ( 1 << ( WHOLE_SHIFT - 1 ) ) ) >> WHOLE_SHIFT;
        stats.largest    = std::max( stats.largest, quarters );
        stats.bits      += mvdComponentBits( quarters );
        stats.bitsWhole += mvdComponentBits( whole );
      }
    }
  }
  return stats;
}

void EncModeCtrl::devNoteMeResult( const CodingUnit& cu )
{
  if( ( m_searchCensusFile || xAmvrGateIsOn() ) && comprCUCtx )
  {
    comprCUCtx->dev.meMvd = devMvdStats( cu );
  }
}

void EncModeCtrl::devNoteBeforeImv( const CodingStructure& bestCS )
{
  if( !m_searchCensusFile || !comprCUCtx || bestCS.cus.empty() )
  {
    return;
  }
  const CodingUnit& bestCU = *bestCS.cus.front();
  DevNodeStats&     dev    = comprCUCtx->dev;
  dev.costBeforeImv  = bestCS.cost;
  dev.flagsBeforeImv = ( bestCU.skip ? NS_SKIP : 0 ) | ( bestCU.mergeFlag ? NS_MERGE : 0 ) | ( bestCU.rootCbf ? NS_RESIDUAL : 0 )
                     | ( CU::isIntra( bestCU ) ? NS_INTRA : 0 );
}

bool EncModeCtrl::devAmvrGateStops( const CodingStructure& bestCS )
{
  devNoteBeforeImv( bestCS );   // search census: the AMVR search is asked for here
  if( !xAmvrGateIsOn() || !comprCUCtx || bestCS.cus.empty() )
  {
    return false;
  }
  DevNodeStats&     dev         = comprCUCtx->dev;
  const CodingUnit& bestCU      = *bestCS.cus.front();
  const bool        hasMeResult = dev.meMvd.largest >= 0 && dev.meCost != MAX_DOUBLE;
  const bool        stops       = ( m_devAmvrGateSkip && bestCU.skip )
                               || ( m_devAmvrGateBehind > 0.0 && hasMeResult && dev.meCost > m_devAmvrGateBehind * bestCS.cost )
                               || ( m_devAmvrGateMvd > 0 && hasMeResult && dev.meMvd.largest < m_devAmvrGateMvd );
  if( stops )
  {
    dev.amvrGateStopped = true;
    m_devV7Stops[DEV_V7_AMVR_STOPPED]++;
  }
  return stops;
}

bool EncModeCtrl::devAmvrGateStops4Pel()
{
  if( m_devAmvrGate4PelMvd <= 0 || !comprCUCtx || comprCUCtx->dev.meMvd.largest < 0 || comprCUCtx->dev.meMvd.largest >= m_devAmvrGate4PelMvd )
  {
    return false;
  }
  m_devV7Stops[DEV_V7_4PEL_STOPPED]++;   // once per bi-prediction weight of the 4-sample search
  return true;
}

bool EncModeCtrl::devSbtGateStops( const int sbtMode, const Distortion estimatedDist, const Distortion blockSse )
{
  if( m_devSbtGate <= 0.0 || estimatedDist == MAX_DISTORTION || blockSse == MAX_DISTORTION || blockSse == 0 )
  {
    return false;
  }
  // InterSearch::xCalcMinDistSbt: estimate = error of the uncoded part + error of the coded part / 32
  const double uncodedSse       = ( 32.0 * ( double ) estimatedDist - ( double ) blockSse ) / 31.0;
  const double uncodedAreaShare = sbtMode < SBT_VER_Q0 ? 0.5 : 0.75;   // half modes, quarter modes
  if( uncodedSse <= m_devSbtGate * uncodedAreaShare * ( double ) blockSse )
  {
    return false;
  }
  m_devV7Stops[DEV_V7_SBT_TESTS_STOPPED]++;
  return true;
}

// DEV search census (v7), smart SBT: one line per RD test of an SBT mode at a node of a census CTU, costs as -1 where
// there is none: B id | test mode (EncTestModeType) sbtMode modeFromHistory | the mode's estimated least distortion
// (InterSearch::xCalcMinDistSbt: the part left without a residual keeps its squared error, the coded part 1 / 32 of
// it) and the squared error of the whole block, both of the prediction, chroma weighted | cost of the same motion
// without SBT, best cost of the node before the test, cost of the test | ns in the test
void EncModeCtrl::devNoteSbtTest( const EncTestMode& encTestmode, const int sbtMode, const Distortion estimatedDist, const Distortion blockSse,
                                  const double sbtOffCost, const double bestCostBefore, const bool modeFromHistory, const int64_t startTime )
{
  if( !devCensusOfNode() )
  {
    return;
  }
  const auto   costOrNone = []( const double cost ) { return cost == MAX_DOUBLE ? -1.0 : cost; };
  const bool   known      = estimatedDist != MAX_DISTORTION && blockSse != MAX_DISTORTION;
  fprintf( m_searchCensusFile, "B %d | %d %d %d | %lld %lld | %.7g %.7g %.7g | %lld\n", comprCUCtx->dev.id, ( int ) encTestmode.type, sbtMode,
           ( int ) modeFromHistory, known ? ( long long ) estimatedDist : -1ll, known ? ( long long ) blockSse : -1ll, costOrNone( sbtOffCost ),
           costOrNone( bestCostBefore ), costOrNone( comprCUCtx->dev.lastNotedCost ), ( long long ) ( censusClock() - startTime ) );
}

// DEV search census: one line for a finished luma node. Fields, costs as -1 where there is none, times in ns:
// S id parentId poc temporalLayer isIntra qp x y width height qtDepth mtDepth mttDepthLimit lambda |
// noSplitCost noSplitType (EncTestModeType) nsFlags (NsFlag bits) toolsOfBestNoSplit (DevTool bits) |
// finalCost winner (EncTestModeType) | time of each of the NUM_DEV_TIMERS timers | best cost of each of the
// NUM_DEV_SEARCHES searches | best cost with each of the NUM_DEV_TOOLS tools | best cost without each |
// toolsOfNearCus toolsOfNearCtus toolsOfRefCtus isProbeCtu toolsStopped mttGate (MttGate, 4 = not decided) |
// left CU: width height qtDepth mtDepth | above CU: the same | prediction error in 4 row strips, 4 column strips |
// time in the node | v3: isGateNode (the MTT entry gate would decide here, by VVENC_DEV_MTT_GATE_LIMIT / _MTDEPTHS)
// qtBeforeBt | motion estimate: SAD of the NUM_DEV_MOTION_SPLITS partitions, candidates | toolsOfParent
void EncModeCtrl::xWriteSearchCensusNode( const Partitioner& partitioner )
{
  const ComprCUCtx& cuECtx = *comprCUCtx;
  if( !cuECtx.bestCS || partitioner.chType != CH_L )
  {
    return;
  }
  const CodingStructure& cs   = *cuECtx.bestCS;
  const DevNodeStats&    dev  = cuECtx.dev;
  const Area             area = partitioner.currArea().Y();
  FILE* const            file = m_searchCensusFile;
  const auto writeCosts = [file]( const double* costs, const int count )
  {
    for( int i = 0; i < count; i++ )
    {
      fprintf( file, " %.7g", costs[i] == MAX_DOUBLE ? -1.0 : costs[i] );
    }
    fprintf( file, " |" );
  };
  const auto writeInts = [file]( const int* values, const int count )
  {
    for( int i = 0; i < count; i++ )
    {
      fprintf( file, " %d", values[i] );
    }
  };

  fprintf( file, "S %d %d %d %d %d %d %d %d %d %d %d %d %d %.6g |", dev.id, dev.parentId,
           cs.slice->poc, ( int ) cs.slice->TLayer, ( int ) cs.slice->isIntra(), ( int ) cs.cus.front()->qp,
           area.x, area.y, ( int ) area.width, ( int ) area.height, ( int ) partitioner.currQtDepth, ( int ) partitioner.currMtDepth,
           ( int ) cs.pcv->getMaxMTTDepth( *cs.slice, CH_L ), m_pcRdCost->getLambda() );
  fprintf( file, " %.7g %d %d %u | %.7g %d |", cuECtx.bestCostBeforeSplit == MAX_DOUBLE ? -1.0 : cuECtx.bestCostBeforeSplit,
           ( int ) cuECtx.bestNsPredMode.type, ( int ) cuECtx.nsFlags, dev.toolsOfBestNoSplit, cs.cost, ( int ) cuECtx.bestMode.type );
  for( int timer = 0; timer < NUM_DEV_TIMERS; timer++ )
  {
    fprintf( file, " %lld", ( long long ) dev.time[timer] );
  }
  fprintf( file, " |" );
  writeCosts( dev.searchCost,  NUM_DEV_SEARCHES );
  writeCosts( dev.costWith,    NUM_DEV_TOOLS );
  writeCosts( dev.costWithout, NUM_DEV_TOOLS );
  fprintf( file, " %u %u %u %d %u %d |", dev.toolsOfNearCus, m_devToolsNearCtu[DEV_NEAR_CTUS], m_devToolsNearCtu[DEV_NEAR_REFS],
           ( int ) m_devCtuIsProbe, dev.toolsStopped, ( int ) cuECtx.mttGate );
  for( const DevNeighbourCu* neighbour : { &dev.left, &dev.above } )
  {
    fprintf( file, " %d %d %d %d |", neighbour->width, neighbour->height, neighbour->qtDepth, neighbour->mtDepth );
  }
  writeInts( dev.predErrorRows, NUM_ERROR_STRIPS );
  writeInts( dev.predErrorCols, NUM_ERROR_STRIPS );
  fprintf( file, " | %lld | %d %d |", ( long long ) ( censusClock() - cuECtx.censusNodeStart ), ( int ) xIsMttGateNode( cs, partitioner ), ( int ) cuECtx.qtBeforeBt );
  writeInts( dev.motionSad, NUM_DEV_MOTION_SPLITS );
  fprintf( file, " %d | %u\n", dev.motionCands, dev.toolsOfParent );
  if( dev.interCostBeforeIntra != MAX_DOUBLE )
  {
    // v4, where the intra search of an inter picture was asked for after an inter mode:
    // I id | cost of the best inter mode then, its NsFlag bits, what the nearby-use gate said (DevIntraGate)
    fprintf( file, "I %d | %.7g %d %d\n", dev.id, dev.interCostBeforeIntra, ( int ) dev.interFlagsBeforeIntra, ( int ) dev.intraGateSaid );
  }
  // v5: K id | ns in the MMVD SATD pass, ns in the affine / sub-block merge list and SATD pass (both inside the merge
  // search's time) | DevParentSearch bits the skipped-parent gate stopped at this node
  fprintf( file, "K %d | %lld %lld | %u\n", dev.id, ( long long ) dev.time[DEV_PART_MMVD], ( long long ) dev.time[DEV_PART_AFFINE_MERGE], dev.parentGateStopped );
  if( dev.meMvd.largest >= 0 || dev.costBeforeImv != MAX_DOUBLE )
  {
    // v7, smart AMVR, where the node has a motion search result or its AMVR search was asked for:
    // A id | the motion search with quarter-sample MVDs and equal weights: largest MVD component in quarter samples
    // (-1: no result), MVD bins with quarter-sample precision, MVD bins when rounded to whole samples, biPred affine
    // smvd, RD cost, fractional bits | best cost and its NsFlag bits when the AMVR search was asked for |
    // ns of the AMVR search with whole-sample, 4-sample, half-sample MVDs | best cost with each |
    // the smart AMVR gate stopped the AMVR search
    const DevMvdStats& mvd = dev.meMvd;
    fprintf( file, "A %d | %d %d %d %d %d %d %.7g %llu | %.7g %d | %lld %lld %lld |", dev.id, mvd.largest, mvd.bits, mvd.bitsWhole,
             ( int ) mvd.biPred, ( int ) mvd.affine, ( int ) mvd.smvd, dev.meCost == MAX_DOUBLE ? -1.0 : dev.meCost,
             ( unsigned long long ) dev.meFracBits, dev.costBeforeImv == MAX_DOUBLE ? -1.0 : dev.costBeforeImv, ( int ) dev.flagsBeforeImv,
             ( long long ) dev.time[DEV_PART_IMV_FPEL], ( long long ) dev.time[DEV_PART_IMV_4PEL], ( long long ) dev.time[DEV_PART_IMV_HPEL] );
    writeCosts( dev.imvCost, NUM_IMV_PRECISIONS );
    fprintf( file, " %d\n", ( int ) dev.amvrGateStopped );
  }
}

// DEV MTT census: one line for a finished luma node of the QT tree in an inter picture. Fields, costs as -1 where
// there is none: N poc temporalLayer qp x y width height qtDepth lambda | noSplitCost nsFlags (NsFlag bits) |
// qtCost qtBeforeBt qtLeavesKnown gapTop gapBottom gapLeft gapRight | cost of BT hor, BT ver, TT hor, TT ver |
// ns spent in each of those four | ns spent in the node | finalCost winner (EncTestModeType) |
// isGateNode gateDecision (MttGate, 4 = the gate did not decide)
void EncModeCtrl::xWriteCensusNode( const Partitioner& partitioner )
{
  const ComprCUCtx& cuECtx = *comprCUCtx;
  if( !cuECtx.bestCS || cuECtx.bestCS->slice->isIntra() || partitioner.chType != CH_L || partitioner.currMtDepth != 0 )
  {
    return;
  }
  const CodingStructure& cs   = *cuECtx.bestCS;
  const Area             area = partitioner.currArea().Y();
  const auto costOrNone = []( const double cost ) { return cost == MAX_DOUBLE ? -1.0 : cost; };

  fprintf( m_censusFile, "N %d %d %d %d %d %d %d %d %.6g | %.9g %d | %.9g %d %d %d %d %d %d |",
           cs.slice->poc, ( int ) cs.slice->TLayer, ( int ) cs.cus.front()->qp, area.x, area.y, ( int ) area.width, ( int ) area.height,
           ( int ) partitioner.currQtDepth, m_pcRdCost->getLambda(),
           costOrNone( cuECtx.bestCostBeforeSplit ), ( int ) cuECtx.nsFlags,
           costOrNone( cuECtx.qtSplitCost ), ( int ) cuECtx.qtBeforeBt, ( int ) cuECtx.qtLeavesKnown,
           cuECtx.qtPairMvGap[QT_PAIR_TOP], cuECtx.qtPairMvGap[QT_PAIR_BOTTOM], cuECtx.qtPairMvGap[QT_PAIR_LEFT], cuECtx.qtPairMvGap[QT_PAIR_RIGHT] );
  for( int i = 0; i < NUM_MTT_SPLITS; i++ )
  {
    fprintf( m_censusFile, " %.9g", costOrNone( cuECtx.censusSplitCost[i] ) );
  }
  fprintf( m_censusFile, " |" );
  for( int i = 0; i < NUM_MTT_SPLITS; i++ )
  {
    fprintf( m_censusFile, " %lld", ( long long ) cuECtx.censusSplitTime[i] );
  }
  fprintf( m_censusFile, " | %lld | %.9g %d | %d %d\n",
           ( long long ) ( censusClock() - cuECtx.censusNodeStart ), cs.cost, ( int ) cuECtx.bestMode.type,
           ( int ) xIsMttGateNode( cs, partitioner ), ( int ) cuECtx.mttGate );
}

void EncModeCtrl::finishCULevel( Partitioner &partitioner )
{
  if( m_censusFile )
  {
    xWriteCensusNode( partitioner );
  }
  if( m_searchCensusFile && m_devCtuInCensus )
  {
    xWriteSearchCensusNode( partitioner );
  }
  if( m_deepSplitPass == DSP_CAPPED && xChoosesSplitsToRefine( partitioner ) )
  {
    xStoreSplitsToRefine( partitioner );
  }
  if( m_devSplitCover && comprCUCtx->deepSplitsSearched )
  {
    xStampDeepSplitResult( partitioner.currArea().Y(), *comprCUCtx );
  }

  m_ComprCUCtxList.pop_back();
  comprCUCtx = m_ComprCUCtxList.size() ? &m_ComprCUCtxList.back() : nullptr;
}

bool EncModeCtrl::trySplit( const EncTestMode& encTestmode, const CodingStructure &cs, Partitioner& partitioner, const EncTestMode& lastTestmode )
{
  ComprCUCtx& cuECtx = *comprCUCtx;

  const PartSplit implicitSplit = partitioner.getImplicitSplit( cs );
  const bool isBoundary         = implicitSplit != CU_DONT_SPLIT;

  if( ( m_pcEncCfg->m_IntraPeriod == 1 ) && ( partitioner.chType == CH_C ) && ( !cuECtx.doQtChromaSplit ) )
  {
    cuECtx.maxDepth         = partitioner.currDepth;
  }

  if( isBoundary )
  {
    if( encTestmode.type != ETM_SPLIT_QT )
    {
      return getPartSplit( encTestmode ) == implicitSplit;
    }
    else
    {
      return partitioner.canSplit( CU_QUAD_SPLIT, cs );
    }
  }

  const Slice&           slice       = *cs.slice;
  const uint32_t         width       = partitioner.currArea().lumaSize().width;
  const CodingStructure *bestCS      = cuECtx.bestCS;
  const CodingUnit      *bestCU      = cuECtx.bestCU;

  if( cuECtx.minDepth > partitioner.currQtDepth && partitioner.canSplit( CU_QUAD_SPLIT, cs ) )
  {
    // enforce QT
    return encTestmode.type == ETM_SPLIT_QT;
  }
  else if( encTestmode.type == ETM_SPLIT_QT && cuECtx.maxDepth <= partitioner.currQtDepth )
  {
    // don't check this QT depth
    return false;
  }

  CHECK( ! isModeSplit( encTestmode ), "wrong method" );

  // DEV split refinement: in the refinement pass a child of the MTT root searches again exactly the splits noted for
  // it at the end of the capped pass. The rules below are not asked a second time: they let these splits through in
  // the capped pass, and now they would read the results of sibling splits that are not searched again.
  // (With an MTT depth limit of 2 nothing was noted: the child's splits are the ones left out, and the rules below
  // decide on them now for the first time.)
  if( m_deepSplitPass == DSP_REFINE && xChoosesSplitsToRefine( partitioner ) )
  {
    return ( getBlkInfo( partitioner.currArea() ).refineSplits >> getPartSplit( encTestmode ) ) & 1;
  }

  //////////////////////////////////////////////////////////////////////////
  // skip-history rule - don't split further if at least for three past levels
  //                     in the split tree it was found that skip is the best mode
  //////////////////////////////////////////////////////////////////////////
  int skipScore = 0;

  if ((!slice.isIntra() || slice.sps->IBC) && cuECtx.isBestNoSplitSkip )
  {
    for( int i = 2; i <= m_ComprCUCtxList.size(); i++ )
    {
      if( ( m_ComprCUCtxList.end() - i )->isBestNoSplitSkip )
      {
        skipScore += 1;
      }
      else
      {
        break;
      }
    }
  }

  const PartSplit split = getPartSplit( encTestmode );
  if( !partitioner.canSplit( split, cs ) || skipScore >= 2 || ( skipScore == 1 && m_ComprCUCtxList.size() == 2 ) )
  {
    if( split == CU_HORZ_SPLIT ) cuECtx.didHorzSplit = false;
    if( split == CU_VERT_SPLIT ) cuECtx.didVertSplit = false;
    if( split == CU_QUAD_SPLIT ) cuECtx.didQuadSplit = false;
    return false;
  }

  if( isChroma( partitioner.chType ) && !cuECtx.doHorChromaSplit && ( split == CU_HORZ_SPLIT || split == CU_TRIH_SPLIT ) )
  {
    if( split == CU_HORZ_SPLIT )
    {
      cuECtx.didHorzSplit = false;
    }

    return false;
  }

  if( isChroma( partitioner.chType ) && !cuECtx.doVerChromaSplit && ( split == CU_VERT_SPLIT || split == CU_TRIV_SPLIT ) )
  {
    if( split == CU_VERT_SPLIT )
    {
      cuECtx.didVertSplit = false;
    }

    return false;
  }

  if( m_pcEncCfg->m_contentBasedFastQtbt )
  {
    const CompArea& currArea = partitioner.currArea().Y();
    int cuHeight  = currArea.height;
    int cuWidth   = currArea.width;

    const bool condIntraInter = m_pcEncCfg->m_IntraPeriod == 1 ? ( partitioner.currBtDepth == 0 ) : ( cuHeight > 32 && cuWidth > 32 );

    if( cuWidth == cuHeight && condIntraInter && split != CU_QUAD_SPLIT )
    {
      const double th1 = m_pcEncCfg->m_IntraPeriod == 1 ?  1.2              :  1.0;
      const double th2 = m_pcEncCfg->m_IntraPeriod == 1 ? (1.2 / sqrt( 2 )) : (1.0 / sqrt( 2 ));

      if( cuECtx.grad_horVal > th1 * cuECtx.grad_verVal && cuECtx.grad_horVal > th2 * cuECtx.grad_dowVal && cuECtx.grad_horVal > th2 * cuECtx.grad_dupVal && ( split == CU_HORZ_SPLIT || split == CU_TRIH_SPLIT ) )
      {
        return false;
      }
      if( th2 * cuECtx.grad_dupVal < cuECtx.grad_verVal && th2 * cuECtx.grad_dowVal < cuECtx.grad_verVal && th1 * cuECtx.grad_horVal < cuECtx.grad_verVal && ( split == CU_VERT_SPLIT || split == CU_TRIV_SPLIT ) )
      {
        return false;
      }
    }

    if( m_pcEncCfg->m_IntraPeriod == 1 && cuWidth <= 32 && cuHeight <= 32 && bestCS && bestCS->tus.size() == 1 && bestCU && bestCU->depth == partitioner.currDepth && partitioner.currBtDepth > 1 && isLuma( partitioner.chType ) )
    {
      if( !bestCU->rootCbf )
      {
        return false;
      }
    }
  }

  if( m_pcEncCfg->m_qtbttSpeedUp > 1 )
  {
    const int availDepth = xRootRuleLimit( DRR_SKIP_DEPTH, partitioner, cuECtx.mttDepthLimit ) - partitioner.currMtDepth;
    if( bestCU && bestCU->skip && availDepth <= ( 3 - m_skipThresholdE0023FastEnc ) && !isModeSplit( lastTestmode ) && split != CU_QUAD_SPLIT )
    {
      return false;
    }
  }
  if( bestCU && bestCU->skip && bestCU->mtDepth >= m_skipThresholdE0023FastEnc && !isModeSplit( lastTestmode ) )
  {
    return false;
  }

  bool resetFeature = false;
  // DEV root rules DRR_TT_* (EncModeCtrl.h): at an MTT root in the capped pass the TT rules below that read the BT
  // results are left out, since those results lack the deepest level; all three are true without them
  const bool ttNoResidualRule = !xRootRuleOff( DRR_TT_RULES | DRR_TT_NO_RESIDUAL, partitioner );
  const bool ttDirectionRule  = !xRootRuleOff( DRR_TT_RULES | DRR_TT_DIRECTION,   partitioner );
  const bool ttThresholdRule  = !xRootRuleOff( DRR_TT_RULES | DRR_TT_THRESHOLD,   partitioner );

  switch( split )
  {
    case CU_QUAD_SPLIT:
      {
        if( !cuECtx.qtBeforeBt && bestCU )
        {
          unsigned maxBTD        = xRootRuleLimit( DRR_QT_SKIP, partitioner, cuECtx.mttDepthLimit );
          const CodingUnit *cuBR = bestCS->cus.back();
          unsigned height        = partitioner.currArea().lumaSize().height;

          if(((bestCU->btDepth == 0 && maxBTD >= ((slice.isIntra() && !slice.sps->IBC) ? 3 : 2))
            || (bestCU->btDepth == 1 && cuBR && cuBR->btDepth == 1 && maxBTD >= ((slice.isIntra() && !slice.sps->IBC) ? 4 : 3)))
            && (width <= MAX_TB_SIZEY && height <= MAX_TB_SIZEY)
            && cuECtx.didHorzSplit && cuECtx.didVertSplit )
          {
            return false;
          }
        }
        if( bestCS )
        {
          if( m_pcEncCfg->m_useEarlyCU == 2 && bestCS->cost != MAX_DOUBLE && bestCU && bestCU->skip && cuECtx.nonSkipWasTested && bestCS->cus.size() == 1 )
          {
            return false;
          }

          int stopSplit = (m_pcEncCfg->m_FastInferMerge >> 4) && (bestCS->slice->TLayer > 4);
          int limitBLsize = stopSplit ? 2048 : 1024;
          if((m_pcEncCfg->m_useEarlyCU == 1 || stopSplit) && bestCS->cost != MAX_DOUBLE && bestCU && bestCU->skip && partitioner.currArea().lumaSize().area() < limitBLsize )
          {
            return false;
          }
        }
      }
      break;
    case CU_HORZ_SPLIT:
    case CU_VERT_SPLIT:
      resetFeature = true;
      break;
    case CU_TRIH_SPLIT:
      if( ttNoResidualRule && cuECtx.didHorzSplit && bestCU && bestCU->btDepth == partitioner.currBtDepth && !bestCU->rootCbf )
      {
        return false;
      }

      if( m_pcEncCfg->m_qtbttSpeedUp > 1 && ttDirectionRule )
      {
        //unsigned maxBTD = cs.pcv->getMaxMTTDepth( slice, partitioner.chType );
        if( /*maxBTD == 1 && */cuECtx.didHorzSplit && cuECtx.didVertSplit && cuECtx.bestCostHorzSplit > cuECtx.bestCostVertSplit )
        {
          return false;
        }
      }

      if( !cuECtx.doTriHorzSplit )
      {
        return false;
      }

      if (m_pcEncCfg->m_fastTTSplit && ttThresholdRule && cuECtx.bestCostHorzSplit < MAX_DOUBLE)
      {
        if (cuECtx.bestCostBeforeSplit < MAX_DOUBLE && (cuECtx.bestCostHorzSplit > m_pcEncCfg->m_fastTT_th * cuECtx.bestCostBeforeSplit))
        {
          return false;
        }
        if (cuECtx.bestCostVertSplit < MAX_DOUBLE && (cuECtx.bestCostHorzSplit > m_pcEncCfg->m_fastTT_th * cuECtx.bestCostVertSplit))
        {
          return false;
        }
      }
      break;
    case CU_TRIV_SPLIT:
      if( ttNoResidualRule && cuECtx.didVertSplit && bestCU && bestCU->btDepth == partitioner.currBtDepth && !bestCU->rootCbf )
      {
        return false;
      }

      if( m_pcEncCfg->m_qtbttSpeedUp > 1 && ttDirectionRule )
      {
        //unsigned maxBTD = cs.pcv->getMaxMTTDepth( slice, partitioner.chType );
        if( /*maxBTD == 1 && */cuECtx.didHorzSplit && cuECtx.didVertSplit && cuECtx.bestCostHorzSplit < cuECtx.bestCostVertSplit )
        {
          return false;
        }
      }

      if( !cuECtx.doTriVertSplit )
      {
        return false;
      }

      if (m_pcEncCfg->m_fastTTSplit && ttThresholdRule && cuECtx.bestCostVertSplit < MAX_DOUBLE)
      {
        if (cuECtx.bestCostBeforeSplit < MAX_DOUBLE && (cuECtx.bestCostVertSplit > m_pcEncCfg->m_fastTT_th * cuECtx.bestCostBeforeSplit))
        {
          return false;
        }
        if (cuECtx.bestCostHorzSplit < MAX_DOUBLE && (cuECtx.bestCostVertSplit > m_pcEncCfg->m_fastTT_th * cuECtx.bestCostHorzSplit))
        {
          return false;
        }
      }
      break;
    default:
      THROW( "Only CU split modes are governed by the EncModeCtrl" );
      return false;
      break;
  }

  // DEV root rule DRR_QT_DEEPER (EncModeCtrl.h): where only the capped pass's lowered limit put QT first, a deeper QT
  // split does not stop the root's BT / TT splits; always true without it
  const bool qtDeeperStops = !( cuECtx.qtFirstByLowLimit && xRootRuleOff( DRR_QT_DEEPER, partitioner ) );

  switch( split )
  {
    case CU_HORZ_SPLIT:
    case CU_TRIH_SPLIT:
      if( qtDeeperStops && cuECtx.qtBeforeBt && cuECtx.didQuadSplit )
      {
        if( cuECtx.maxQtSubDepth > partitioner.currQtDepth + 1 )
        {
          if( resetFeature ) cuECtx.didHorzSplit = false;
          return false;
        }
      }
      break;
    case CU_VERT_SPLIT:
    case CU_TRIV_SPLIT:
      if( qtDeeperStops && cuECtx.qtBeforeBt && cuECtx.didQuadSplit )
      {
        if( cuECtx.maxQtSubDepth > partitioner.currQtDepth + 1 )
        {
          if( resetFeature ) cuECtx.didVertSplit = false;
          return false;
        }
      }
      break;
    default:
      break;
  }

  // The DEV rules (EncModeCtrl.h) are asked last, so they only stop BT / TT searches that every rule above lets through
  const bool passedAllRules = m_pcEncCfg->m_qtbttSpeedUp <= 1 || !!cuECtx.doMoreSplits;
  // DEV split stop, VVENC_DEV_SPLIT_STOP_DIR2: no second BT direction where the first one lost by much
  if( split == CU_VERT_SPLIT && passedAllRules && xSecondDirectionStops( cs, partitioner ) )
  {
    m_devCtuCounts[DEV_COUNT_DIR2_NOT_TRIED]++;
    cuECtx.devBtStopped[splitDirection( split )] = true;
    cuECtx.didVertSplit                         = false;
    return false;
  }
  // DEV split stop, VVENC_DEV_SPLIT_STOP_TT: no TT split in a direction whose BT split was stopped
  if( m_devStopTtFollows && ( split == CU_TRIH_SPLIT || split == CU_TRIV_SPLIT ) && passedAllRules && cuECtx.devBtStopped[splitDirection( split )] )
  {
    m_devCtuCounts[DEV_COUNT_TT_NOT_TRIED]++;
    return false;
  }
  // DEV skipped-parent gate: no splits at or below a skipped node. A stopped split counts as not done.
  if( m_devParentGate && passedAllRules && !slice.isIntra() && xParentGateStopsSplit( split ) )
  {
    if( split == CU_HORZ_SPLIT ) cuECtx.didHorzSplit = false;
    if( split == CU_VERT_SPLIT ) cuECtx.didVertSplit = false;
    if( split == CU_QUAD_SPLIT ) cuECtx.didQuadSplit = false;
    return false;
  }
  // DEV MTT entry gate: it decides at the node's first BT / TT search. A stopped BT split counts as not done.
  // The DEV nearby-use gate stops the BT / TT searches of a luma node the same way (BT / TT splits as a tool).
  if( split != CU_QUAD_SPLIT && passedAllRules
      && ( ( m_devMttGate && xMttGateVetoes( split, cs, partitioner ) ) || ( isLuma( partitioner.chType ) && !xNearbyGateAllows( DEV_TOOL_MTT ) ) ) )
  {
    // VVENC_DEV_MTT_GATE_AS_TRIED: where the gate stops one direction only, its BT split counts as searched and lost
    const bool countsAsTried = m_devGateAsTried && ( cuECtx.mttGate == MTT_GATE_HOR || cuECtx.mttGate == MTT_GATE_VER );
    if( split == CU_HORZ_SPLIT && !countsAsTried ) cuECtx.didHorzSplit = false;
    if( split == CU_VERT_SPLIT && !countsAsTried ) cuECtx.didVertSplit = false;
    return false;
  }

  if( split == CU_QUAD_SPLIT )
  {
    cuECtx.didQuadSplit = passedAllRules;
  }
  else if( passedAllRules && xDevGateSkipsDeepSplit( cs, partitioner ) )
  {
    // asked last, so that the DEV gates see exactly the splits the rules above would search
    return false;
  }

  return passedAllRules;
}

// True for the nodes the DEV split gates act on: luma, in an inter picture with a gated MTT depth limit
bool EncModeCtrl::xGatesDeepSplits( const CodingStructure& cs, const Partitioner& partitioner ) const
{
  return isLuma( partitioner.chType ) && !cs.slice->isIntra()
      && partitioner.maxBTD >= m_lowestGatedMttLimit && partitioner.maxBTD <= m_highestGatedMttLimit;
}

// DEV split refinement, asked below a gated MTT root: true for a child of the root that has a level of nodes between
// it and the deepest MTT level (picture limit 3). Such a child had its own splits searched in the capped pass, so
// the refinement pass searches again only the ones chosen by their cost (xStoreSplitsToRefine). With a picture
// limit of 2 the root's children are the parents of the deepest level: all their splits were left out, and in the
// refinement pass the split rules decide on them as in any search.
bool EncModeCtrl::xChoosesSplitsToRefine( const Partitioner& partitioner ) const
{
  const int depth = freeMttDepth( partitioner );
  return isLuma( partitioner.chType ) && depth == 1 && depth + 1 < ( int ) partitioner.maxBTD;
}

// The MTT depth limit that a node's split rules, its reuse of stored results (BestEncInfoCache::isReusingCuValid)
// and its split-cost table (EncCu::xCheckModeSplitInternal) read; called when the node begins. Without the DEV
// split refinement it is the picture's. With it, an MTT root and the nodes of its capped pass are searched as in a
// picture whose limit is one lower, so they must read that limit too: with the picture's, the rules take the splits
// left out for searched and found useless (the root's QT split was skipped after a BT split with unsplit halves).
// What one lower changes depends on the picture's limit: from 3 to 2 the rules that compare depths and the row of
// the split-cost table; from 2 to 1 also QT is searched before BT / TT (QtbttExtraFast above 1), no stored result
// is reused, and the split-cost table is another one (medium's, without the depth indices).
// The nodes of the refinement pass read the picture's limit.
unsigned EncModeCtrl::xRuleMttDepthLimit( const CodingStructure& cs, const Partitioner& partitioner ) const
{
  if( !m_refinesThisCtu || !xGatesDeepSplits( cs, partitioner ) )
  {
    // also the pictures below VVENC_DEV_SPLIT_REFINE_MINTL, and the chroma nodes of a CTU whose search limit the
    // check of the capped pass has raised
    return codedMttDepthLimit( partitioner );
  }
  // outside the two passes (DSP_ALL) the only nodes the gates act on are the MTT roots
  const bool searchedWithLowerLimit = m_deepSplitPass != DSP_REFINE;
  return partitioner.maxBTD - ( searchedWithLowerLimit ? 1 : 0 );
}

// DEV root rules (v9, EncModeCtrl.h): the MTT depth limit that one group of split rules reads at a node whose rules
// read ruleLimit. With the rule's bit, a node searched with the lowered limit of the capped pass reads the picture's
// limit instead; every other node already reads it (the bits are off in the check of the capped pass).
unsigned EncModeCtrl::xRootRuleLimit( const DevRootRule rule, const Partitioner& partitioner, const unsigned ruleLimit ) const
{
  return ( m_devRootRules & rule ) ? std::max( ruleLimit, codedMttDepthLimit( partitioner ) ) : ruleLimit;
}

// DEV root rules: true at an MTT root in its capped pass if one of the DevRootRule bits in rules is set, for the
// rules that are left out there (those that read the capped pass's BT or QT results)
bool EncModeCtrl::xRootRuleOff( const unsigned rules, const Partitioner& partitioner ) const
{
  return ( m_devRootRules & rules ) && m_deepSplitPass == DSP_CAPPED && isLuma( partitioner.chType ) && freeMttDepth( partitioner ) == 0;
}

// DEV check of the capped pass: called for the luma partitioner of a CTU before its search. Lets the search of an
// inter picture whose MTT depth limit is one below the gated limit run with the gated limit, which the capped pass
// then takes back; split flags are priced with codedMttDepthLimit.
void EncModeCtrl::raiseSearchDepthLimitForCheck( Partitioner& partitioner, const Slice& slice )
{
  m_searchDepthRaise  = m_devCappedPassCheck && !slice.isIntra() && partitioner.maxBTD + 1 == m_highestGatedMttLimit ? 1 : 0;
  partitioner.maxBTD += m_searchDepthRaise;
  m_devCheckRaisedCtus += m_searchDepthRaise;
}

// Refinement after QT: true if the root's QT split was searched and the best BT / TT split of the capped pass costs
// more than the margin times as much (also if no BT / TT split was searched)
bool EncModeCtrl::xQtBeatsCappedSplits( const ComprCUCtx& rootCtx ) const
{
  if( !m_devRefineAfterQt || rootCtx.refineQtSplitCost == MAX_DOUBLE )
  {
    return false;
  }
  const double bestCappedCost = *std::min_element( std::begin( rootCtx.mttSplitCost ), std::end( rootCtx.mttSplitCost ) );
  return bestCappedCost > m_devRefineQtMargin * rootCtx.refineQtSplitCost;
}

// The DEV split gates, asked for a BT / TT split that would be searched now. Without the DEV switches the pass is
// always DSP_ALL and the first line returns false.
bool EncModeCtrl::xDevGateSkipsDeepSplit( const CodingStructure& cs, const Partitioner& partitioner )
{
  if( m_deepSplitPass == DSP_ALL && !m_devSplitCover )
  {
    return false;
  }

  const bool makesDeepestLevel = partitioner.currMtDepth + 1 == partitioner.maxBTD + partitioner.currImplicitBtDepth;
  if( !makesDeepestLevel || !xGatesDeepSplits( cs, partitioner ) )
  {
    return false;
  }

  ComprCUCtx& cuECtx = *comprCUCtx;

  if( m_deepSplitPass == DSP_CAPPED )
  {
    // tell the parent that the refinement pass has something new to search below the split that made this node
    ComprCUCtx& parentCtx      = *( m_ComprCUCtxList.end() - 2 );
    parentCtx.refinableSplits |= 1 << partitioner.currPartLevel().split;
    return true;
  }

  if( m_devSplitCover )
  {
    if( cuECtx.deepCoverGate < 0 )
    {
      cuECtx.deepCoverGate = xDeepSplitsCovered( partitioner.currArea().Y(), cuECtx.mttRootId ) ? 1 : 0;
    }
    if( cuECtx.deepCoverGate == 1 )
    {
      return true;
    }
    cuECtx.deepSplitsSearched = true;
  }

  return false;
}

// Cover gate: true if the same MTT root has already searched the deepest level on every part of the area, coming by
// other splits, and a split of the deepest level did not win on all of it
bool EncModeCtrl::xDeepSplitsCovered( const Area& lumaArea, const unsigned rootId ) const
{
  bool seenEverywhere = true;
  bool wonEverywhere  = true;
  forEachMinCu( lumaArea, *CacheBlkInfoCtrl::m_pcv, [&]( const int minCu )
  {
    seenEverywhere &= m_deepSeenByRoot[minCu] == rootId;
    wonEverywhere  &= m_deepWonByRoot [minCu] == rootId;
  } );
  return seenEverywhere && !wonEverywhere;
}

// Cover gate: marks the area of a node that has searched the deepest level, and whether one of its splits won.
// A node the gate has skipped is not marked.
void EncModeCtrl::xStampDeepSplitResult( const Area& lumaArea, const ComprCUCtx& cuECtx )
{
  const bool splitWon = cuECtx.bestCS && isModeSplit( cuECtx.bestMode );
  forEachMinCu( lumaArea, *CacheBlkInfoCtrl::m_pcv, [&]( const int minCu )
  {
    m_deepSeenByRoot[minCu] = cuECtx.mttRootId;
    if( splitWon )
    {
      m_deepWonByRoot[minCu] = cuECtx.mttRootId;
    }
  } );
}

// The BT / TT splits of a node (bits 1 << PartSplit) that the refinement pass searches again: those whose cost in the
// capped pass is within the margin of the node's best cost without QT, and that have a left-out search below them
unsigned EncModeCtrl::xSplitsWithinMargin( const ComprCUCtx& cuECtx, const double margin ) const
{
  const double bestCost = std::min( cuECtx.nonSplitComparedCost, *std::min_element( std::begin( cuECtx.mttSplitCost ), std::end( cuECtx.mttSplitCost ) ) );
  unsigned     splits   = 0;
  for( int split = CU_HORZ_SPLIT; split <= CU_TRIV_SPLIT; split++ )
  {
    const double cost       = cuECtx.mttSplitCost[split - CU_HORZ_SPLIT];
    const bool   refinable  = ( cuECtx.refinableSplits >> split ) & 1;
    if( refinable && cost != MAX_DOUBLE && cost <= margin * bestCost )
    {
      splits |= 1 << split;
    }
  }
  return splits;
}

// End of the capped pass at a child of the MTT root that had splits of its own searched (xChoosesSplitsToRefine):
// notes the splits the child searches again in the refinement pass, in the block info of its area (within one MTT
// root, each of its children has an area of its own), and tells the root that the split that made this child is
// worth searching again.
void EncModeCtrl::xStoreSplitsToRefine( const Partitioner& partitioner )
{
  const ComprCUCtx& cuECtx   = *comprCUCtx;
  const unsigned    toRefine = xSplitsWithinMargin( cuECtx, m_devRefineNodeMargin );
  if( toRefine )
  {
    ComprCUCtx& rootCtx      = *( m_ComprCUCtxList.end() - 2 );
    rootCtx.refinableSplits |= 1 << partitioner.currPartLevel().split;
  }

  // The child's best split is searched again too, with or without a left-out search below it: if a sibling makes
  // the root search this child again, the child must be able to find the result it had in the capped pass.
  const PartSplit bestSplit   = cuECtx.bestCS ? getPartSplit( cuECtx.bestMode ) : CU_DONT_SPLIT;
  const unsigned  searchAgain = toRefine | ( isBtOrTt( bestSplit ) ? 1 << bestSplit : 0 );
  getBlkInfo( partitioner.currArea() ).refineSplits = ( uint8_t ) searchAgain;
}

// DEV split refinement, step 1: called at a node before its BT / TT splits. If the node is an MTT root the refinement
// acts on, the capped pass begins and true is returned; the caller then runs the steps below after the splits.
bool EncModeCtrl::beginCappedSplitPass( const CodingStructure& cs, const Partitioner& partitioner )
{
  if( !m_refinesThisCtu || !xGatesDeepSplits( cs, partitioner ) || freeMttDepth( partitioner ) != 0 )
  {
    return false;
  }
  CHECK( m_deepSplitPass != DSP_ALL, "MTT roots do not nest" );
  m_deepSplitPass = DSP_CAPPED;
  m_devCappedPasses++;
  return true;
}

// DEV split refinement, step 2: ends the capped pass of the MTT root. The root's QT split may be searched between
// this and step 3: its sub-blocks are MTT roots with passes of their own.
void EncModeCtrl::endCappedSplitPass()
{
  m_deepSplitPass = DSP_ALL;
}

// DEV split refinement, step 3: begins the refinement pass of the MTT root and returns the root's BT / TT splits
// (bits 1 << PartSplit) that the caller searches again; endSplitRefinePass follows.
unsigned EncModeCtrl::beginSplitRefinePass( const Partitioner& partitioner )
{
  m_deepSplitPass = DSP_REFINE;
  // From here on nodes below the root are searched a second time: see BestEncInfoCache. That includes the results
  // a QT split of the root has stored since the capped pass, for areas the BT / TT splits reach as well.
  m_revisitFirstStore   = m_storeCount;
  m_revisitSharedLevels = ( int ) partitioner.getPartStack().size();

  const ComprCUCtx& rootCtx = *comprCUCtx;
  if( m_devCappedPassCheck || xQtBeatsCappedSplits( rootCtx ) )
  {
    return 0;   // the check has no deeper level to code; a root that QT wins clearly is not refined
  }
  return xSplitsWithinMargin( rootCtx, m_devRefineRootMargin );
}

void EncModeCtrl::endSplitRefinePass()
{
  m_deepSplitPass     = DSP_ALL;
  m_revisitFirstStore = 0;
}

bool EncModeCtrl::tryMode( const EncTestMode& encTestmode, const CodingStructure &cs, Partitioner& partitioner )
{
  CHECK( isModeSplit( encTestmode ), "wrong method");

  ComprCUCtx& cuECtx = m_ComprCUCtxList.back();

  if( cuECtx.minDepth > partitioner.currQtDepth && partitioner.canSplit( CU_QUAD_SPLIT, cs ) )
  {
    // enforce QT
    return false;
  }

  const Slice&           slice        = *cs.slice;
  const uint32_t         numComp      = getNumberValidComponents( slice.sps->chromaFormatIdc );
  const CodedCUInfo      &relatedCU   = getBlkInfo( partitioner.currArea() );
  const Area             &lumaArea    = partitioner.currArea().Y();
  const CodingStructure* bestCS       = cuECtx.bestCS;
  const EncTestMode      bestMode     = bestCS ?cuECtx.bestMode : EncTestMode();

  if( encTestmode.type == ETM_INTRA )
  {
    // if this is removed, the IntraSearch::xIntraCodingLumaQT needs to be adapted to support Intra TU split
    // also isXXAvailable in IntraPrediction.cpp need to be fixed to check availability within the same CU without isDecomp
    if (m_pcEncCfg->m_FastInferMerge && !slice.isIntra() && !slice.isIRAP() && !(cs.area.lwidth() == 4 && cs.area.lheight() == 4) && !partitioner.isConsIntra())
    {
      if (bestCS && (bestCS->slice->TLayer > (m_pcEncCfg->m_maxTLayer - (m_pcEncCfg->m_FastInferMerge & 7)))
        && (bestCS->bestParent != nullptr) && bestCS->bestParent->cus.size() && (bestCS->bestParent->cus[0]->skip))
      {
        return false;
      }
    }

    if( lumaArea.width > cs.sps->getMaxTbSize() || lumaArea.height > cs.sps->getMaxTbSize() )
    {
      return false;
    }

    if (m_pcEncCfg->m_usePbIntraFast && (!cs.slice->isIntra() || cs.slice->sps->IBC) && cuECtx.interHad == 0 && cuECtx.bestCU && !CU::isIntra(*cuECtx.bestCU))
    {
      return false;
    }

    // DEV skipped-parent gate: no intra search in the split parts of a skipped node that have an inter mode already
    if( m_devParentGate && !slice.isIntra() && !partitioner.isConsIntra() && cuECtx.bestCU && CU::isInter( *cuECtx.bestCU ) && xParentGateStops( DEV_TOOL_INTRA ) )
    {
      return false;
    }

    // DEV nearby-use gate: intra CUs of inter pictures, only where an inter mode was found (elsewhere intra is the only choice)
    // (v4: unless the fallback keeps it, xIntraGateStops)
    if( ( ( m_devToolGate >> DEV_TOOL_INTRA & 1 ) || m_searchCensusFile )
        && !slice.isIntra() && !partitioner.isConsIntra() && bestCS && cuECtx.bestCU && CU::isInter( *cuECtx.bestCU ) && xIntraGateStops( *bestCS, *cuECtx.bestCU, lumaArea ) )
    {
      return false;
    }

    // INTRA MODES
    if (cs.sps->IBC && !cuECtx.bestTU)
    {
     // return true;
    }
    else
    if( partitioner.isConsIntra() && !cuECtx.bestTU )
    {
      //return true;
    }
    else
    if ( lumaArea.width == 4 && lumaArea.height == 4 && !slice.isIntra() && !cuECtx.bestTU )
    {
      //return true;
    }
    else
    if (!(slice.isIRAP() || bestMode.type == ETM_INTRA || !cuECtx.bestTU ||
      ((!m_pcEncCfg->m_bDisableIntraCUsInInterSlices) && (!relatedCU.isInter || !relatedCU.isIBC) && (
                                    ( cuECtx.bestTU->cbf[0] != 0 ) ||
           ( ( numComp > COMP_Cb ) && cuECtx.bestTU->cbf[1] != 0 ) ||
           ( ( numComp > COMP_Cr ) && cuECtx.bestTU->cbf[2] != 0 )  // avoid very complex intra if it is unlikely
         ) ) ) )
    {
      return false;
    }
    else
    if( cuECtx.bestCS && cuECtx.bestCU && cuECtx.interHad )
    {
      // Get SATD threshold from best Inter-CU
      if( !cs.slice->isIRAP() && m_pcEncCfg->m_usePbIntraFast )
      {
        const DFunc dfunc = DF_HAD;
        DistParam distParam = m_pcRdCost->setDistParam( cs.getOrgBuf( COMP_Y ), cuECtx.bestCS->getPredBuf( COMP_Y ), cs.sps->bitDepths[ CH_L ], dfunc );
        cuECtx.interHad = distParam.distFunc( distParam );
      }
    }
    if ((m_pcEncCfg->m_IBCFastMethod > 1) && !cs.slice->isIntra()//IBC_FAST_METHOD_NOINTRA_IBCCBF0
      && (bestMode.type == ETM_IBC || bestMode.type == ETM_IBC_MERGE)
      && (!cuECtx.bestCU->Y().valid() || cuECtx.bestTU->cbf[0] == 0)
      && (!cuECtx.bestCU->Cb().valid() || cuECtx.bestTU->cbf[1] == 0)
      && (!cuECtx.bestCU->Cr().valid() || cuECtx.bestTU->cbf[2] == 0))
    {
      return false;
    }
    if (m_pcEncCfg->m_FastIntraTools)
    {
      if (relatedCU.relatedCuIsValid)
      {
        cuECtx.relatedCuIsValid = relatedCU.relatedCuIsValid;
      }
      if (relatedCU.isIntra)
      {
        cuECtx.isIntra = relatedCU.isIntra;
      }
    }
  }
  else if( isModeInter( encTestmode ) )
  {
    // INTER MODES (ME + MERGE/SKIP)
    CHECK( slice.isIntra(), "Inter-mode should not be in the I-Slice mode list!" );

    // DEV motion search gate (EncModeCtrl.h): no motion search in the split parts of a skipped node
    if( m_devMeGate && ( encTestmode.type == ETM_INTER_ME || encTestmode.type == ETM_INTER_IMV ) && m_ComprCUCtxList.size() > 1
        && ( m_ComprCUCtxList.end() - 2 )->isBestNoSplitSkip && ( m_devMeGate == 1 || partitioner.currMtDepth > 0 ) )
    {
      return false;
    }

    // DEV skipped-parent gate, the same for the motion search (AMVR: devToolAllowed in EncCu::xCompressCU), where
    // the merge search has given the node a mode
    if( m_devParentGate && encTestmode.type == ETM_INTER_ME && cuECtx.bestCS && xParentGateStops( DEV_PARENT_ME ) )
    {
      return false;
    }

    // --- Check if we can quit current mode using SAVE/LOAD coding history

    if( encTestmode.type == ETM_INTER_ME )
    {
      if( encTestmode.opts == ETO_STANDARD )
      {
        if (m_pcEncCfg->m_FastInferMerge)
        {
          if (bestCS && (bestCS->slice->TLayer > (m_pcEncCfg->m_maxTLayer - (m_pcEncCfg->m_FastInferMerge & 7)))
            && (bestCS->bestParent != nullptr) && bestCS->bestParent->cus.size() && (bestCS->bestParent->cus[0]->skip))
          {
            return false;
          }
        }
        if( relatedCU.isSkip || relatedCU.isIntra )
        {
          return false;
        }
      }
    }
  }
  else if (encTestmode.type == ETM_IBC || encTestmode.type == ETM_IBC_MERGE)
  {
    if ((m_pcEncCfg->m_IBCFastMethod > 1) && !(slice.isIRAP() || bestMode.type == ETM_INTRA || !cuECtx.bestTU ||
      ((!m_pcEncCfg->m_bDisableIntraCUsInInterSlices) && (!relatedCU.isInter || !relatedCU.isIBC) && (
        (cuECtx.bestTU->cbf[0] != 0) ||
        ((numComp > COMP_Cb) && cuECtx.bestTU->cbf[1] != 0) ||
        ((numComp > COMP_Cr) && cuECtx.bestTU->cbf[2] != 0)  // avoid very complex intra if it is unlikely
        ))))
    {
      return false;
    }
    if ((m_pcEncCfg->m_IBCFastMethod > 3) &&(lumaArea.width == 4 && lumaArea.height == 4 && !slice.isIntra()))
    {
      return false;
    }
    // IBC MODES
    return slice.sps->IBC && (partitioner.currArea().lumaSize().width < 128 && partitioner.currArea().lumaSize().height < 128);
  }
  else
  {
    THROW("problem");
    return false;
  }

  STAT_COUNT_CU_MODES( partitioner.chType == CH_L, g_cuCounters1D[CU_MODES_TRIED][0][!cs.slice->isIntra() + cs.slice->depth] );
  STAT_COUNT_CU_MODES( partitioner.chType == CH_L && !cs.slice->isIntra(), g_cuCounters2D[CU_MODES_TRIED][Log2( cs.area.lheight() )][Log2( cs.area.lwidth() )] );
  return true;
}


void EncModeCtrl::beforeSplit( Partitioner& partitioner )
{
  ComprCUCtx&           cuECtx  = m_ComprCUCtxList.back();

  if( ! cuECtx.bestCS )
    return;

  CodedCUInfo    &relatedCU   = getBlkInfo( partitioner.currArea() );
  const CodingUnit&  bestCU   = *cuECtx.bestCU;

  cuECtx.bestNsPredMode       = cuECtx.bestMode;
  cuECtx.bestCostBeforeSplit  = cuECtx.bestCS->cost;
  cuECtx.nsFlags              = ( bestCU.skip ? NS_SKIP : 0 ) | ( bestCU.mergeFlag ? NS_MERGE : 0 ) | ( bestCU.rootCbf ? NS_RESIDUAL : 0 )
                              | ( CU::isIntra( bestCU ) ? NS_INTRA : 0 ) | ( CU::isIBC( bestCU ) ? NS_IBC : 0 );   // DEV MTT entry gate
  if( m_searchCensusFile || m_devGateShape > 0.0 || m_devGateShapeDir > 0.0 || xMotionGateIsOn() || ( m_devToolGateSources >> DEV_NEAR_PARENT & 1 ) || xSplitStopReadsError() )
  {
    xNoteNoSplitResult( bestCU, partitioner );
  }
  cuECtx.nonSplitComparedCost = cuECtx.bestCS->cost + cuECtx.bestCS->costDbOffset;   // read by the DEV split refinement only

  setFromCs( *cuECtx.bestCS, cuECtx.bestMode, partitioner );

  if( bestCU.skip )
  {
    cuECtx.doMoreSplits--;
  }

  if( partitioner.modeType == MODE_TYPE_INTRA && partitioner.chType == CH_L )
  {
    return; //not set best coding mode for intra coding pass
  }

  // assume the non-split modes are done and set the marks for the best found mode
  if( CU::isInter( bestCU ) )
  {
    relatedCU.isInter     = true;
    relatedCU.isSkip     |= bestCU.skip;
    relatedCU.isMMVDSkip |= bestCU.mmvdSkip;
    relatedCU.BcwIdx      = bestCU.BcwIdx;
  }
  else if (CU::isIBC(bestCU))
  {
    relatedCU.isIBC = true;
    relatedCU.isSkip |= bestCU.skip;
  }
  else if( CU::isIntra( bestCU ) )
  {
    relatedCU.isIntra     = true;
    if (m_pcEncCfg->m_FastIntraTools)
    {
      if ( cuECtx.intraWasTested && (!relatedCU.relatedCuIsValid || cuECtx.bestCS->cost < relatedCU.bestCost))
      {
        relatedCU.bestCost = cuECtx.bestCS->cost;
        relatedCU.relatedCuIsValid = true;
      }
    }
  }
  cuECtx.isBestNoSplitSkip = bestCU.skip;
}


bool EncModeCtrl::useModeResult( const EncTestMode& encTestmode, CodingStructure*& tempCS, Partitioner& partitioner, const bool useEDO )
{
  ComprCUCtx& cuECtx = m_ComprCUCtxList.back();

  if( m_searchCensusFile || xAmvrGateIsOn() )   // the smart AMVR gate reads the motion search's cost
  {
    xNoteModeResult( encTestmode, *tempCS, useEDO );
  }

  const bool isLumaBtOrTt = isLuma( partitioner.chType ) && isBtOrTt( getPartSplit( encTestmode ) );
  if( m_deepSplitPass == DSP_REFINE && isLumaBtOrTt && freeMttDepth( partitioner ) == 0 )
  {
    // DEV split refinement: a split of the MTT root, searched a second time, only competes with the results so far.
    // What the rules know about the root's splits stays as the capped pass left it; the root's QT decision reads it.
    if( !xKeepIfBest( encTestmode, tempCS, useEDO ) )
    {
      return false;
    }
    // the root's best result now comes from a search with the picture's limit, so the QT decision reads that one
    cuECtx.mttDepthLimit = partitioner.maxBTD;
    return true;
  }
  if( m_deepSplitPass == DSP_CAPPED && isLumaBtOrTt && tempCS->cost != MAX_DOUBLE )
  {
    // the lower of the two where a split is searched once with inter and once with intra sub-blocks
    double& splitCost = cuECtx.mttSplitCost[getPartSplit( encTestmode ) - CU_HORZ_SPLIT];
    splitCost         = std::min( splitCost, tempCS->cost + ( useEDO ? tempCS->costDbOffset : 0 ) );
  }

  if(      encTestmode.type == ETM_SPLIT_BT_H )
  {
    cuECtx.bestCostHorzSplit = tempCS->cost;
  }
  else if( encTestmode.type == ETM_SPLIT_BT_V )
  {
    cuECtx.bestCostVertSplit = tempCS->cost;
  }
  else if( !isModeSplit( encTestmode ) && isModeInter( encTestmode ) && tempCS->cus.size() == 1 )
  {
    cuECtx.nonSkipWasTested |= !tempCS->cus.front()->skip;
  }
  if (m_pcEncCfg->m_AMVRspeed && encTestmode.type == ETM_INTER_ME)
  {
    int imvMode = (encTestmode.opts & ETO_IMV) >> ETO_IMV_SHIFT;

    if (imvMode == 0)
    {
      if (tempCS->cost < cuECtx.bestCostNoImv )
      {
        cuECtx.bestCostNoImv = tempCS->cost;
      }
    }
  }

  if( encTestmode.type == ETM_SPLIT_QT )
  {
    int maxQtD = 0;
    for( const auto& cu : tempCS->cus )
    {
      maxQtD = std::max<int>( maxQtD, cu->qtDepth );
    }
    cuECtx.maxQtSubDepth = maxQtD;
    if( m_devMttGate || m_censusFile )
    {
      xNoteQtSplitResult( *tempCS );
    }

    if( m_devRefineAfterQt && tempCS->cost != MAX_DOUBLE )
    {
      // the lower of the two where the split is searched once with inter and once with intra sub-blocks
      cuECtx.refineQtSplitCost = std::min( cuECtx.refineQtSplitCost, tempCS->cost + ( useEDO ? tempCS->costDbOffset : 0 ) );
    }
  }

  if( m_censusFile && mttSplitIndex( getPartSplit( encTestmode ) ) >= 0 )
  {
    double& splitCost = cuECtx.censusSplitCost[mttSplitIndex( getPartSplit( encTestmode ) )];
    splitCost         = std::min( splitCost, tempCS->cost );   // a split with a mode constraint is searched twice
  }

  int maxMtD = cuECtx.mttDepthLimit + partitioner.currImplicitBtDepth;

  if( encTestmode.type == ETM_SPLIT_BT_H )
  {
    if( tempCS->cus.size() > 2 )
    {
      int h_2   = tempCS->area.blocks[partitioner.chType].height / 2;
      int cu1_h = tempCS->cus.front()->blocks[partitioner.chType].height;
      int cu2_h = tempCS->cus.back() ->blocks[partitioner.chType].height;

      cuECtx.doTriHorzSplit = cu1_h < h_2 || cu2_h < h_2 || partitioner.currMtDepth + 1 == maxMtD;
    }
  }
  else if( encTestmode.type == ETM_SPLIT_BT_V )
  {
    if( tempCS->cus.size() > 2 )
    {
      int w_2   = tempCS->area.blocks[partitioner.chType].width / 2;
      int cu1_w = tempCS->cus.front()->blocks[partitioner.chType].width;
      int cu2_w = tempCS->cus.back() ->blocks[partitioner.chType].width;

      cuECtx.doTriVertSplit = cu1_w < w_2 || cu2_w < w_2 || partitioner.currMtDepth + 1 == maxMtD;
    }
  }

  // DEV root rule DRR_MORE_SPLITS (EncModeCtrl.h): the capped BT results of an MTT root do not count here
  if( ( encTestmode.type == ETM_SPLIT_BT_V || encTestmode.type == ETM_SPLIT_BT_H || encTestmode.type == ETM_SPLIT_QT ) && !xRootRuleOff( DRR_MORE_SPLITS, partitioner ) )
  {
    bool isAllSkip = true;

    for( const auto& cu : tempCS->cus )
    {
      isAllSkip &= cu->skip;
    }

    if( isAllSkip ) cuECtx.doMoreSplits--;
    if( isAllSkip && encTestmode.type == ETM_SPLIT_QT )
                    cuECtx.doMoreSplits--;
  }

  return xKeepIfBest( encTestmode, tempCS, useEDO );
}

// Makes tempCS the best result of the current CU if it is better than the best so far; returns whether it did
bool EncModeCtrl::xKeepIfBest( const EncTestMode& encTestmode, CodingStructure* tempCS, const bool useEDO )
{
  ComprCUCtx& cuECtx = m_ComprCUCtxList.back();

  // for now just a simple decision based on RD-cost or choose tempCS if bestCS is not yet coded
  if( tempCS->cost != MAX_DOUBLE && ( !cuECtx.bestCS || ( ( tempCS->cost + ( useEDO ? tempCS->costDbOffset : 0 ) ) < ( cuECtx.bestCS->cost + ( useEDO ? cuECtx.bestCS->costDbOffset : 0 ) ) ) ) )
  {
    cuECtx.bestCS   = tempCS;
    cuECtx.bestCU   = tempCS->cus[0];
    cuECtx.bestTU   = cuECtx.bestCU->firstTU;
    cuECtx.bestMode = encTestmode;

    return true;
  }
  else
  {
    return false;
  }
}

} // namespace vvenc

//! \}

