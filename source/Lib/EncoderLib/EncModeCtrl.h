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
/** \file     EncModeCtrl.h
    \brief    Encoder controller for trying out specific modes
*/

#pragma once

#include "InterSearch.h"
#include "CommonLib/CommonDef.h"
#include "CommonLib/CodingStructure.h"

#include <algorithm>
#include <cstdio>
#include <memory>
#include <typeinfo>
#include <vector>

//! \ingroup EncoderLib
//! \{

namespace vvenc {

//////////////////////////////////////////////////////////////////////////
// Encoder modes to try out
//////////////////////////////////////////////////////////////////////////


enum EncTestModeType
{
  ETM_MERGE_SKIP,
  ETM_INTER_ME,
  ETM_INTER_IMV,
  ETM_INTRA,
  ETM_SPLIT_QT,
  ETM_SPLIT_BT_H,
  ETM_SPLIT_BT_V,
  ETM_SPLIT_TT_H,
  ETM_SPLIT_TT_V,
  ETM_RECO_CACHED,
  ETM_IBC,
  ETM_IBC_MERGE,
  ETM_INVALID
};

enum EncTestModeOpts
{
  ETO_STANDARD    =  0,                   // empty      (standard option)
  ETO_FORCE_MERGE =  1<<0,                // bit   0    (indicates forced merge)
  ETO_IMV_SHIFT   =  1,                   // bits  1-3  (imv parameter starts at bit 1)
  ETO_IMV         =  7<<ETO_IMV_SHIFT,    // bits  1-3  (imv parameter uses 3 bits)
  ETO_DUMMY       =  1<<5,                // bit   5    (dummy)
  ETO_INVALID     = 0xffffffff            // bits 0-31  (invalid option)
};

static inline void getAreaIdxNew(const Area& area, const PreCalcValues &pcv, unsigned& idx1, unsigned& idx2, unsigned& idx3, unsigned& idx4)
{
  idx1 = Log2( area.width  )-2;
  idx2 = Log2( area.height )-2;
  idx3 = (area.x & pcv.maxCUSizeMask) >> MIN_CU_LOG2;
  idx4 = (area.y & pcv.maxCUSizeMask) >> MIN_CU_LOG2;
}

struct EncTestMode
{
  EncTestMode()
    : type( ETM_INVALID ), opts( ETO_INVALID  ), qp( -1  ), lossless( false ) {}
  EncTestMode( EncTestModeType _type )
    : type( _type       ), opts( ETO_STANDARD ), qp( -1  ), lossless( false ) {}
  EncTestMode( EncTestModeType _type, int _qp, bool _lossless )
    : type( _type       ), opts( ETO_STANDARD ), qp( _qp ), lossless( _lossless ) {}
  EncTestMode( EncTestModeType _type, EncTestModeOpts _opts, int _qp, bool _lossless )
    : type( _type       ), opts( _opts        ), qp( _qp ), lossless( _lossless ) {}

  EncTestModeType type;
  EncTestModeOpts opts;
  int             qp;
  bool            lossless;
  double          maxCostAllowed;
};


inline bool isModeSplit( const EncTestMode& encTestmode )
{
  switch( encTestmode.type )
  {
  case ETM_SPLIT_QT     :
  case ETM_SPLIT_BT_H   :
  case ETM_SPLIT_BT_V   :
  case ETM_SPLIT_TT_H   :
  case ETM_SPLIT_TT_V   :
    return true;
  default:
    return false;
  }
}

inline bool isModeNoSplit( const EncTestMode& encTestmode )
{
  return !isModeSplit( encTestmode );
}

inline bool isModeInter( const EncTestMode& encTestmode ) // perhaps remove
{
  return (   encTestmode.type == ETM_INTER_ME
          || encTestmode.type == ETM_INTER_IMV
          || encTestmode.type == ETM_MERGE_SKIP
         );
}

inline PartSplit getPartSplit( const EncTestMode& encTestmode )
{
  switch( encTestmode.type )
  {
  case ETM_SPLIT_QT     : return CU_QUAD_SPLIT;
  case ETM_SPLIT_BT_H   : return CU_HORZ_SPLIT;
  case ETM_SPLIT_BT_V   : return CU_VERT_SPLIT;
  case ETM_SPLIT_TT_H   : return CU_TRIH_SPLIT;
  case ETM_SPLIT_TT_V   : return CU_TRIV_SPLIT;
  default:                return CU_DONT_SPLIT;
  }
}

//////////////////////////////////////////////////////////////////////////
// EncModeCtrl controls if specific modes should be tested
//////////////////////////////////////////////////////////////////////////

// DEV MTT entry gate (see EncModeCtrl): what the best mode without a split looked like (ComprCUCtx::nsFlags)
enum NsFlag : uint8_t { NS_SKIP = 1, NS_MERGE = 2, NS_RESIDUAL = 4, NS_INTRA = 8, NS_IBC = 16 };

// DEV MTT entry gate: the BT / TT directions a node may search (ComprCUCtx::mttGate)
enum MttGate : uint8_t { MTT_GATE_NONE = 0, MTT_GATE_HOR = 1, MTT_GATE_VER = 2, MTT_GATE_BOTH = 3, MTT_GATE_UNDECIDED = 4 };

// DEV MTT entry gate: the four pairs of neighbouring leaves of a QT split (ComprCUCtx::qtPairMvGap)
enum QtPair { QT_PAIR_TOP, QT_PAIR_BOTTOM, QT_PAIR_LEFT, QT_PAIR_RIGHT, NUM_QT_PAIRS };

static constexpr int NUM_MTT_SPLITS = 4;   // BT horizontal, BT vertical, TT horizontal, TT vertical

// DEV search census (see EncModeCtrl): what a node's time is kept for. First its searches, whose best cost is kept
// too, then parts of those searches that are timed on their own (their time is also inside the search's time).
enum DevTimer
{
  DEV_SEARCH_MERGE, DEV_SEARCH_INTER_ME, DEV_SEARCH_IMV, DEV_SEARCH_IBC, DEV_SEARCH_INTRA,
  DEV_SEARCH_QT, DEV_SEARCH_BT_H, DEV_SEARCH_BT_V, DEV_SEARCH_TT_H, DEV_SEARCH_TT_V,
  NUM_DEV_SEARCHES,
  DEV_PART_AFFINE_ME = NUM_DEV_SEARCHES,   // affine motion estimation
  DEV_PART_SBT,                            // sub-block transform: its pre-analysis and its RD tests
  DEV_PART_GEO,                            // GEO candidate list and SATD pass (not the RD tests of GEO candidates)
  DEV_PART_CIIP,                           // CIIP SATD pass (not the RD tests of CIIP candidates)
  DEV_PART_SMVD,                           // symmetric MVD search
  DEV_PART_BCW,                            // the motion searches and RD tests with unequal bi-prediction weights
  NUM_DEV_TIMERS,                          // the timers of the S line of the census
  // v5: written in the K line, so that the S line stays as the readers of v2 to v4 know it
  DEV_PART_MMVD = NUM_DEV_TIMERS,          // MMVD SATD pass (not the RD tests of MMVD candidates)
  DEV_PART_AFFINE_MERGE,                   // affine / sub-block merge candidate list and SATD pass (not their RD tests)
  // v7: written in the A line. The AMVR search (DEV_SEARCH_IMV) by MVD precision, in the order of the IMV modes
  DEV_PART_IMV_FPEL,                       // whole-sample MVDs
  DEV_PART_IMV_4PEL,                       // 4-sample MVDs
  DEV_PART_IMV_HPEL,                       // half-sample MVDs (with the smoothing interpolation filter)
  NUM_ALL_DEV_TIMERS
};
static constexpr int NUM_IMV_PRECISIONS = 3;

// DEV search census and smart AMVR gate (v7): the MVDs of the result of a motion search
struct DevMvdStats
{
  int  largest   = -1;      // largest MVD component in quarter samples; -1: there is no result
  int  bits      = 0;       // bins of the MVDs as coded with quarter-sample precision (each bin counted as one bit)
  int  bitsWhole = 0;       // the same with every MVD rounded to whole samples and coded with that precision
  bool biPred    = false;
  bool affine    = false;
  bool smvd      = false;
};

// DEV search census and nearby-use gate (see EncModeCtrl): the coding tools, as bit positions of a tool mask. The gate
// can stop the search of the first NUM_GATED_DEV_TOOLS; the others are only counted.
enum DevTool
{
  DEV_TOOL_IMV,            // AMVR: a motion search with integer, 4-sample or half-sample MVDs
  DEV_TOOL_SBT,
  DEV_TOOL_INTRA,          // intra CUs of inter pictures
  DEV_TOOL_AFFINE,         // affine motion estimation
  DEV_TOOL_GEO,
  DEV_TOOL_CIIP,
  DEV_TOOL_BCW,            // unequal bi-prediction weights found by motion estimation
  DEV_TOOL_SMVD,
  DEV_TOOL_MTT,            // BT / TT splits: the tool of every CU below one
  NUM_GATED_DEV_TOOLS,
  DEV_TOOL_AFFINE_MERGE = NUM_GATED_DEV_TOOLS,
  DEV_TOOL_MMVD,
  DEV_TOOL_MERGE,          // any merge mode
  DEV_TOOL_ISP,
  DEV_TOOL_MIP,
  NUM_DEV_TOOLS
};

// DEV skipped-parent gate (see EncModeCtrl): the searches it can stop, as bit positions of VVENC_DEV_PARENT_GATE. The
// first are the DevTool bits of the searches (AMVR 0x1, SBT 0x2, intra 0x4, affine ME 0x8, GEO 0x10, CIIP 0x20,
// BCW 0x40, SMVD 0x80, BT / TT splits 0x100, affine merge 0x200, MMVD 0x400), then two that are not tools:
enum DevParentSearch
{
  DEV_PARENT_ME = NUM_DEV_TOOLS,   // 0x4000: the whole motion search without AMVR (with its affine, SMVD and BCW searches)
  DEV_PARENT_QT,                   // 0x8000: QT splits
  NUM_DEV_PARENT_SEARCHES
};

// DEV nearby-use gate: where the gate looks for CUs that use a tool (bit positions of VVENC_DEV_TOOL_GATE_SRC)
enum DevNearby
{
  DEV_NEAR_CUS,            // the CUs left of and above the node
  DEV_NEAR_CTUS,           // the CTUs left, above and above left of the node's CTU
  DEV_NEAR_REFS,           // the CTU at the same place in the first reference picture of each list
  DEV_NEAR_PARENT,         // the best mode without a split of the node this one is a split part of (x265's --limit-modes idea)
  NUM_DEV_NEARBY
};

static constexpr int NUM_ERROR_STRIPS = 4;   // a block's prediction error is kept for four row and four column strips

// DEV motion estimate of the splits (EncModeCtrl::xEstimateSplitMotion): the partitions it is made for
enum DevMotionSplit { DEV_MOTION_NONE, DEV_MOTION_BT_H, DEV_MOTION_BT_V, DEV_MOTION_TT_H, DEV_MOTION_TT_V, DEV_MOTION_QT, NUM_DEV_MOTION_SPLITS };

// DEV search census: size and depths of a neighbouring CU, -1 where there is none
struct DevNeighbourCu
{
  int width = -1, height = -1, qtDepth = -1, mtDepth = -1;
};

// DEV search census and nearby-use gate: what is kept per node. Nothing reads it unless one of them is on.
struct DevNodeStats
{
  DevNodeStats()
  {
    std::fill_n( predErrorRows, NUM_ERROR_STRIPS, -1 );
    std::fill_n( predErrorCols, NUM_ERROR_STRIPS, -1 );
    std::fill_n( time,          NUM_ALL_DEV_TIMERS, 0 );
    std::fill_n( searchCost,    NUM_DEV_SEARCHES, MAX_DOUBLE );
    std::fill_n( costWith,      NUM_DEV_TOOLS,    MAX_DOUBLE );
    std::fill_n( costWithout,   NUM_DEV_TOOLS,    MAX_DOUBLE );
    std::fill_n( motionSad,     NUM_DEV_MOTION_SPLITS, -1 );
  }

  int            id                 = -1;   // number of the node in its encoder thread, in the order the nodes start
  int            parentId           = -1;   // id of the node this one is a split part of, -1 for a CTU
  uint32_t       toolsOfNearCus     = 0;    // DevTool bits of the CUs left of and above the node
  uint32_t       toolsStopped       = 0;    // DevTool bits the nearby-use gate stopped at this node
  uint32_t       toolsOfBestNoSplit = 0;    // DevTool bits of the best mode without a split
  uint32_t       toolsOfParent      = ~0u;  // toolsOfBestNoSplit of the node this one is a part of; all bits where that node has no such mode
  // DEV motion estimate: SAD of the node with each part of a split predicted by the best of the node's candidate
  // motions (a whole-sample MV into one reference picture); -1 where it was not made. motionCands: the candidates.
  int            motionSad[NUM_DEV_MOTION_SPLITS];
  int            motionCands        = 0;
  DevNeighbourCu left, above;
  // SAD between the source and the best inter prediction without a split, in four strips from top to bottom and four
  // from left to right; -1 where that mode is not inter or was taken from the cache (its prediction is not kept)
  int            predErrorRows[NUM_ERROR_STRIPS];
  int            predErrorCols[NUM_ERROR_STRIPS];
  int64_t        time[NUM_ALL_DEV_TIMERS];      // ns; a split's time includes the searches of the nodes below it
  double         searchCost[NUM_DEV_SEARCHES];  // best cost each search reached, MAX_DOUBLE where it did not run
  // best cost of the RD-tested modes without a split that use / do not use each tool
  double         costWith[NUM_DEV_TOOLS];
  double         costWithout[NUM_DEV_TOOLS];
  // v4, intra gate with fallback: the best inter mode when the intra search of an inter picture was asked for (its
  // cost and NsFlag bits; MAX_DOUBLE where intra was not asked for after an inter mode), and what the gate said there
  double         interCostBeforeIntra  = MAX_DOUBLE;
  uint8_t        interFlagsBeforeIntra = 0;
  uint8_t        intraGateSaid         = 0;   // DevIntraGate
  uint32_t       parentGateStopped     = 0;   // v5: DevParentSearch bits the skipped-parent gate stopped at this node
  // v7, smart AMVR: the result of the motion search with quarter-sample MVDs and equal weights (its MVDs, RD cost
  // and bits), the node's best mode when the AMVR search was asked for (cost and NsFlag bits; MAX_DOUBLE where it
  // was not asked for), and the best cost the AMVR search reached with each MVD precision
  DevMvdStats    meMvd;
  double         meCost                = MAX_DOUBLE;
  uint64_t       meFracBits            = 0;
  double         costBeforeImv         = MAX_DOUBLE;
  uint8_t        flagsBeforeImv        = 0;
  double         imvCost[NUM_IMV_PRECISIONS] = { MAX_DOUBLE, MAX_DOUBLE, MAX_DOUBLE };
  bool           amvrGateStopped       = false;   // the smart AMVR gate stopped the AMVR search of this node
  double         lastNotedCost         = MAX_DOUBLE;   // cost of the last RD-tested mode, as compared (xNoteModeResult)
};

// DEV intra gate with fallback: what the nearby-use gate said about the intra search of a node
enum DevIntraGate : uint8_t { DEV_INTRA_NOT_ASKED, DEV_INTRA_USED_NEARBY, DEV_INTRA_STOPPED, DEV_INTRA_KEPT_BY_FALLBACK };

// DEV split stop and split census: the kinds of splits (bit positions of VVENC_DEV_SPLIT_STOP_SPLITS)
enum DevSplitKind { DEV_SPLIT_KIND_BT, DEV_SPLIT_KIND_TT, DEV_SPLIT_KIND_QT };

// DEV split stop: how the cost of the parts of a split that are not searched yet is estimated
// (VVENC_DEV_SPLIT_STOP_SHARE): a share of a cost of the whole block, or costs of parts that are known
enum DevRestShare
{
  DEV_REST_BY_AREA,       // the share of the block's luma samples in those parts
  DEV_REST_BY_ERROR,      // the share of the block's prediction error (best inter mode without a split) in those parts
  DEV_REST_BY_BT_PARTS    // for a TT split after the BT split of its direction: the costs of that split's halves
};

// DEV split stop and intra gate with fallback: what is counted per CTU (the G line of the search census)
enum DevCtuCount
{
  DEV_COUNT_SPLITS_STOPPED,    // splits the split stop ended between two parts
  DEV_COUNT_TT_NOT_TRIED,      // TT splits not searched after a stopped BT split
  DEV_COUNT_INTRA_STOPPED,     // intra searches the nearby-use gate stopped
  DEV_COUNT_INTRA_KEPT,        // intra searches its fallback kept
  DEV_COUNT_DIR2_NOT_TRIED,    // second BT directions not searched
  NUM_DEV_CTU_COUNTS
};

// DEV smart AMVR gate and smart SBT gate (v7): what they stopped, counted per CTU (the J line of the search census)
enum DevV7Stop
{
  DEV_V7_AMVR_STOPPED,         // AMVR searches not made
  DEV_V7_4PEL_STOPPED,         // AMVR searches made without 4-sample MVDs
  DEV_V7_SBT_TESTS_STOPPED,    // RD tests of SBT modes not made
  NUM_DEV_V7_STOPS
};

static constexpr int MAX_SPLIT_PARTS = 4;

// DEV split stop and split census (see EncModeCtrl): one search of a split of a node, part by part
struct DevSplitTry
{
  bool        active       = false;   // the stop rule or the census follows this search
  bool        stoppable    = false;   // the stop rule may act here (EncModeCtrl::devStartSplit)
  bool        ruleOn       = false;   // ... and it is switched on for this kind of split and this MTT depth
  EncTestMode testMode;
  Area        nodeArea;               // luma area of the node that is split
  double      bestCost     = 0.0;     // best cost of the node when the search of the split starts
  int64_t     splitBits    = 0;       // fractional bits of the split flags
  int64_t     startTime    = 0;       // census clock
  int         firstNodeId  = 0;       // the nodes below part k have the ids [nodesAfterPart[k - 1], nodesAfterPart[k])
  int         partsDone    = 0;
  int         stoppedAfter = 0;       // number of parts searched when the rule stopped the search, 0 = not stopped
  int         lumaDone     = 0;       // luma samples of the parts searched
  int         quartersDone = 0;       // BT / TT: how far the parts searched reach across the split, in quarters of the block
  double      stopEstimate = 0.0;     // what the rule compared with T x the best cost when it stopped
  double      costAfterPart[MAX_SPLIT_PARTS];    // RD cost of the parts searched so far, with the split flags
  int64_t     timeAfterPart[MAX_SPLIT_PARTS];    // ns since startTime
  int         nodesAfterPart[MAX_SPLIT_PARTS];
};

struct ComprCUCtx
{
  ComprCUCtx()
  {
  }

  ComprCUCtx( const CodingStructure& cs, const uint32_t _minDepth, const uint32_t _maxDepth )
    : minDepth      ( _minDepth  )
    , maxDepth      ( _maxDepth  )
    , bestCS        ( nullptr    )
    , bestCU        ( nullptr    )
    , bestTU        ( nullptr    )
    , bestMode      ()
    , bestCostBeforeSplit   (MAX_DOUBLE)
    , bestCostVertSplit     (MAX_DOUBLE)
    , bestCostHorzSplit     (MAX_DOUBLE)
    , bestCostNoImv         (MAX_DOUBLE *.5)
    , grad_horVal           (0)
    , grad_verVal           (0)
    , grad_dupVal           (0)
    , grad_dowVal           (0)
    , interHad              (MAX_DISTORTION)
    , maxQtSubDepth         (0)
    , isReusingCu           (false)
    , qtBeforeBt            (false)
    , doTriHorzSplit        (false)
    , doTriVertSplit        (false)
    , didQuadSplit          (false)
    , didHorzSplit          (false)
    , didVertSplit          (false)
    , doHorChromaSplit      (false)
    , doVerChromaSplit      (false)
    , doQtChromaSplit       (false)
    , isBestNoSplitSkip     (false)
    , intraWasTested        (false)
    , relatedCuIsValid      (false)
    , isIntra               (false)
    , nonSkipWasTested      (false)
    , bestNsPredMode        (EncTestMode())
    , nonSplitComparedCost  (MAX_DOUBLE)
    , mttSplitCost          { MAX_DOUBLE, MAX_DOUBLE, MAX_DOUBLE, MAX_DOUBLE }
    , refinableSplits       (0)
    , mttRootId             (0)
    , deepCoverGate         (-1)
    , deepSplitsSearched    (false)
    , mttDepthLimit         (0)
    , refineQtSplitCost     (MAX_DOUBLE)
  {
  }

  unsigned          minDepth;
  unsigned          maxDepth;
  CodingStructure*  bestCS;
  CodingUnit*       bestCU;
  TransformUnit*    bestTU;
  EncTestMode       bestMode;
  double            bestCostBeforeSplit;
  double            bestCostVertSplit;
  double            bestCostHorzSplit;
  double            bestCostNoImv;
  double            grad_horVal;
  double            grad_verVal;
  double            grad_dupVal;
  double            grad_dowVal;
  Distortion        interHad;
  int               maxQtSubDepth;
  bool              isReusingCu;
  bool              qtBeforeBt;
  bool              doTriHorzSplit;
  bool              doTriVertSplit;
  int               doMoreSplits;
  bool              didQuadSplit;
  bool              didHorzSplit;
  bool              didVertSplit;
  bool              doHorChromaSplit;
  bool              doVerChromaSplit;
  bool              doQtChromaSplit;
  bool              isBestNoSplitSkip;
  bool              intraWasTested;
  bool              relatedCuIsValid;
  bool              isIntra;
  bool              nonSkipWasTested;
  EncTestMode       bestNsPredMode;

  // DEV MTT entry gate and census (see EncModeCtrl): nothing reads these without VVENC_DEV_MTT_GATE / _CENSUS
  uint8_t           nsFlags                         = 0;                    // NsFlag bits of the best mode without a split
  uint8_t           mttGate                         = MTT_GATE_UNDECIDED;   // MttGate, decided at the node's first BT / TT test
  bool              qtLeavesKnown                   = false;                // the QT split was searched and gave four unsplit CUs
  double            qtSplitCost                     = MAX_DOUBLE;           // cost of the node's QT split, once searched
  int               qtPairMvGap[NUM_QT_PAIRS]       = { -1, -1, -1, -1 };   // largest MV difference of a pair of QT leaves in MV units; -1: different kind of motion
  double            censusSplitCost[NUM_MTT_SPLITS] = { MAX_DOUBLE, MAX_DOUBLE, MAX_DOUBLE, MAX_DOUBLE };
  int64_t           censusSplitTime[NUM_MTT_SPLITS] = { 0, 0, 0, 0 };       // ns inside each BT / TT search
  int64_t           censusNodeStart                 = 0;                    // clock at the start of the node
  // DEV split stop: the BT split of each direction (0 horizontal, 1 vertical): the costs of its halves once it was
  // searched to the end, and whether the rule stopped it
  double            devBtHalfCost[2][2]             = { { MAX_DOUBLE, MAX_DOUBLE }, { MAX_DOUBLE, MAX_DOUBLE } };
  bool              devBtStopped[2]                 = { false, false };
  DevNodeStats      dev;                                                    // DEV search census and nearby-use gate

  // DEV split gates (smart-slow prototype, EncModeCtrl::xDevGateSkipsDeepSplit). Split sets are bit masks of 1 << PartSplit.
  double            nonSplitComparedCost;   // cost of the best non-split result plus its deblocking offset
  double            mttSplitCost[4];        // capped pass: cost of BT-H, BT-V, TT-H, TT-V as compared, MAX_DOUBLE if not searched
  unsigned          refinableSplits;        // capped pass: splits below which a search of the deepest MTT level was left out
  unsigned          mttRootId;              // cover gate: number of the MTT root this node belongs to
  int8_t            deepCoverGate;          // cover gate at the deepest level's parent: -1 not decided, 0 search, 1 skip
  bool              deepSplitsSearched;     // cover gate: a split of the deepest MTT level was searched at this node
  unsigned          mttDepthLimit;          // the MTT depth limit the split rules of this node read (EncModeCtrl::xRuleMttDepthLimit); without the DEV split refinement the picture's
  double            refineQtSplitCost;            // refinement after QT: cost of the QT split as compared, MAX_DOUBLE if not searched
  bool              qtFirstByLowLimit = false;    // v9 root rules: QT comes before BT / TT only because the node reads the capped pass's limit
};

//////////////////////////////////////////////////////////////////////////
// some utility interfaces that expose some functionality that can be used without concerning about which particular controller is used
//////////////////////////////////////////////////////////////////////////

static const int MAX_STORED_CU_INFO_REFS = 4;

struct CodedCUInfo
{
  bool      relatedCuIsValid;
  bool      isInter;
  bool      isIntra;
  bool      isSkip;
  bool      isMMVDSkip;
  bool      isIBC;
  uint8_t   BcwIdx;
  uint8_t   numPuInfoStored;
  int       poc, ctuRsAddr;
  bool      validMv[NUM_REF_PIC_LIST_01][MAX_STORED_CU_INFO_REFS];
  Mv        saveMv [NUM_REF_PIC_LIST_01][MAX_STORED_CU_INFO_REFS];
  uint32_t  puSse  [SBT_NUM_SL];
  uint8_t   puSbt  [SBT_NUM_SL];
  double    bestCost;
  uint8_t   refineSplits;   // DEV split refinement: the BT / TT splits (bits 1 << PartSplit) this block searches again as a child of its MTT root (EncModeCtrl::xChoosesSplitsToRefine)

  bool getMv  ( const RefPicList refPicList, const int iRefIdx,       Mv& rMv ) const;
  void setMv  ( const RefPicList refPicList, const int iRefIdx, const Mv& rMv );
};

class CacheBlkInfoCtrl
{
protected:
  // x in CTU, y in CTU, width, height
  CodedCUInfo*         m_codedCUInfo[6][6][MAX_CU_SIZE >> MIN_CU_LOG2][MAX_CU_SIZE >> MIN_CU_LOG2];
  CodedCUInfo*         m_codedCUInfoBuf;
  const PreCalcValues* m_pcv;

protected:

  void create   ( int ctuSize );
  void destroy  ();
  void init     ( const Slice &slice );

public:
  CacheBlkInfoCtrl() : m_codedCUInfoBuf( nullptr ) {}
  ~CacheBlkInfoCtrl () {}

  CodedCUInfo& getBlkInfo   ( const UnitArea& area );
  void         initBlk      ( const UnitArea& area, int poc );

  uint8_t      findBestSbt  ( const UnitArea& area, const uint32_t curPuSse );
  bool         saveBestSbt  ( const UnitArea& area, const uint32_t curPuSse, const uint8_t curPuSbt );
};

struct BestEncodingInfo
{ 
  CodingUnit      cu;
  TransformUnit   tu;
  EncTestMode     testMode;
  int             poc;
  Distortion      dist;
  double          costEDO;
  uint32_t        storeNumber;   // position of this result among the results stored for the CTU (BestEncInfoCache::m_storeCount)
};

// Hands out runs of T from chunks that are kept for reuse. Runs never move, so pointers into them
// stay valid until reset() makes all chunks available again.
template<typename T, size_t ChunkSize>
class ReusableChunkArena
{
public:
  T* take( const size_t n )
  {
    CHECK( n > ChunkSize, "Run larger than an arena chunk" );
    if( m_current < 0 || m_used + n > ChunkSize )
    {
      m_current++;
      if( m_current == ( int ) m_chunks.size() )
      {
        m_chunks.emplace_back( new T[ChunkSize] );
      }
      m_used = 0;
    }
    T* run  = m_chunks[m_current].get() + m_used;
    m_used += n;
    return run;
  }

  void reset()   { m_current = -1; m_used = 0; }
  void release() { m_chunks.clear(); reset(); }

private:
  std::vector<std::unique_ptr<T[]>> m_chunks;
  int    m_current = -1;
  size_t m_used    = 0;
};

class BestEncInfoCache
{
private:
  const PreCalcValues* m_pcv;
  // The stored result of every CU size and position inside the CTU being encoded, or null. A result
  // only matches a CU of the same picture and position, so none outlives the CTU it was stored for,
  // and init() empties the slots of the previous CTU and reuses their storage.
  BestEncodingInfo*    m_bestEncInfo[6][6][MAX_CU_SIZE >> MIN_CU_LOG2][MAX_CU_SIZE >> MIN_CU_LOG2];
  std::vector<BestEncodingInfo**>           m_filledSlots;
  ReusableChunkArena<BestEncodingInfo, 256> m_entryArena;
  ReusableChunkArena<TCoeffSig, 131072>     m_coeffArena;
  ReusableChunkArena<Mv, 4096>              m_dmvrMvArena;
  ChromaFormat         m_chFmt;
  CodingStructure      m_dummyCS;
  XUCache              m_dummyCache;

  BestEncodingInfo*    xNewEntry( const int wIdx, const int hIdx );
  int                  xSharedLevels( const BestEncodingInfo& encInfo ) const;

protected:
  bool                 m_reuseCuResults;
  // A stored result is reused only if the CU still has the neighbours it was found with. The check (isTheSameNbHood)
  // relies on every node being searched once. The DEV split refinement searches the splits of an MTT root a second
  // time, so while that runs (m_revisitFirstStore > 0) the results stored before it are trusted only for what lies
  // outside the root: the first m_revisitSharedLevels levels of the split path.
  uint32_t             m_storeCount;            // results stored for the current CTU
  uint32_t             m_revisitFirstStore;     // m_storeCount when the second search began, 0 when none is running
  int                  m_revisitSharedLevels;
  void create   ( const bool reuseCuResults, const ChromaFormat chFmt, int ctuSize );
  void destroy  ();

public:
  BestEncInfoCache() : m_pcv( nullptr ), m_chFmt( CHROMA_420 ), m_dummyCS( m_dummyCache, nullptr ), m_storeCount( 0 ), m_revisitFirstStore( 0 ), m_revisitSharedLevels( 0 ) {}
  ~BestEncInfoCache() {}

  void init             ( const Slice &slice );
  bool setCsFrom        (       CodingStructure& cs,       EncTestMode& testMode, const Partitioner& partitioner ) const;
  bool setFromCs        ( const CodingStructure& cs, const EncTestMode& testMode, const Partitioner& partitioner );
  // mttDepthLimit: the limit the CU's split rules read (ComprCUCtx::mttDepthLimit); partitioner.maxBTD without the
  // DEV split refinement
  bool isReusingCuValid ( const CodingStructure &cs, const Partitioner &partitioner, int qp, unsigned mttDepthLimit );
};

//////////////////////////////////////////////////////////////////////////
// EncModeCtrl - allows and controls modes introduced by QTBT (inkl. multi-type-tree)
//                    - only 2Nx2N, no RQT, additional binary/triary CU splits
//////////////////////////////////////////////////////////////////////////

class EncModeCtrl: public CacheBlkInfoCtrl, public BestEncInfoCache
{
protected:

  const VVEncCfg*       m_pcEncCfg;
        RdCost*         m_pcRdCost;
  static_vector<ComprCUCtx, ( MAX_CU_DEPTH << 2 )> m_ComprCUCtxList;
  unsigned              m_skipThresholdE0023FastEnc;

  // DEV MTT entry gate (VVENC_DEV_MTT_GATE=1; a prototype for a smart medium preset). It acts in inter pictures whose
  // MTT depth limit is 1, on the luma nodes of the QT tree away from the picture border: there the BT / TT splits
  // make leaf CUs, so each one costs about as much as the whole search without a split. The gate decides once per
  // node, from what the search without a split (and the QT split, where that comes first: blocks above 16x16 at
  // medium) has found, whether the BT / TT splits are searched at all, and in which direction.
  // VVENC_DEV_MTT_GATE_LIMIT (default 1) raises the MTT depth limit of the pictures it acts in (2 = preset slow), and
  // VVENC_DEV_MTT_GATE_MTDEPTHS (bits, default 1) picks the MTT depths of the nodes it decides for: bit 0 the nodes
  // of the QT tree, bit 1 the parts of a first BT / TT split. The rules:
  //  - no residual  (VVENC_DEV_MTT_GATE_SKIP, default 1): the best mode without a split is inter and codes no
  //    residual: no BT / TT. With 1 the unsearched BT splits also count as "all skip", as their results would have,
  //    so a QT split that follows them (16x16) is stopped the way it was before; 2 leaves that QT split to its rules.
  //  - QT far behind (VVENC_DEV_MTT_GATE_QT, a ratio, default 1.02, 0 = off): the QT split was searched and costs
  //    more than ratio x the cost without a split: no BT / TT. A QT split can code anything a BT split can, for two
  //    more CU headers, so a BT split cannot be much cheaper than the QT split.
  //  - cheap block (VVENC_DEV_MTT_GATE_DENSITY, bits per luma sample, default 0 = off): the best cost so far is
  //    below density x lambda x luma samples: no BT / TT. What a split can gain is bounded by the cost it starts from.
  //  - unlike halves (VVENC_DEV_MTT_GATE_PAIR, an MV tolerance in MV units of 1/16 sample, default -1 = off): the QT
  //    split gave four unsplit CUs; a direction is searched only where both pairs of QT leaves that a BT split of that
  //    direction joins have the same kind of motion within the tolerance (x265's --limit-modes idea).
  //  - even error (VVENC_DEV_MTT_GATE_SHAPE, a share of the block's prediction error, default 0 = off): a split pays
  //    where one prediction fits one part of the block and not the other. The SAD of the best inter prediction
  //    without a split is compared between the parts a BT split (two halves) and a TT split (the middle half against
  //    the outer quarters) of each direction would make; a direction is searched only where one of the two
  //    differences is at least this share of the block's SAD (DevNodeStats::predErrorRows, splitErrorContrast).
  //  - weaker direction (VVENC_DEV_MTT_GATE_SHAPE_DIR, a ratio, default 0 = off): a direction whose error difference
  //    is below ratio x the other direction's is not searched.
  //  - one motion (v3; VVENC_DEV_MTT_GATE_MOTION, a share of the block's SAD, and VVENC_DEV_MTT_GATE_MOTION_BITS, bits
  //    at the motion lambda; both default 0 = off): a cheap first pass in the manner of SVT-AV1's PD0 and libaom's
  //    simple motion search, made from motion that is already known instead of a search. Most of what BT / TT gain in
  //    inter pictures is a block whose parts move differently (parts coded as skip / merge with other motion). The
  //    candidates a part could merge from are known before any split is searched: the motion of the best mode without
  //    a split, of the CUs along the node's left and upper edge, and of the reference picture at the same place
  //    (temporal candidates). xEstimateSplitMotion takes the SAD of each candidate over a 4 x 4 grid of cells, so the
  //    SAD of any QT / BT / TT partition with the best candidate per part is a sum of cells. A direction is searched
  //    only where its BT or TT partition lowers the SAD of the whole block by at least the share and the bits; a node
  //    whose best mode without a split is intra is left to the other rules.
  // VVENC_DEV_MTT_GATE_AS_TRIED=1 (v3, default 0): a BT split the gate stops while it lets the other direction through
  // counts as searched and lost for the rules that follow (QT after BT, TT of the other direction), instead of as not
  // done: without it those rules search more than before and give back part of the time saved.
  // DEV MTT census (VVENC_DEV_MTT_CENSUS=<path prefix>): every encoder thread writes one line per luma QT-tree node
  // of an inter picture to <prefix>.<n>.txt: costs and times of each split, the gate's inputs, the winner
  // (xWriteCensusNode lists the fields; work/smart_slow/mtt_census.py reads them). It changes no decision.
  bool                  m_devMttGate        = false;
  int                   m_devGateSkip       = 1;
  double                m_devGateQtRatio    = 1.02;
  double                m_devGateDensity    = 0.0;
  int                   m_devGatePairMvTol  = -1;
  double                m_devGateShape      = 0.0;
  double                m_devGateShapeDir   = 0.0;
  unsigned              m_devGateMaxLimit   = 1;
  unsigned              m_devGateMtDepths   = 1;
  double                m_devGateMotion     = 0.0;
  double                m_devGateMotionBits = 0.0;
  bool                  m_devGateAsTried    = false;
  FILE*                 m_censusFile        = nullptr;

  // DEV motion search gate (v3; VVENC_DEV_ME_GATE): in inter pictures, no motion search (ETM_INTER_ME, AMVR) at a
  // node that is a split part of a node whose best mode without a split is skip: 1 = every such part, 2 = the parts
  // of BT / TT splits only. The parts of a block that one merge candidate predicts without a residual rarely need a
  // searched motion vector (search census: 9 to 11 % of the ME + AMVR time for 1.5 to 2.4 % of its gain).
  int                   m_devMeGate         = 0;

  // DEV split stop (v4; VVENC_DEV_SPLIT_STOP=<T>, 0 = off): ends the search of a split between two of its parts once
  // the parts searched so far show that it will lose, the way libaom ends a partition type when the summed RD cost of
  // its sub-blocks passes the best cost so far, and SVT-AV1 stops a depth / an NSQ shape on the cost of its first
  // blocks. After each part but the last:
  //     cost of the parts searched (with the split flags)  +  REST x estimate for the parts not searched  >  T x best cost
  // stops the split: it gets no result, its other parts are not searched. The best cost is the node's best so far
  // (no split and the splits searched before). With T = 1 and REST = 0 the stopped split could not have won (the
  // costs of the other parts are not negative, apart from their deblocking offsets with EDO), and little time is
  // saved; the estimate for the rest is what makes the rule act early:
  //  - VVENC_DEV_SPLIT_STOP_REST (default 0): its weight. 1 = the rest is taken to cost its full share.
  //  - VVENC_DEV_SPLIT_STOP_SHARE (DevRestShare, default 1): the share of the whole-block cost that the parts not
  //    searched are taken to need: by area, by the prediction error of the best inter mode without a split in them
  //    (DevNodeStats::predErrorRows / Cols; by area where that is not known, and for QT), or, for a TT split after
  //    the BT split of its direction, from the costs of that BT split's halves (first part searched: half of the
  //    first half and the second half remain; two parts searched: half of the second half).
  //  - VVENC_DEV_SPLIT_STOP_BASE (default 0): the whole-block cost the share is taken of: 0 the best cost without a
  //    split, 1 the best cost so far (so a first direction or a QT split that won tightens the second direction).
  //  - VVENC_DEV_SPLIT_STOP_SPLITS (DevSplitKind bits, default 3 = BT and TT), VVENC_DEV_SPLIT_STOP_MTDEPTHS (bits of
  //    the MTT depth of the node that is split, default all).
  //  - VVENC_DEV_SPLIT_STOP_TT=1 (default 0): the TT split of a direction whose BT split was stopped is not searched.
  // VVENC_DEV_SPLIT_STOP_DIR2=<X> (default 0 = off; works without VVENC_DEV_SPLIT_STOP too): the second BT direction
  // (vertical) is not searched at all where the first one cost more than X x the best cost so far, in inter pictures
  // on luma nodes of the MTT depths above. It counts as a stopped BT split for VVENC_DEV_SPLIT_STOP_TT.
  // A stopped BT split counts as searched, with the rule's estimate as its cost, for the rules that compare the two
  // BT directions and choose the TT splits (trySplit). The rule acts in inter pictures on luma nodes inside the
  // picture whose split has no mode constraint (devStartSplit).
  // DEV split census: with the search census on, every searched split of a luma node of a census CTU writes a P line
  // (devFinishSplit lists the fields; work/smart_slow/split_stop_census.py replays the rule over its settings).
  double                m_devStopT          = 0.0;
  double                m_devStopRest       = 0.0;
  int                   m_devStopShare      = DEV_REST_BY_ERROR;
  int                   m_devStopBase       = 0;
  unsigned              m_devStopSplits     = 3;
  unsigned              m_devStopMtDepths   = ~0u;
  bool                  m_devStopTtFollows  = false;
  double                m_devStopDir2       = 0.0;

  // DEV intra gate with fallback (v4; VVENC_DEV_INTRA_GATE_GOOD=<RD cost / lambda per luma sample>, default 0 = off):
  // the nearby-use gate stops the intra search of a node (VVENC_DEV_TOOL_GATE bit 2) only where the best inter mode
  // is already good, that is where its RD cost is below this x lambda x luma samples (the census has 3 to 17 where
  // the gate acts, crowd, q32). Where inter prediction fails, intra is searched whatever the CUs nearby use: a missed
  // intra CU there costs distortion that the later pictures inherit.
  // (The encoder itself does not try intra after an inter mode without any residual, so that case needs no rule.)
  double                m_devIntraGateGood  = 0.0;
  int                   m_devCtuCounts[NUM_DEV_CTU_COUNTS] = { 0, 0, 0, 0, 0 };   // of the CTU being coded

  // DEV skipped-parent gate (v5; VVENC_DEV_PARENT_GATE=<DevParentSearch bits>, 0 = off): in inter pictures, the
  // chosen searches are not made at a node that is a split part of a "skipped" node, that is of a node whose best
  // mode without a split is skip. Where one merge candidate predicts the whole block so well that no residual is
  // coded, its parts rarely need more than a merge candidate of their own (x265 --limit-modes and rskip, the depth
  // refinement of SVT-AV1). VVENC_DEV_ME_GATE=1 is this rule for the motion search (0x4001 here); VVenC's own
  // FastInferMerge is this rule for the motion search and intra in the highest temporal layers only (off in medium
  // and slow).
  //  - VVENC_DEV_PARENT_GATE_LEVEL (default 1): what counts as skipped. 1 = skip (merge without a residual);
  //    2 = any merge mode, with or without a residual (weaker evidence, several times the nodes).
  //  - The bits: 0x4000 motion search, 0x1 AMVR; inside the motion search, for use without 0x4000: 0x8 affine ME,
  //    0x80 SMVD, 0x40 BCW; inside the merge search: 0x10 GEO, 0x20 CIIP, 0x400 MMVD, 0x200 affine / sub-block merge;
  //    0x2 SBT (merge and searched motion); 0x4 intra, only where the node has an inter mode already; 0x100 the
  //    BT / TT splits and 0x8000 the QT split of the node.
  //  - VVENC_DEV_PARENT_GATE_DEPTH (default 1), for the two split bits: the splits of a node are not searched where
  //    the node this many levels above it is skipped. 1 = the parts of a skipped node are not split again; 2 = they
  //    are, and their parts are not; 0 = a skipped node itself is not split (the reverse question: is the split
  //    search needed there at all).
  //  - VVENC_DEV_PARENT_GATE_OWN=1 (default 0), for the two split bits: only where the node's own best mode without
  //    a split is skipped too (VVenC's skip-history rule in trySplit asks for two skipped levels above the node).
  // The split bits are asked after VVenC's own split rules, like the other DEV rules, and a BT split they stop counts
  // as not done. With the search census on, each CTU writes an H line with what the gate stopped (devFinishCtu).
  uint32_t              m_devParentGate      = 0;
  int                   m_devParentLevel     = 1;
  int                   m_devParentDepth     = 1;
  bool                  m_devParentOwn       = false;
  int                   m_devParentStops[NUM_DEV_PARENT_SEARCHES] = { 0 };   // of the CTU being coded, per search

  // DEV nearby-use gate (VVENC_DEV_TOOL_GATE=<DevTool bits>; a prototype for smart medium / slow). In inter pictures
  // the search of a gated tool runs at a node only where a CU nearby was coded with that tool: where a tool wins
  // depends on the content (affine on zooms, GEO on object edges, intra on uncovered areas, BT / TT on detail), and
  // content is alike nearby. "Nearby" (VVENC_DEV_TOOL_GATE_SRC, DevNearby bits, default 7) is the CUs left of
  // and above the node, the CTUs left, above and above left of its CTU, and the CTU at the same place in the first
  // reference picture of each list; bit 3 (v3) adds the best mode without a split of the node this one is a part of. So that a tool can come back where nothing nearby uses it, every
  // VVENC_DEV_TOOL_GATE_PROBE-th CTU (default 4, 0 = none; a lattice that moves with the POC) searches everything.
  // All of this is the same whatever the thread count: only CTUs and pictures that are finished before the CTU
  // starts are read (with IFP a reference picture may not be, so the reference pictures are not read then).
  // DEV search census (VVENC_DEV_SEARCH_CENSUS=<path prefix>): every encoder thread writes to <prefix>.<n>.txt one
  // line per luma node (S) with the time and best cost of each of its searches, the best costs with and without each
  // tool, and what the gates read; and one line per CTU (C) with the tools of its final CUs. xWriteSearchCensusNode
  // and devFinishCtu list the fields; work/smart_slow/search_census.py reads them. VVENC_DEV_SEARCH_CENSUS_STEP=<n>
  // writes the nodes of every n-th CTU only. It changes no decision.
  uint32_t              m_devToolGate        = 0;       // DevTool bits the nearby-use gate acts on
  uint32_t              m_devToolGateSources = 7;       // DevNearby bits
  int                   m_devToolGateProbe   = 4;
  FILE*                 m_searchCensusFile   = nullptr;
  int                   m_searchCensusStep   = 1;
  bool                  m_devFinalCensus     = false;   // v8: the search census also writes the CUs of the final partition (xWriteFinalCus)
  int                   m_searchCensusPoc    = -1;      // v8, VVENC_DEV_SEARCH_CENSUS_POC: node lines for this picture only (-1: all)
  // state of the CTU being coded (devInitCtu)
  bool                  m_devGateInPicture   = false;   // the nearby-use gate acts in this picture
  bool                  m_devCtuIsProbe      = false;   // this CTU searches every tool
  bool                  m_devCtuInCensus     = false;   // the nodes of this CTU are written
  uint32_t              m_devToolsNearCtu[NUM_DEV_NEARBY] = { 0, 0, 0, 0 };   // DevTool bits of the CTUs nearby; only [DEV_NEAR_CTUS] and [DEV_NEAR_REFS] are used
  int                   m_devNodeCount       = 0;       // nodes started by this encoder thread
  int64_t               m_devCtuStart        = 0;       // census clock at the start of the CTU

  // DEV smart AMVR gate and smart SBT gate (v7; described at devAmvrGateStops and devSbtGateStops below)
  bool                  m_devAmvrGateSkip    = false;
  double                m_devAmvrGateBehind  = 0.0;
  int                   m_devAmvrGateMvd     = 0;
  int                   m_devAmvrGate4PelMvd = 0;
  double                m_devSbtGate         = 0.0;
  int                   m_devV7Stops[NUM_DEV_V7_STOPS] = { 0, 0, 0 };   // of the CTU being coded
  bool    xAmvrGateIsOn         () const { return m_devAmvrGateSkip || m_devAmvrGateBehind > 0.0 || m_devAmvrGateMvd > 0 || m_devAmvrGate4PelMvd > 0; }

  bool    xToolUsedNearby       ( const DevTool tool );
  bool    xNearbyGateAllows     ( const DevTool tool ) { return !( m_devToolGate >> tool & 1 ) || xToolUsedNearby( tool ); }
  bool    xIsSkippedNode        ( const ComprCUCtx& node ) const;
  bool    xParentGateStops      ( const int search );
  bool    xParentGateStopsSplit ( const PartSplit split );
  void    xNoteNoSplitResult    ( const CodingUnit& bestCU, const Partitioner& partitioner );
  void    xEstimateSplitMotion  ( const CodingUnit& bestCU, const Partitioner& partitioner );
  bool    xMotionGateIsOn       () const { return m_devGateMotion > 0.0 || m_devGateMotionBits > 0.0; }
  void    xNoteModeResult       ( const EncTestMode& encTestmode, const CodingStructure& tempCS, const bool useEDO );
  void    xWriteSearchCensusNode( const Partitioner& partitioner );
  void    xWriteFinalCus        ( const CodingStructure& ctuCS ) const;
  bool    xIntraGateStops       ( const CodingStructure& bestCS, const CodingUnit& bestCU, const Area& lumaArea );
  double  xRestOfSplitEstimate  ( const DevSplitTry& splitTry ) const;
  bool    xSecondDirectionStops ( const CodingStructure& cs, const Partitioner& partitioner ) const;
  bool    xSplitStopReadsError  () const { return m_devStopT > 0.0 && m_devStopRest > 0.0 && m_devStopShare != DEV_REST_BY_AREA; }

  bool    xIsMttGateNode    ( const CodingStructure& cs, const Partitioner& partitioner ) const;
  MttGate xDecideMttGate    ( const Partitioner& partitioner );
  bool    xMttGateVetoes    ( const PartSplit split, const CodingStructure& cs, const Partitioner& partitioner );
  void    xNoteQtSplitResult( const CodingStructure& qtCS );
  void    xWriteCensusNode  ( const Partitioner& partitioner );

  // DEV prototype of the smart-slow split gates, both off unless their environment variable is set (read in init).
  // They act on the inter pictures whose MTT depth limit L is the highest of the encode's layers and is 2 or 3
  // (3 at preset slower, whose top layer with limit 2 is left as it is; 2 at preset slow), on the BT / TT splits
  // that make CUs of MTT depth L, the deepest level. VVENC_DEV_SPLIT_ALL_LAYERS=1 lets them act on the layers with
  // a lower limit of at least 2 as well.
  // Split refinement (VVENC_DEV_SPLIT_REFINE=1): every MTT root (a luma node that no chosen BT / TT split lies above)
  // searches its BT / TT splits twice. The capped pass leaves the deepest level out. The refinement pass then
  // searches again the root's splits whose cost is within VVENC_DEV_SPLIT_REFINE_TR (default 1.0) of the root's
  // best, now with the deepest level searched as always. With L = 3 it searches, inside them, only the children's
  // splits within VVENC_DEV_SPLIT_REFINE_TP (default 1.0) of that child's best. With L = 2 the root's children are
  // the nodes whose splits were left out, so there are no costs to choose by: they search their splits as always.
  // The capped pass is searched as a picture with MTT depth limit L - 1 is: its split rules, its reuse of stored CU
  // results and its split-cost table read that limit (ComprCUCtx::mttDepthLimit). Only the split flags are coded
  // and priced with the picture's limit L. With L = 2 and QtbttExtraFast above 1 (preset slow) this makes the MTT
  // roots search their QT split before BT / TT, as a picture with limit 1 does.
  // Refinement after QT (VVENC_DEV_SPLIT_REFINE_AFTERQT=1, with the split refinement): the refinement pass of a root
  // runs after the root's QT split instead of before it, and only if QT was not searched or the best BT / TT split
  // of the capped pass costs at most VVENC_DEV_SPLIT_REFINE_QTM (default 1.05) times the QT split.
  // Check of the capped pass (VVENC_DEV_SPLIT_REFINE_CHECK=1, with the split refinement, for encodes with one
  // MaxMTTDepth of 1 or 2 on all layers): the inter pictures, of limit L - 1, are searched as pictures with limit L
  // by capped passes alone, and their split flags are priced with the limit L - 1 they are coded with. The stream
  // must then be identical to the one without DEV switches; a difference means that something still reads the
  // limit L in the capped pass.
  // Cover gate (VVENC_DEV_SPLIT_COVER=1): the deepest level is not searched in a block whose whole area lies in
  // blocks where the same MTT root has searched it before, unless it won on all of that area.
  // Root rules (v9, VVENC_DEV_SPLIT_REFINE_RULES=<DevRootRule bits>, with the split refinement, not with its check):
  // a BT / TT split of an MTT root that the capped pass does not search has no cost, so the refinement pass never
  // searches it again. At preset slow (L = 2) the capped pass reads limit 1, and fast rules that a picture with limit
  // 2 does not run stop such splits; at preset slower (limit 3 to 2) none of these rules changes. Each bit makes one
  // group of rules at the root read what the full search would: the picture's limit, or no rule where the rule reads
  // the capped BT results.
  // Lowest temporal layer of the split refinement (v10, VVENC_DEV_SPLIT_REFINE_MINTL=<n>, with the split refinement,
  // not with its check): the refinement and its capped pass act only on pictures with temporal id n or higher; the
  // others are searched as without the refinement. A wrong split in a picture that later pictures predict from
  // costs in those pictures too, and the pictures of the highest temporal ids are predicted from by few or none.
  // 0 (the default): all pictures.
  enum DevRootRule
  {
    DRR_QT_ORDER    = 1 << 0,   // QT before BT / TT (QtbttExtraFast above 1 with limit 1), with its stop of BT / TT where QT went deeper
    DRR_SKIP_DEPTH  = 1 << 1,   // no BT / TT where the best mode is skip and the MTT depth left is small (QtbttExtraFast)
    DRR_COST_TABLE  = 1 << 2,   // the split-cost table of the root's QT, BT and TT splits
    DRR_QT_SKIP     = 1 << 3,   // the stop of QT after both BT splits lost to no split (where BT comes before QT)
    DRR_TT_RULES    = 1 << 4,   // no TT stop that reads the capped BT results (fast TT threshold, direction, no residual)
    DRR_MORE_SPLITS = 1 << 5,   // capped BT results whose CUs are all skipped do not use up the root's later splits
    // the three TT rules of DRR_TT_RULES one by one (a rule is left out if its bit or DRR_TT_RULES is set)
    DRR_TT_NO_RESIDUAL = 1 << 6,   // no TT where no split beat BT in that direction and has no residual
    DRR_TT_DIRECTION   = 1 << 7,   // no TT in the direction whose BT split cost more (QtbttExtraFast)
    DRR_TT_THRESHOLD   = 1 << 8,   // no TT where its direction's BT cost more than FastTTSplit's threshold times no split or the other BT
    // keeps the QT order of the capped pass (QT first) but not its stop of BT / TT after a QT split that went deeper,
    // where QT comes first only because of the lowered limit
    DRR_QT_DEEPER      = 1 << 9,
  };
  enum DeepSplitPass { DSP_ALL, DSP_CAPPED, DSP_REFINE };
  unsigned              m_lowestGatedMttLimit  = 1;   // the gates act on inter pictures with an MTT depth limit from this ...
  unsigned              m_highestGatedMttLimit = 0;   // ... to this; the range is empty (as here) when they act on none
  bool                  m_devSplitRefine      = false;
  double                m_devRefineRootMargin = 1.0;
  double                m_devRefineNodeMargin = 1.0;
  bool                  m_devRefineAfterQt    = false;
  double                m_devRefineQtMargin   = 1.05;
  bool                  m_devCappedPassCheck  = false;
  unsigned              m_devRootRules        = 0;      // v9: DevRootRule bits
  unsigned              m_devRefineMinTLayer  = 0;      // v10: VVENC_DEV_SPLIT_REFINE_MINTL
  bool                  m_refinesThisCtu      = false;  // v10: the split refinement acts on the current CTU's picture (set per CTU: an encoder thread's CTUs may come from different pictures)
  bool                  m_devGateAllLayers    = false;   // VVENC_DEV_SPLIT_ALL_LAYERS; off in the check of the capped pass
  unsigned              m_searchDepthRaise    = 0;    // check of the capped pass: how much the search's MTT depth limit of this CTU is above the picture's (0 or 1)
  unsigned              m_devCheckRaisedCtus  = 0;    // check of the capped pass: CTUs whose search limit was raised (printed at the end)
  unsigned              m_devCappedPasses     = 0;    // ... and capped passes begun
  bool                  m_devSplitCover       = false;
  DeepSplitPass         m_deepSplitPass       = DSP_ALL;
  unsigned              m_mttRootCount        = 0;    // cover gate: MTT roots started in this CTU
  std::vector<unsigned> m_deepSeenByRoot;             // cover gate, per 4x4 luma block of the CTU: the last MTT root that searched the deepest level here
  std::vector<unsigned> m_deepWonByRoot;              // ... and the last one for which a split of the deepest level won here

  bool     xGatesDeepSplits       ( const CodingStructure& cs, const Partitioner& partitioner ) const;
  bool     xChoosesSplitsToRefine ( const Partitioner& partitioner ) const;
  unsigned xRuleMttDepthLimit     ( const CodingStructure& cs, const Partitioner& partitioner ) const;
  unsigned xRootRuleLimit         ( const DevRootRule rule, const Partitioner& partitioner, const unsigned ruleLimit ) const;
  bool     xRootRuleOff           ( const unsigned rules, const Partitioner& partitioner ) const;
  bool     xQtBeatsCappedSplits   ( const ComprCUCtx& rootCtx ) const;
  bool     xDevGateSkipsDeepSplit ( const CodingStructure& cs, const Partitioner& partitioner );
  unsigned xSplitsWithinMargin    ( const ComprCUCtx& cuECtx, const double margin ) const;
  void     xStoreSplitsToRefine   ( const Partitioner& partitioner );
  bool     xDeepSplitsCovered     ( const Area& lumaArea, const unsigned rootId ) const;
  void     xStampDeepSplitResult  ( const Area& lumaArea, const ComprCUCtx& cuECtx );
  bool     xKeepIfBest            ( const EncTestMode& encTestmode, CodingStructure* tempCS, const bool useEDO );

public:
  ComprCUCtx*           comprCUCtx;

  ~EncModeCtrl    () { destroy(); }

  void init               ( const VVEncCfg& encCfg, RdCost *pRdCost );
  void destroy            ();
  void initCTUEncoding    ( const Slice &slice );
  // cuLeft / cuAbove: the restricted left and above neighbours of the CU (DeriveCtx::determineNeighborCus)
  void initCULevel        ( Partitioner &partitioner, const CodingStructure& cs, int  MergeSimpleFlag, const CodingUnit* cuLeft, const CodingUnit* cuAbove );
  void finishCULevel      ( Partitioner &partitioner );

  bool tryMode            ( const EncTestMode& encTestmode, const CodingStructure &cs, Partitioner& partitioner );
  bool trySplit           ( const EncTestMode& encTestmode, const CodingStructure &cs, Partitioner& partitioner, const EncTestMode& lastTestmode );
  bool useModeResult      ( const EncTestMode& encTestmode, CodingStructure*& tempCS,  Partitioner& partitioner, const bool useEDO );

  void beforeSplit        ( Partitioner& partitioner );

  // DEV split refinement, called by EncCu::xCompressCU at an MTT root, in this order. The root's QT split is searched
  // outside the two passes: before the capped pass, or after either of them (refinesAfterQt).
  bool     beginCappedSplitPass ( const CodingStructure& cs, const Partitioner& partitioner );
  void     endCappedSplitPass   ();
  unsigned beginSplitRefinePass ( const Partitioner& partitioner );
  void     endSplitRefinePass   ();
  bool     refinesAfterQt       () const { return m_devRefineAfterQt; }

  // DEV check of the capped pass (see above). Without it the first changes nothing and the second returns
  // partitioner.maxBTD.
  void     raiseSearchDepthLimitForCheck( Partitioner& partitioner, const Slice& slice );
  unsigned codedMttDepthLimit   ( const Partitioner& partitioner ) const { return partitioner.maxBTD - m_searchDepthRaise; }
  // the MTT depth limit whose split-cost table the current node's splits read (EncCu::xCheckModeSplitInternal)
  unsigned splitCostMttDepthLimit( const Partitioner& partitioner ) const { return xRootRuleLimit( DRR_COST_TABLE, partitioner, comprCUCtx->mttDepthLimit ); }

  // DEV MTT census and search census: the clock is 0, and the time is dropped, unless one of them is on
  int64_t censusClock        () const;
  void    censusAddSplitTime ( const EncTestMode& encTestmode, const int64_t startTime );
  void    devAddTime         ( const DevTimer timer, const int64_t startTime );

  // DEV search census (v7) and smart AMVR gate: the result of the node's motion search with quarter-sample MVDs, and
  // the node's best mode when its AMVR search is asked for
  void    devNoteMeResult    ( const CodingUnit& cu );
  void    devNoteBeforeImv   ( const CodingStructure& bestCS );

  // DEV smart AMVR gate (v7), all off by default. AMVR (MVDs in whole, 4 or half samples) pays where coding the MVD
  // coarsely saves bits, or where the half-sample filter predicts better, and only where a searched motion vector
  // is near the node's best mode at all. The AMVR search of a node is not made where
  //  - VVENC_DEV_AMVR_GATE_SKIP=1: the node's best mode is skip (as AMVR speeds 3 and 4 do; slow has AMVR 1);
  //  - VVENC_DEV_AMVR_GATE_BEHIND=<ratio>: the motion search with quarter-sample MVDs cost more than ratio x the
  //    node's best mode (a merge mode or intra is that far ahead of any searched motion);
  //  - VVENC_DEV_AMVR_GATE_MVD=<quarter samples>: the largest MVD component of that search's result is below this
  //    (2: every MVD is 0 or a quarter sample, so there are no MVD bits to save).
  // VVENC_DEV_AMVR_GATE_4PEL=<quarter samples>: inside the AMVR search, no 4-sample MVDs where that largest MVD
  // component is below this.
  // The last two rules do not act where the motion search gave no result. Census (three HD clips, q27, slow): SKIP
  // with BEHIND 1.2 and MVD 2 stops 48 % of the AMVR search time for 13 % of its gain; AMVR 5 (no AMVR search where
  // merge is best) 73 % for 36 %.
  bool    devAmvrGateStops     ( const CodingStructure& bestCS );
  bool    devAmvrGateStops4Pel ();
  // DEV smart SBT gate (v7; VVENC_DEV_SBT_GATE=<evenness>, 0 = off): SBT codes the residual of one half or quarter
  // of the block and leaves the rest without one, so it pays where the prediction error is concentrated in the coded
  // part. An SBT mode is not RD-tested where the part it leaves uncoded has more than evenness x its share of the
  // area of the block's squared prediction error (1 = the error is spread evenly; the encoder's own rules stop a
  // mode only where the uncoded part's error alone costs more than the best mode). A mode taken from the SBT history
  // is always tested. estimatedDist and blockSse: InterSearch::getEstDistSbt of the mode and of NUMBER_SBT_MODE.
  bool    devSbtGateStops      ( const int sbtMode, const Distortion estimatedDist, const Distortion blockSse );
  // DEV search census (v7), smart SBT: one RD test of an SBT mode at the current node (writes a B line)
  bool    devCensusOfNode    () const { return m_searchCensusFile && m_devCtuInCensus && comprCUCtx; }
  void    devNoteSbtTest     ( const EncTestMode& encTestmode, const int sbtMode, const Distortion estimatedDist, const Distortion blockSse,
                               const double sbtOffCost, const double bestCostBefore, const bool modeFromHistory, const int64_t startTime );

  // DEV nearby-use gate and search census, once per CTU: before its first node, and with its final luma CUs
  void    devInitCtu         ( const CodingStructure& picCS, const UnitArea& ctuArea );
  void    devFinishCtu       ( const CodingStructure& ctuCS, const UnitArea& ctuArea );
  // DEV nearby-use gate and skipped-parent gate: false where one of them stops the search of this tool at the current node
  bool    devToolAllowed     ( const DevTool tool ) { return xNearbyGateAllows( tool ) && !( m_devParentGate && xParentGateStops( tool ) ); }

  // DEV split stop and split census, around the search of one split of the current node: before its first part,
  // after each part (true: stop the search here), and once at its end (finalCost: MAX_DOUBLE for a stopped split)
  void    devStartSplit      ( DevSplitTry& splitTry, const EncTestMode& encTestmode, const CodingStructure& bestCS, const Partitioner& partitioner,
                               const int64_t splitBits, const bool hasModeConstraint );
  bool    devSplitPartDone   ( DevSplitTry& splitTry, const CodingStructure& splitCS, const Area& partArea, const bool isLastPart );
  void    devFinishSplit     ( const DevSplitTry& splitTry, const double finalCost );
};

// DEV search census: adds the time from its construction to its destruction to a timer of the current node
class DevScopedTimer
{
public:
  DevScopedTimer( EncModeCtrl& modeCtrl, const DevTimer timer, const bool active = true )
    : m_modeCtrl( modeCtrl ), m_timer( timer ), m_startTime( active ? modeCtrl.censusClock() : 0 ) {}
  ~DevScopedTimer() { if( m_startTime ) m_modeCtrl.devAddTime( m_timer, m_startTime ); }

private:
  EncModeCtrl&   m_modeCtrl;
  const DevTimer m_timer;
  const int64_t  m_startTime;
};

} // namespace vvenc

//! \}

