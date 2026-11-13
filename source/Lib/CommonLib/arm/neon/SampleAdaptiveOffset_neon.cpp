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

/** \file     SampleAdaptiveOffset_neon.cpp
    \brief    Sample adaptive offset class, Neon version of calcSaoStatisticsBo.
*/

#include "../CommonDefARM.h"
#include "CommonLib/CommonDef.h"
#include "CommonLib/SampleAdaptiveOffset.h"
#include "sum_neon.h"
#include <arm_neon.h>

namespace vvenc
{

#if ENABLE_SIMD_OPT_SAO && defined( TARGET_SIMD_ARM )

// One sample of the band-offset statistics in a packed table entry: the entry holds count * 2^32 + diffSum.
static inline void addToBand( int64_t* table, const int band, const int diff )
{
  table[band] += ( int64_t( 1 ) << 32 ) + diff;
}

// Lane j (0 to 3) of four int16 values packed in a uint64
static inline int lane16( const uint64_t lanes, const int j )
{
  return int16_t( lanes >> ( 16 * j ) );
}

void calcSaoStatisticsBo_neon( int width, int endX, int endY, Pel* srcLine, Pel* orgLine, int srcStride, int orgStride,
                               int channelBitDepth, int64_t* count, int64_t* diff )
{
  const int shift = channelBitDepth - NUM_SAO_BO_CLASSES_LOG2;
  const int16x8_t negShift = vdupq_n_s16( -shift );

  // Table k takes the samples at x % 8 == k; the tables are summed at the end. Each entry packs count and diffSum: at
  // most 128 x 128 samples with |org - src| < 2^15 keep |diffSum| < 2^29, so the low 32 bits read as a signed integer
  // are diffSum and the rest is the count.
  int64_t packed[8][NUM_SAO_BO_CLASSES] = {};

  do
  {
    int x = 0;
    for( ; x + 8 <= endX; x += 8 )
    {
      const int16x8_t src = vld1q_s16( srcLine + x );
      const int16x8_t org = vld1q_s16( orgLine + x );
      const int16x8_t band = vshlq_s16( src, negShift );
      const int16x8_t d = vsubq_s16( org, src );

      // the differences are moved to general registers as two uint64 and taken from them with shifts, the bands lane
      // by lane
      const uint64_t diffLo = vgetq_lane_u64( vreinterpretq_u64_s16( d ), 0 );
      const uint64_t diffHi = vgetq_lane_u64( vreinterpretq_u64_s16( d ), 1 );

      addToBand( packed[0], vgetq_lane_s16( band, 0 ), lane16( diffLo, 0 ) );
      addToBand( packed[1], vgetq_lane_s16( band, 1 ), lane16( diffLo, 1 ) );
      addToBand( packed[2], vgetq_lane_s16( band, 2 ), lane16( diffLo, 2 ) );
      addToBand( packed[3], vgetq_lane_s16( band, 3 ), lane16( diffLo, 3 ) );
      addToBand( packed[4], vgetq_lane_s16( band, 4 ), lane16( diffHi, 0 ) );
      addToBand( packed[5], vgetq_lane_s16( band, 5 ), lane16( diffHi, 1 ) );
      addToBand( packed[6], vgetq_lane_s16( band, 6 ), lane16( diffHi, 2 ) );
      addToBand( packed[7], vgetq_lane_s16( band, 7 ), lane16( diffHi, 3 ) );
    }
    for( ; x < endX; ++x )
    {
      addToBand( packed[0], srcLine[x] >> shift, orgLine[x] - srcLine[x] );
    }

    srcLine += srcStride;
    orgLine += orgStride;
  } while( --endY != 0 );

  for( int i = 0; i < NUM_SAO_BO_CLASSES; i += 2 )
  {
    int64x2_t sum = vld1q_s64( packed[0] + i );
    for( int k = 1; k < 8; k++ )
    {
      const int64x2_t next = vld1q_s64( packed[k] + i );
      sum = vaddq_s64( sum, next );
    }

    const int64x2_t diffSum = vmovl_s32( vmovn_s64( sum ) );
    const int64x2_t bandCount = vshrq_n_s64( vsubq_s64( sum, diffSum ), 32 );

    vst1q_s64( diff + i, diffSum );
    vst1q_s64( count + i, bandCount );
  }
}

// The EO statistics kernels follow the x86 kernels' vector path (widths that are multiples of 16): per 16 samples the
// edge type, sign( src - neighbour1 ) + sign( src - neighbour2 ) + 2 in 0..4, selects the class whose difference sum
// gets org - src and whose count gets 1. Masked samples (the first one when startX is 1, the last width - endX ones)
// get type -1, which no class matches. The 32-bit sums cannot overflow (at most 128 x 128 differences, each below
// 2^10 in magnitude). Other widths take the C functions, which compute what x86's scalar branch does.
void calcSaoStatisticsEo0_Core( int width, int startX, int endX, int endY, Pel* srcLine, Pel* orgLine, int srcStride,
                                int orgStride, int64_t* count, int64_t* diff );
void calcSaoStatisticsEo90_Core( int width, int endX, int startY, int endY, Pel* srcLine, Pel* orgLine, int srcStride,
                                 int orgStride, int64_t* count, int64_t* diff, int8_t* signUpLine );
void calcSaoStatisticsEo135_Core( int width, int startX, int endX, int endY, Pel* srcLine, Pel* orgLine, int srcStride,
                                  int orgStride, int64_t* count, int64_t* diff, int8_t* signUpLine,
                                  int8_t* signDownLine );
void calcSaoStatisticsEo45_Core( int width, int startX, int endX, int endY, Pel* srcLine, Pel* orgLine, int srcStride,
                                 int orgStride, int64_t* count, int64_t* diff, int8_t* signUpLine );

// sign( a - b ) per lane: -1, 0 or 1
static inline int16x8_t saoSign_neon( const int16x8_t a, const int16x8_t b )
{
  return vsubq_s16( vreinterpretq_s16_u16( vcltq_s16( a, b ) ), vreinterpretq_s16_u16( vcgtq_s16( a, b ) ) );
}

struct SaoEoSums
{
  int32x4_t diff[NUM_SAO_EO_CLASSES];
  int32x4_t count[NUM_SAO_EO_CLASSES];   // negative: the sum of the -1 masks

  SaoEoSums()
  {
    for( int c = 0; c < NUM_SAO_EO_CLASSES; c++ )
    {
      diff[c]  = vdupq_n_s32( 0 );
      count[c] = vdupq_n_s32( 0 );
    }
  }

  // adds 16 samples: types t0, t1 (-1 for masked samples), differences org - src d0, d1
  void add( const int16x8_t t0, const int16x8_t t1, const int16x8_t d0, const int16x8_t d1 )
  {
    for( int c = 0; c < NUM_SAO_EO_CLASSES; c++ )
    {
      const int16x8_t m0 = vreinterpretq_s16_u16( vceqq_s16( t0, vdupq_n_s16( c ) ) );
      const int16x8_t m1 = vreinterpretq_s16_u16( vceqq_s16( t1, vdupq_n_s16( c ) ) );
      diff[c]  = vpadalq_s16( diff[c], vaddq_s16( vandq_s16( m0, d0 ), vandq_s16( m1, d1 ) ) );
      count[c] = vpadalq_s16( count[c], vaddq_s16( m0, m1 ) );
    }
  }

  // stores the class totals (Eo0, Eo90) or adds them to the caller's first-line statistics (Eo135, Eo45), as x86 does
  void store( int64_t* countOut, int64_t* diffOut, const bool accumulate ) const
  {
    for( int c = 0; c < NUM_SAO_EO_CLASSES; c++ )
    {
      const int64_t d = horizontal_add_s32x4( diff[c] );
      const int64_t n = -horizontal_add_s32x4( count[c] );
      diffOut[c]  = accumulate ? diffOut[c] + d : d;
      countOut[c] = accumulate ? countOut[c] + n : n;
    }
  }
};

// The masks of the first sample (lane 0 of the first 16) and of the last width - endX samples (the upper lanes of the
// last 16; width - endX is at most 5)
static inline int16x8_t saoMaskFirst_neon( const int startX )
{
  return vsetq_lane_s16( startX ? -1 : 0, vdupq_n_s16( 0 ), 0 );
}

static inline int16x8_t saoMaskLast_neon( const int numMasked )
{
  CHECKD( numMasked > 8, "at most 8 masked samples" );
  static const int16_t maskTab[16] = { 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1 };
  return vld1q_s16( &maskTab[numMasked] );
}

// Eo0, Eo135, Eo45: the neighbours of srcLine[x] are srcLine[x + offset1] and srcLine[x + offset2]
static void saoEoTwoNeighbours_neon( SaoEoSums& sums, const int width, const int startX, const int endX,
                                     const int numRows, const Pel* srcLine, const Pel* orgLine, const int srcStride,
                                     const int orgStride, const ptrdiff_t offset1, const ptrdiff_t offset2 )
{
  const int16x8_t maskFirst = saoMaskFirst_neon( startX );
  const int16x8_t maskLast  = saoMaskLast_neon( width - endX );
  const int16x8_t two       = vdupq_n_s16( 2 );
  for( int y = 0; y < numRows; y++, srcLine += srcStride, orgLine += orgStride )
  {
    for( int x = 0; x < width; x += 16 )
    {
      const int16x8_t s0  = vld1q_s16( srcLine + x );
      const int16x8_t s1  = vld1q_s16( srcLine + x + 8 );
      const int16x8_t a0  = vld1q_s16( srcLine + x + offset1 );
      const int16x8_t a1  = vld1q_s16( srcLine + x + 8 + offset1 );
      const int16x8_t b0  = vld1q_s16( srcLine + x + offset2 );
      const int16x8_t b1  = vld1q_s16( srcLine + x + 8 + offset2 );
      const int16x8_t o0  = vld1q_s16( orgLine + x );
      const int16x8_t o1  = vld1q_s16( orgLine + x + 8 );
      int16x8_t       t0  = vaddq_s16( vaddq_s16( saoSign_neon( s0, a0 ), saoSign_neon( s0, b0 ) ), two );
      int16x8_t       t1  = vaddq_s16( vaddq_s16( saoSign_neon( s1, a1 ), saoSign_neon( s1, b1 ) ), two );
      if( x == 0 )
      {
        t0 = vorrq_s16( t0, maskFirst );
      }
      if( x == width - 16 )
      {
        t1 = vorrq_s16( t1, maskLast );
      }
      sums.add( t0, t1, vsubq_s16( o0, s0 ), vsubq_s16( o1, s1 ) );
    }
  }
}

static void calcSaoStatisticsEo0_neon( int width, int startX, int endX, int endY, Pel* srcLine, Pel* orgLine,
                                       int srcStride, int orgStride, int64_t* count, int64_t* diff )
{
  if( width & 15 )
  {
    calcSaoStatisticsEo0_Core( width, startX, endX, endY, srcLine, orgLine, srcStride, orgStride, count, diff );
    return;
  }
  SaoEoSums sums;
  saoEoTwoNeighbours_neon( sums, width, startX, endX, endY, srcLine, orgLine, srcStride, orgStride, -1, 1 );
  sums.store( count, diff, false );
}

// count and diff point at class 2 here, as for the C function; the rows start at the second line (j = 1 .. endY - 1)
static void calcSaoStatisticsEo135_neon( int width, int startX, int endX, int endY, Pel* srcLine, Pel* orgLine,
                                         int srcStride, int orgStride, int64_t* count, int64_t* diff,
                                         int8_t* signUpLine, int8_t* signDownLine )
{
  if( width & 15 )
  {
    calcSaoStatisticsEo135_Core( width, startX, endX, endY, srcLine, orgLine, srcStride, orgStride, count, diff,
                                 signUpLine, signDownLine );
    return;
  }
  SaoEoSums sums;
  saoEoTwoNeighbours_neon( sums, width, startX, endX, endY - 1, srcLine, orgLine, srcStride, orgStride,
                           -1 - srcStride, 1 + srcStride );
  sums.store( count - 2, diff - 2, true );
}

static void calcSaoStatisticsEo45_neon( int width, int startX, int endX, int endY, Pel* srcLine, Pel* orgLine,
                                        int srcStride, int orgStride, int64_t* count, int64_t* diff,
                                        int8_t* signUpLine )
{
  if( width & 15 )
  {
    calcSaoStatisticsEo45_Core( width, startX, endX, endY, srcLine, orgLine, srcStride, orgStride, count, diff,
                                signUpLine );
    return;
  }
  SaoEoSums sums;
  saoEoTwoNeighbours_neon( sums, width, startX, endX, endY - 1, srcLine, orgLine, srcStride, orgStride,
                           1 - srcStride, srcStride - 1 );
  sums.store( count - 2, diff - 2, true );
}

// Eo90: the up sign of each row is the previous row's down sign, kept per 16 samples (x86's vsigns); the type is
// sign( src - below ) - sign( above - src ) + 2 with the first up sign from the row above startY's first row
static void calcSaoStatisticsEo90_neon( int width, int endX, int startY, int endY, Pel* srcLine, Pel* orgLine,
                                        int srcStride, int orgStride, int64_t* count, int64_t* diff,
                                        int8_t* signUpLine )
{
  if( width & 15 )
  {
    calcSaoStatisticsEo90_Core( width, endX, startY, endY, srcLine, orgLine, srcStride, orgStride, count, diff,
                                signUpLine );
    return;
  }
  const int16x8_t maskLast = saoMaskLast_neon( width - endX );
  const int16x8_t two      = vdupq_n_s16( 2 );
  int16x8_t       signUp[2 * ( MAX_CU_SIZE / 16 )];   // sign( above - src ), two vectors per 16 samples
  for( int x = 0; x < endX; x += 16 )
  {
    const int16x8_t s0 = vld1q_s16( srcLine + x );
    const int16x8_t s1 = vld1q_s16( srcLine + x + 8 );
    const int16x8_t a0 = vld1q_s16( srcLine + x - srcStride );
    const int16x8_t a1 = vld1q_s16( srcLine + x + 8 - srcStride );
    signUp[x / 8]     = saoSign_neon( a0, s0 );
    signUp[x / 8 + 1] = saoSign_neon( a1, s1 );
  }
  SaoEoSums sums;
  for( int y = startY; y < endY; y++, srcLine += srcStride, orgLine += orgStride )
  {
    for( int x = 0; x < endX; x += 16 )
    {
      const int16x8_t s0    = vld1q_s16( srcLine + x );
      const int16x8_t s1    = vld1q_s16( srcLine + x + 8 );
      const int16x8_t b0    = vld1q_s16( srcLine + x + srcStride );
      const int16x8_t b1    = vld1q_s16( srcLine + x + 8 + srcStride );
      const int16x8_t o0    = vld1q_s16( orgLine + x );
      const int16x8_t o1    = vld1q_s16( orgLine + x + 8 );
      const int16x8_t down0 = saoSign_neon( s0, b0 );
      const int16x8_t down1 = saoSign_neon( s1, b1 );
      const int16x8_t t0    = vaddq_s16( vsubq_s16( down0, signUp[x / 8] ), two );
      int16x8_t       t1    = vaddq_s16( vsubq_s16( down1, signUp[x / 8 + 1] ), two );
      signUp[x / 8]         = down0;
      signUp[x / 8 + 1]     = down1;
      if( x >= width - 16 )
      {
        t1 = vorrq_s16( t1, maskLast );
      }
      sums.add( t0, t1, vsubq_s16( o0, s0 ), vsubq_s16( o1, s1 ) );
    }
  }
  sums.store( count, diff, false );
}

template<>
void SampleAdaptiveOffset::_initSampleAdaptiveOffsetARM<NEON>()
{
  calcSaoStatisticsBo     = calcSaoStatisticsBo_neon;
  calcSaoStatisticsEo0    = calcSaoStatisticsEo0_neon;
  calcSaoStatisticsEo90   = calcSaoStatisticsEo90_neon;
  calcSaoStatisticsEo135  = calcSaoStatisticsEo135_neon;
  calcSaoStatisticsEo45   = calcSaoStatisticsEo45_neon;
}

#endif // ENABLE_SIMD_OPT_SAO && defined( TARGET_SIMD_ARM )

} // namespace vvenc
