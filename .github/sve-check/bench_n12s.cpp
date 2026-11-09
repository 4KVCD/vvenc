// N2 audit of N12 (NEON SAO band-offset statistics): master's calcSaoStatisticsBo_neon against N12's, both pasted
// verbatim from the trees, on 128x128 10-bit regions of two synthetic 1080p pictures: "smooth" (sinusoids plus small
// noise, so neighbouring samples often share a band, as in real video) and "random" (uniform 10-bit). Statistics
// must be equal; best of 5 per-call times. Self-contained (no library).
#include <arm_neon.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include <algorithm>
typedef int16_t Pel;
#define NUM_SAO_BO_CLASSES_LOG2 5
#define NUM_SAO_BO_CLASSES ( 1 << NUM_SAO_BO_CLASSES_LOG2 )

void boMaster( int width, int endX, int endY, Pel* srcLine, Pel* orgLine, int srcStride, int orgStride,
                               int channelBitDepth, int64_t* count, int64_t* diff )
{
  const int shift = channelBitDepth - NUM_SAO_BO_CLASSES_LOG2;

  int32_t countTmp[2][32] = { { 0 }, { 0 } };
  int32_t diffTmp[2][32] = { { 0 }, { 0 } };

  do
  {
    for( int x = 0; x < endX; ++x )
    {
      int idx = x & 1;
      int bandIdx = srcLine[x] >> shift;
      diffTmp[idx][bandIdx] += orgLine[x] - srcLine[x];
      countTmp[idx][bandIdx]++;
    }

    srcLine += srcStride;
    orgLine += orgStride;
  } while( --endY != 0 );

  for( int i = 0; i < 32; i += 4 )
  {
    int32x4_t c0 = vld1q_s32( countTmp[0] + i );
    int32x4_t c1 = vld1q_s32( countTmp[1] + i );
    int32x4_t d0 = vld1q_s32( diffTmp[0] + i );
    int32x4_t d1 = vld1q_s32( diffTmp[1] + i );

    int64x2_t cTotal_lo = vaddl_s32( vget_low_s32( c0 ), vget_low_s32( c1 ) );
    int64x2_t cTotal_hi = vaddl_s32( vget_high_s32( c0 ), vget_high_s32( c1 ) );
    int64x2_t dTotal_lo = vaddl_s32( vget_low_s32( d0 ), vget_low_s32( d1 ) );
    int64x2_t dTotal_hi = vaddl_s32( vget_high_s32( d0 ), vget_high_s32( d1 ) );

    vst1q_s64( count + i + 0, cTotal_lo );
    vst1q_s64( count + i + 2, cTotal_hi );
    vst1q_s64( diff + i + 0, dTotal_lo );
    vst1q_s64( diff + i + 2, dTotal_hi );
  }
}


// One sample of the band-offset statistics in a packed table entry: the entry holds count * 2^32 + diffSum.
static inline void addToBand_boNew( int64_t* table, int16_t band, int16_t diff )
{
  table[band] += ( int64_t( 1 ) << 32 ) + diff;
}

void boNew( int width, int endX, int endY, Pel* srcLine, Pel* orgLine, int srcStride, int orgStride,
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

      addToBand_boNew( packed[0], vgetq_lane_s16( band, 0 ), vgetq_lane_s16( d, 0 ) );
      addToBand_boNew( packed[1], vgetq_lane_s16( band, 1 ), vgetq_lane_s16( d, 1 ) );
      addToBand_boNew( packed[2], vgetq_lane_s16( band, 2 ), vgetq_lane_s16( d, 2 ) );
      addToBand_boNew( packed[3], vgetq_lane_s16( band, 3 ), vgetq_lane_s16( d, 3 ) );
      addToBand_boNew( packed[4], vgetq_lane_s16( band, 4 ), vgetq_lane_s16( d, 4 ) );
      addToBand_boNew( packed[5], vgetq_lane_s16( band, 5 ), vgetq_lane_s16( d, 5 ) );
      addToBand_boNew( packed[6], vgetq_lane_s16( band, 6 ), vgetq_lane_s16( d, 6 ) );
      addToBand_boNew( packed[7], vgetq_lane_s16( band, 7 ), vgetq_lane_s16( d, 7 ) );
    }
    for( ; x < endX; ++x )
    {
      addToBand_boNew( packed[0], int16_t( srcLine[x] >> shift ), int16_t( orgLine[x] - srcLine[x] ) );
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


static volatile int64_t sink;
int main()
{
  const int W = 1920, H = 1080;
  bool allOk = true;
  for( int kind = 0; kind < 2; kind++ )
  {
    std::mt19937 rng( 3 );
    std::vector<Pel> rec( W * H ), org( W * H );
    for( int y = 0; y < H; y++ )
      for( int x = 0; x < W; x++ )
      {
        int v;
        if( kind == 0 )
          v = int( 512 + 300 * std::sin( x / 37.0 ) * std::cos( y / 23.0 ) + 100 * std::sin( ( x + y ) / 11.0 ) ) + int( rng() % 17 ) - 8;
        else
          v = int( rng() & 1023 );
        rec[y * W + x] = Pel( std::min( 1023, std::max( 0, v ) ) );
        org[y * W + x] = Pel( std::min( 1023, std::max( 0, rec[y * W + x] + int( rng() % 9 ) - 4 ) ) );
      }
    bool ok = true;
    for( int e : { 128, 125, 123, 64, 61 } )
    {
      int64_t c0[32], d0[32], c1[32], d1[32];
      boMaster( 128, e, 120, rec.data() + 7 * W + 33, org.data() + 7 * W + 33, W, W, 10, c0, d0 );
      boNew( 128, e, 120, rec.data() + 7 * W + 33, org.data() + 7 * W + 33, W, W, 10, c1, d1 );
      ok = ok && !memcmp( c0, c1, sizeof( c0 ) ) && !memcmp( d0, d1, sizeof( d0 ) );
    }
    auto run = [&]( auto fn, int k )
    {
      int64_t cnt[32], dif[32];
      auto t0 = std::chrono::steady_clock::now();
      for( int i = 0; i < k; i++ )
      {
        const int bx = ( i * 128 ) % ( W - 128 ), by = ( ( i / 14 ) * 128 ) % ( H - 128 );
        fn( 128, 128, 128, rec.data() + by * W + bx, org.data() + by * W + bx, W, W, 10, cnt, dif );
        sink = sink + cnt[5] + dif[7];
      }
      return std::chrono::duration<double, std::micro>( std::chrono::steady_clock::now() - t0 ).count() / k;
    };
    run( boMaster, 2000 ); run( boNew, 2000 );
    double bm = 1e30, bn = 1e30;
    for( int r = 0; r < 5; r++ ) { bm = std::min( bm, run( boMaster, 20000 ) ); bn = std::min( bn, run( boNew, 20000 ) ); }
    printf( "%-8s equal %s  master %6.2f us  new %6.2f us  %6.1f %% faster\n", kind ? "random" : "smooth", ok ? "yes" : "NO", bm, bn, ( bm / bn - 1 ) * 100 );
    allOk = allOk && ok;
  }
  return allOk ? 0 : 1;
}
