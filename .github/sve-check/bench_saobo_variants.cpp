// #919 review (georges-arm, 2026-11-10): variants of calcSaoStatisticsBo_neon's inner loop.
//   vec8    the PR's kernel: band and diff in vectors, 16 vgetq_lane_s16, 8 packed tables
//   vec4    same, 4 tables (lanes k and k + 4 share table k)
//   u64x8   band and diff in vectors, read as 4 u64 lanes, entries taken with shifts, 8 tables
//   u64x4   same, 4 tables
//   scal8   band and diff computed per sample in scalar code, 8 tables
//   scal4   same, 4 tables
//   scal2   same, 2 tables
// All against master's kernel (2 interleaved int32 tables), outputs compared on several endX.
// Inputs: 128x128 10-bit regions of a smooth and a random synthetic 1080p picture, and, when a y4m path is given,
// its first luma picture (8-bit << 2, org = rec + noise within 4). Self-contained: no library.
#include <arm_neon.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
typedef int16_t Pel;
#define NUM_SAO_BO_CLASSES_LOG2 5
#define NUM_SAO_BO_CLASSES ( 1 << NUM_SAO_BO_CLASSES_LOG2 )
#define NOINLINE __attribute__( ( noinline ) )

typedef void ( *BoFn )( int, int, int, Pel*, Pel*, int, int, int, int64_t*, int64_t* );

NOINLINE void boMaster( int width, int endX, int endY, Pel* srcLine, Pel* orgLine, int srcStride, int orgStride,
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
  for( int i = 0; i < 32; i++ )
  {
    count[i] = countTmp[0][i] + countTmp[1][i];
    diff[i] = diffTmp[0][i] + diffTmp[1][i];
  }
}

static inline void addToBand( int64_t* table, int band, int d )
{
  table[band] += ( int64_t( 1 ) << 32 ) + d;
}

template<int T>
static inline void mergeTables( const int64_t ( *packed )[NUM_SAO_BO_CLASSES], int64_t* count, int64_t* diff )
{
  for( int i = 0; i < NUM_SAO_BO_CLASSES; i += 2 )
  {
    int64x2_t sum = vld1q_s64( packed[0] + i );
    for( int k = 1; k < T; k++ )
    {
      sum = vaddq_s64( sum, vld1q_s64( packed[k] + i ) );
    }
    const int64x2_t diffSum = vmovl_s32( vmovn_s64( sum ) );
    vst1q_s64( diff + i, diffSum );
    vst1q_s64( count + i, vshrq_n_s64( vsubq_s64( sum, diffSum ), 32 ) );
  }
}

template<int T>
NOINLINE void boVec( int, int endX, int endY, Pel* srcLine, Pel* orgLine, int srcStride, int orgStride, int bd,
                     int64_t* count, int64_t* diff )
{
  const int       shift    = bd - NUM_SAO_BO_CLASSES_LOG2;
  const int16x8_t negShift = vdupq_n_s16( -shift );
  int64_t         packed[T][NUM_SAO_BO_CLASSES] = {};
  do
  {
    int x = 0;
    for( ; x + 8 <= endX; x += 8 )
    {
      const int16x8_t src  = vld1q_s16( srcLine + x );
      const int16x8_t org  = vld1q_s16( orgLine + x );
      const int16x8_t band = vshlq_s16( src, negShift );
      const int16x8_t d    = vsubq_s16( org, src );
      addToBand( packed[0 % T], vgetq_lane_s16( band, 0 ), vgetq_lane_s16( d, 0 ) );
      addToBand( packed[1 % T], vgetq_lane_s16( band, 1 ), vgetq_lane_s16( d, 1 ) );
      addToBand( packed[2 % T], vgetq_lane_s16( band, 2 ), vgetq_lane_s16( d, 2 ) );
      addToBand( packed[3 % T], vgetq_lane_s16( band, 3 ), vgetq_lane_s16( d, 3 ) );
      addToBand( packed[4 % T], vgetq_lane_s16( band, 4 ), vgetq_lane_s16( d, 4 ) );
      addToBand( packed[5 % T], vgetq_lane_s16( band, 5 ), vgetq_lane_s16( d, 5 ) );
      addToBand( packed[6 % T], vgetq_lane_s16( band, 6 ), vgetq_lane_s16( d, 6 ) );
      addToBand( packed[7 % T], vgetq_lane_s16( band, 7 ), vgetq_lane_s16( d, 7 ) );
    }
    for( ; x < endX; ++x )
    {
      addToBand( packed[0], srcLine[x] >> shift, orgLine[x] - srcLine[x] );
    }
    srcLine += srcStride;
    orgLine += orgStride;
  } while( --endY != 0 );
  mergeTables<T>( packed, count, diff );
}

// lane j (0..3) of a u64 holding four int16 values
static inline int bandOf( uint64_t v, int j )
{
  return int( ( v >> ( 16 * j ) ) & 0xffff );
}
static inline int diffOf( uint64_t v, int j )
{
  return int16_t( v >> ( 16 * j ) );
}

template<int T>
NOINLINE void boU64( int, int endX, int endY, Pel* srcLine, Pel* orgLine, int srcStride, int orgStride, int bd,
                     int64_t* count, int64_t* diff )
{
  const int       shift    = bd - NUM_SAO_BO_CLASSES_LOG2;
  const int16x8_t negShift = vdupq_n_s16( -shift );
  int64_t         packed[T][NUM_SAO_BO_CLASSES] = {};
  do
  {
    int x = 0;
    for( ; x + 8 <= endX; x += 8 )
    {
      const int16x8_t src  = vld1q_s16( srcLine + x );
      const int16x8_t org  = vld1q_s16( orgLine + x );
      const int16x8_t band = vshlq_s16( src, negShift );
      const int16x8_t d    = vsubq_s16( org, src );
      const uint64_t  b0   = vgetq_lane_u64( vreinterpretq_u64_s16( band ), 0 );
      const uint64_t  b1   = vgetq_lane_u64( vreinterpretq_u64_s16( band ), 1 );
      const uint64_t  d0   = vgetq_lane_u64( vreinterpretq_u64_s16( d ), 0 );
      const uint64_t  d1   = vgetq_lane_u64( vreinterpretq_u64_s16( d ), 1 );
      addToBand( packed[0 % T], bandOf( b0, 0 ), diffOf( d0, 0 ) );
      addToBand( packed[1 % T], bandOf( b0, 1 ), diffOf( d0, 1 ) );
      addToBand( packed[2 % T], bandOf( b0, 2 ), diffOf( d0, 2 ) );
      addToBand( packed[3 % T], bandOf( b0, 3 ), diffOf( d0, 3 ) );
      addToBand( packed[4 % T], bandOf( b1, 0 ), diffOf( d1, 0 ) );
      addToBand( packed[5 % T], bandOf( b1, 1 ), diffOf( d1, 1 ) );
      addToBand( packed[6 % T], bandOf( b1, 2 ), diffOf( d1, 2 ) );
      addToBand( packed[7 % T], bandOf( b1, 3 ), diffOf( d1, 3 ) );
    }
    for( ; x < endX; ++x )
    {
      addToBand( packed[0], srcLine[x] >> shift, orgLine[x] - srcLine[x] );
    }
    srcLine += srcStride;
    orgLine += orgStride;
  } while( --endY != 0 );
  mergeTables<T>( packed, count, diff );
}

template<int T>
NOINLINE void boScalar( int, int endX, int endY, Pel* srcLine, Pel* orgLine, int srcStride, int orgStride, int bd,
                        int64_t* count, int64_t* diff )
{
  const int shift = bd - NUM_SAO_BO_CLASSES_LOG2;
  int64_t   packed[T][NUM_SAO_BO_CLASSES] = {};
  do
  {
    int x = 0;
    for( ; x + 8 <= endX; x += 8 )
    {
      for( int j = 0; j < 8; j++ )
      {
        addToBand( packed[j % T], srcLine[x + j] >> shift, orgLine[x + j] - srcLine[x + j] );
      }
    }
    for( ; x < endX; ++x )
    {
      addToBand( packed[0], srcLine[x] >> shift, orgLine[x] - srcLine[x] );
    }
    srcLine += srcStride;
    orgLine += orgStride;
  } while( --endY != 0 );
  mergeTables<T>( packed, count, diff );
}

static volatile int64_t sink;

int main( int argc, char** argv )
{
  const int W = 1920, H = 1080;
  struct V { const char* name; BoFn fn; };
  const V vars[] = { { "vec8", boVec<8> },   { "vec4", boVec<4> },     { "u64x8", boU64<8> },   { "u64x4", boU64<4> },
                     { "scal8", boScalar<8> }, { "scal4", boScalar<4> }, { "scal2", boScalar<2> } };
  const int nv = sizeof( vars ) / sizeof( vars[0] );
  bool allOk = true;
  for( int kind = 0; kind < ( argc > 1 ? 3 : 2 ); kind++ )
  {
    std::mt19937 rng( 3 );
    std::vector<Pel> rec( W * H ), org( W * H );
    std::vector<uint8_t> y8;
    if( kind == 2 )
    {
      FILE* f = fopen( argv[1], "rb" );
      char line[256];
      y8.resize( W * H );
      if( !f || !fgets( line, sizeof( line ), f ) || !fgets( line, sizeof( line ), f ) ||
          fread( y8.data(), 1, W * H, f ) != size_t( W * H ) )
      {
        printf( "cannot read %s\n", argv[1] );
        return 1;
      }
      fclose( f );
    }
    for( int y = 0; y < H; y++ )
      for( int x = 0; x < W; x++ )
      {
        int v;
        if( kind == 0 )
          v = int( 512 + 300 * std::sin( x / 37.0 ) * std::cos( y / 23.0 ) + 100 * std::sin( ( x + y ) / 11.0 ) ) + int( rng() % 17 ) - 8;
        else if( kind == 1 )
          v = int( rng() & 1023 );
        else
          v = y8[y * W + x] << 2;
        rec[y * W + x] = Pel( std::min( 1023, std::max( 0, v ) ) );
        org[y * W + x] = Pel( std::min( 1023, std::max( 0, rec[y * W + x] + int( rng() % 9 ) - 4 ) ) );
      }
    bool ok = true;
    for( int e : { 128, 125, 123, 64, 61 } )
    {
      int64_t c0[32], d0[32], c1[32], d1[32];
      boMaster( 128, e, 120, rec.data() + 7 * W + 33, org.data() + 7 * W + 33, W, W, 10, c0, d0 );
      for( int v = 0; v < nv; v++ )
      {
        vars[v].fn( 128, e, 120, rec.data() + 7 * W + 33, org.data() + 7 * W + 33, W, W, 10, c1, d1 );
        if( memcmp( c0, c1, sizeof( c0 ) ) || memcmp( d0, d1, sizeof( d0 ) ) )
        {
          printf( "DIFFERENT %s endX %d\n", vars[v].name, e );
          ok = false;
        }
      }
    }
    auto run = [&]( BoFn fn, int k )
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
    double best[16];
    double bm = 1e30;
    for( int v = 0; v < nv; v++ ) { best[v] = 1e30; run( vars[v].fn, 1000 ); }
    run( boMaster, 1000 );
    for( int r = 0; r < 5; r++ )
    {
      bm = std::min( bm, run( boMaster, 10000 ) );
      for( int v = 0; v < nv; v++ )
        best[v] = std::min( best[v], run( vars[v].fn, 10000 ) );
    }
    printf( "%-8s equal %s  master %6.2f us", kind == 0 ? "smooth" : kind == 1 ? "random" : "crowd", ok ? "yes" : "NO", bm );
    for( int v = 0; v < nv; v++ )
      printf( "  %s %5.2f", vars[v].name, best[v] );
    printf( "  (us per 128x128 call, best of 5)\n" );
    allOk = allOk && ok;
  }
  return allOk ? 0 : 1;
}
