// thrash: tools for cache-sensitivity experiments on Windows.
//
//   thrash.exe topo                       prints which logical processors share a core and an L3 cache
//   thrash.exe run MASK MB SECONDS        streams through MB megabytes on every processor in the hex
//                                         affinity MASK for SECONDS seconds, evicting the shared L3
//
// Running the encoder next to "run" on the same L3 shows which encoder functions depend on the
// L3 cache: their share of the profile grows when the cache is constantly evicted.
//
// Build: cl /O2 /EHsc thrash.cpp

#include <windows.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

static void printTopology()
{
  DWORD length = 0;
  GetLogicalProcessorInformationEx( RelationAll, nullptr, &length );
  std::vector<uint8_t> buffer( length );
  auto* info = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>( buffer.data() );
  if( !GetLogicalProcessorInformationEx( RelationAll, info, &length ) ) return;

  for( uint8_t* p = buffer.data(); p < buffer.data() + length; )
  {
    auto* entry = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>( p );
    if( entry->Relationship == RelationProcessorCore )
    {
      printf( "core   mask %016llx\n", ( unsigned long long ) entry->Processor.GroupMask[0].Mask );
    }
    else if( entry->Relationship == RelationCache && entry->Cache.Level >= 2 && entry->Cache.Type != CacheInstruction )
    {
      printf( "L%d %5lu KB mask %016llx\n", entry->Cache.Level, entry->Cache.CacheSize / 1024, ( unsigned long long ) entry->Cache.GroupMask.Mask );
    }
    p += entry->Size;
  }
}

// Each thread walks its own buffer with a 64-byte stride, reading and writing every cache line,
// so the L3 keeps refilling with data the encoder never uses.
static void streamThread( int cpu, size_t bytes, const std::atomic<bool>* stop, std::atomic<uint64_t>* passes )
{
  SetThreadAffinityMask( GetCurrentThread(), 1ull << cpu );
  std::vector<uint8_t> data( bytes, 1 );
  while( !stop->load( std::memory_order_relaxed ) )
  {
    for( size_t i = 0; i < bytes; i += 64 ) data[i] += 1;
    passes->fetch_add( 1, std::memory_order_relaxed );
  }
}

int main( int argc, char** argv )
{
  if( argc >= 2 && std::string( argv[1] ) == "topo" )
  {
    printTopology();
    return 0;
  }
  if( argc < 5 || std::string( argv[1] ) != "run" )
  {
    fprintf( stderr, "usage: thrash topo | thrash run HEXMASK MB SECONDS\n" );
    return 2;
  }

  const uint64_t mask    = strtoull( argv[2], nullptr, 16 );
  const size_t   bytes   = ( size_t ) atoi( argv[3] ) << 20;
  const int      seconds = atoi( argv[4] );

  std::atomic<bool>     stop{ false };
  std::atomic<uint64_t> passes{ 0 };
  std::vector<std::thread> threads;
  for( int cpu = 0; cpu < 64; cpu++ )
  {
    if( mask & ( 1ull << cpu ) ) threads.emplace_back( streamThread, cpu, bytes, &stop, &passes );
  }
  std::this_thread::sleep_for( std::chrono::seconds( seconds ) );
  stop = true;
  for( auto& t : threads ) t.join();
  printf( "threads %zu, passes %llu, %.1f GB/s\n", threads.size(), ( unsigned long long ) passes.load(),
          passes.load() * ( double ) bytes / seconds / 1e9 );
  return 0;
}
