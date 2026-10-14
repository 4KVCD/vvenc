// vvprof: a small sampling profiler for Windows.
//
// It starts a command line as a child process and, at a fixed interval, suspends every thread
// of that child, records its call stack and resumes it. When the child exits it prints:
//   - self time per function (the innermost inlined function at the sampled instruction),
//   - inclusive time per function (every function on the stack, counted once per sample),
//   - the callers of C runtime memory functions (memcpy, memmove, memset),
//   - self time per source line.
// Symbols come from the PDB files next to the executables, read through DbgHelp.
// No administrator rights or kernel tracing are needed, which is why it exists: the PC this work
// was done on had no other profiler that works without elevation.
//
// Build: cl /O2 /EHsc vvprof.cpp
// Usage: vvprof.exe [--interval-ms N] [--depth N] [--top N] [--lines N] [--report FILE] [--stacks FILE] [--affinity HEXMASK] -- command ...

#include <windows.h>
#include <tlhelp32.h>
#include <psapi.h>
#include <dbghelp.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#pragma comment( lib, "dbghelp.lib" )
#pragma comment( lib, "psapi.lib" )

struct ModuleInfo
{
  std::string path;
  std::string name;  // lower-case file name
  uint64_t    base = 0;
  uint64_t    size = 0;
};

struct Options
{
  int         intervalMs   = 1;
  uint64_t    affinity     = 0;  // 0 keeps the default affinity
  int         maxDepth     = 48;
  int         topFunctions = 80;
  int         topLines     = 60;
  std::string reportPath;
  std::string stacksPath;  // folded call stacks for offline analysis
  std::string commandLine;
};

// One symbolized frame: the innermost function at an address and its source line.
struct Frame
{
  std::string module;
  std::string function;
  std::string line;
};

using Stack = std::vector<uint64_t>;  // return addresses, leaf first

struct StackHash
{
  size_t operator()( const Stack& s ) const
  {
    size_t h = 1469598103934665603ull;
    for( uint64_t a : s ) h = ( h ^ a ) * 1099511628211ull;
    return h;
  }
};

static std::string quoteArgument( const std::string& arg )
{
  if( !arg.empty() && arg.find_first_of( " \t\"" ) == std::string::npos )
  {
    return arg;
  }
  std::string quoted = "\"";
  for( char c : arg )
  {
    if( c == '"' ) quoted += '\\';
    quoted += c;
  }
  quoted += '"';
  return quoted;
}

static bool parseOptions( int argc, char** argv, Options& opt )
{
  int i = 1;
  for( ; i < argc; i++ )
  {
    const std::string a = argv[i];
    if( a == "--" ) { i++; break; }
    if( i + 1 >= argc ) return false;
    if     ( a == "--interval-ms" ) opt.intervalMs   = atoi( argv[++i] );
    else if( a == "--depth"       ) opt.maxDepth     = atoi( argv[++i] );
    else if( a == "--top"         ) opt.topFunctions = atoi( argv[++i] );
    else if( a == "--lines"       ) opt.topLines     = atoi( argv[++i] );
    else if( a == "--report"      ) opt.reportPath   = argv[++i];
    else if( a == "--stacks"      ) opt.stacksPath   = argv[++i];
    else if( a == "--affinity"    ) opt.affinity     = strtoull( argv[++i], nullptr, 16 );
    else return false;
  }
  for( ; i < argc; i++ )
  {
    if( !opt.commandLine.empty() ) opt.commandLine += ' ';
    opt.commandLine += quoteArgument( argv[i] );
  }
  return !opt.commandLine.empty();
}

static std::string lowerBaseName( const std::string& path )
{
  const size_t slash = path.find_last_of( "\\/" );
  std::string  name  = slash == std::string::npos ? path : path.substr( slash + 1 );
  std::transform( name.begin(), name.end(), name.begin(), ::tolower );
  return name;
}

static bool isWaitModule( const std::string& name )
{
  return name == "ntdll.dll" || name == "kernelbase.dll" || name == "kernel32.dll";
}

static bool isCrtMemoryModule( const std::string& name )
{
  return name == "vcruntime140.dll" || name == "ucrtbase.dll";
}

// Memory reads during a stack walk are served from a snapshot of the sampled thread's stack and
// from cached pages of module images, so that one sample costs one cross-process read instead of
// dozens. DbgHelp's read callback has no user pointer, so this state is global; the profiler
// only walks stacks from its main thread.
struct WalkMemory
{
  HANDLE                                             process = nullptr;
  uint64_t                                           stackLo = 0;
  std::vector<uint8_t>                               stack;
  std::unordered_map<uint64_t, std::vector<uint8_t>> pages;  // module image pages by address
};
static WalkMemory g_walkMemory;

static const std::vector<uint8_t>* cachedPage( uint64_t pageAddress )
{
  auto found = g_walkMemory.pages.find( pageAddress );
  if( found != g_walkMemory.pages.end() ) return found->second.empty() ? nullptr : &found->second;

  std::vector<uint8_t> page( 4096 );
  SIZE_T read = 0;
  if( !ReadProcessMemory( g_walkMemory.process, ( LPCVOID ) pageAddress, page.data(), page.size(), &read ) || read != page.size() )
  {
    page.clear();  // remember the failure so the page is not read again
  }
  auto& stored = g_walkMemory.pages[pageAddress] = std::move( page );
  return stored.empty() ? nullptr : &stored;
}

static BOOL CALLBACK readWalkMemory( HANDLE process, DWORD64 address, PVOID buffer, DWORD size, LPDWORD bytesRead )
{
  if( address >= g_walkMemory.stackLo && address + size <= g_walkMemory.stackLo + g_walkMemory.stack.size() )
  {
    memcpy( buffer, g_walkMemory.stack.data() + ( address - g_walkMemory.stackLo ), size );
    *bytesRead = size;
    return TRUE;
  }

  // Anything else that a walk reads is unwind data or code inside module images, which never change.
  uint8_t* out = static_cast<uint8_t*>( buffer );
  DWORD    done = 0;
  while( done < size )
  {
    const uint64_t current  = address + done;
    const uint64_t pageAddr = current & ~4095ull;
    const std::vector<uint8_t>* page = cachedPage( pageAddr );
    if( !page ) break;
    const DWORD chunk = ( DWORD ) std::min<uint64_t>( size - done, pageAddr + 4096 - current );
    memcpy( out + done, page->data() + ( current - pageAddr ), chunk );
    done += chunk;
  }
  if( done == size )
  {
    *bytesRead = size;
    return TRUE;
  }

  SIZE_T read = 0;
  const BOOL ok = ReadProcessMemory( process, ( LPCVOID ) address, buffer, size, &read );
  *bytesRead = ( DWORD ) read;
  return ok;
}

// Copies the used part of the thread's stack, from the stack pointer up to the stack base.
static void snapshotStack( HANDLE process, uint64_t rsp, uint64_t stackBase )
{
  g_walkMemory.stackLo = rsp;
  const uint64_t size  = stackBase > rsp ? std::min<uint64_t>( stackBase - rsp, 1 << 20 ) : 0;
  g_walkMemory.stack.resize( size );
  SIZE_T read = 0;
  if( !size || !ReadProcessMemory( process, ( LPCVOID ) rsp, g_walkMemory.stack.data(), size, &read ) )
  {
    g_walkMemory.stack.clear();
  }
}

struct TrackedThread
{
  HANDLE   handle    = nullptr;
  uint64_t stackBase = 0;
};

// The stack base is stored in the thread environment block, which NtQueryInformationThread locates.
static uint64_t queryStackBase( HANDLE process, HANDLE thread )
{
  struct ThreadBasicInformation
  {
    LONG      exitStatus;
    PVOID     tebBaseAddress;
    ULONG_PTR clientId[2];
    ULONG_PTR affinityMask;
    LONG      priority;
    LONG      basePriority;
  };
  using QueryFn = LONG( WINAPI* )( HANDLE, int, PVOID, ULONG, PULONG );
  static QueryFn query = ( QueryFn ) GetProcAddress( GetModuleHandleA( "ntdll.dll" ), "NtQueryInformationThread" );

  ThreadBasicInformation info = {};
  if( !query || query( thread, 0, &info, sizeof( info ), nullptr ) != 0 ) return 0;
  NT_TIB tib = {};
  SIZE_T read = 0;
  if( !ReadProcessMemory( process, info.tebBaseAddress, &tib, sizeof( tib ), &read ) ) return 0;
  return ( uint64_t ) tib.StackBase;
}

// Opens handles to all threads of the process that are not tracked yet.
static void refreshThreads( HANDLE process, DWORD pid, std::unordered_map<DWORD, TrackedThread>& threads )
{
  HANDLE snapshot = CreateToolhelp32Snapshot( TH32CS_SNAPTHREAD, 0 );
  if( snapshot == INVALID_HANDLE_VALUE ) return;

  THREADENTRY32 entry = { sizeof( entry ) };
  for( BOOL ok = Thread32First( snapshot, &entry ); ok; ok = Thread32Next( snapshot, &entry ) )
  {
    if( entry.th32OwnerProcessID != pid || threads.count( entry.th32ThreadID ) ) continue;
    HANDLE h = OpenThread( THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE, entry.th32ThreadID );
    if( h ) threads[entry.th32ThreadID] = { h, queryStackBase( process, h ) };
  }
  CloseHandle( snapshot );
}

class Profiler
{
public:
  explicit Profiler( const Options& opt ) : m_opt( opt ) {}

  // Samples the child until it exits.
  void run( const PROCESS_INFORMATION& pi )
  {
    m_process = pi.hProcess;
    g_walkMemory.process = pi.hProcess;
    SymSetOptions( SYMOPT_UNDNAME | SYMOPT_LOAD_LINES | SYMOPT_FAIL_CRITICAL_ERRORS | SYMOPT_DEFERRED_LOADS );
    SymInitialize( m_process, nullptr, FALSE );

    std::unordered_map<DWORD, TrackedThread> threads;
    DWORD lastThreadRefresh = 0;
    DWORD lastModuleRefresh = 0;

    // Windows 11 ignores timeBeginPeriod for processes without a visible window, so Sleep(1)
    // would last about 15 ms. A high-resolution waitable timer keeps the requested interval.
    HANDLE timer = CreateWaitableTimerExW( nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS );
    LARGE_INTEGER due;
    due.QuadPart = -10000LL * m_opt.intervalMs;

    while( WaitForSingleObject( m_process, 0 ) == WAIT_TIMEOUT )
    {
      // A thread snapshot lists every thread on the system and costs milliseconds, so it is
      // refreshed rarely. Encoder threads are created at startup and live until the end.
      const DWORD now = GetTickCount();
      if( now - lastThreadRefresh >= 250 )
      {
        refreshThreads( m_process, pi.dwProcessId, threads );
        lastThreadRefresh = now;
      }
      if( now - lastModuleRefresh >= 1000 )
      {
        refreshModules();
        lastModuleRefresh = now;
      }

      for( auto it = threads.begin(); it != threads.end(); )
      {
        if( !sampleThread( it->second ) )
        {
          CloseHandle( it->second.handle );
          it = threads.erase( it );
          continue;
        }
        ++it;
      }

      if( timer && SetWaitableTimer( timer, &due, 0, nullptr, nullptr, FALSE ) )
      {
        WaitForSingleObject( timer, INFINITE );
      }
      else
      {
        Sleep( m_opt.intervalMs );
      }
    }

    if( timer ) CloseHandle( timer );
    for( auto& t : threads ) CloseHandle( t.second.handle );
  }

  void report( FILE* out );
  void writeFoldedStacks( FILE* out );

private:
  void refreshModules()
  {
    HMODULE handles[1024];
    DWORD   bytesNeeded = 0;
    if( !EnumProcessModulesEx( m_process, handles, sizeof( handles ), &bytesNeeded, LIST_MODULES_ALL ) ) return;

    const DWORD count = std::min<DWORD>( bytesNeeded / sizeof( HMODULE ), 1024 );
    for( DWORD i = 0; i < count; i++ )
    {
      MODULEINFO info = {};
      char       path[MAX_PATH * 4] = {};
      if( !GetModuleInformation( m_process, handles[i], &info, sizeof( info ) ) ) continue;
      if( !GetModuleFileNameExA( m_process, handles[i], path, sizeof( path ) ) ) continue;
      const uint64_t base = ( uint64_t ) info.lpBaseOfDll;
      if( m_moduleBases.count( base ) ) continue;
      m_moduleBases.insert( base );
      m_modules.push_back( { path, lowerBaseName( path ), base, info.SizeOfImage } );
      SymLoadModuleEx( m_process, nullptr, path, nullptr, base, info.SizeOfImage, nullptr, 0 );
    }
  }

  const ModuleInfo* findModule( uint64_t address ) const
  {
    for( const ModuleInfo& m : m_modules )
    {
      if( address >= m.base && address < m.base + m.size ) return &m;
    }
    return nullptr;
  }

  // Suspends one thread, records its stack and resumes it. Returns false if the thread is gone.
  bool sampleThread( const TrackedThread& tracked )
  {
    const HANDLE thread = tracked.handle;
    if( SuspendThread( thread ) == ( DWORD ) -1 ) return false;

    alignas( 16 ) CONTEXT context = {};
    context.ContextFlags = CONTEXT_FULL;
    if( GetThreadContext( thread, &context ) )
    {
      Stack stack;
      stack.push_back( context.Rip );

      // Waiting threads are recorded without their stack: it would only cost time.
      const ModuleInfo* leafModule = findModule( context.Rip );
      if( !leafModule || !isWaitModule( leafModule->name ) )
      {
        snapshotStack( m_process, context.Rsp, tracked.stackBase );
        STACKFRAME64 frame = {};
        frame.AddrPC.Offset    = context.Rip;
        frame.AddrPC.Mode      = AddrModeFlat;
        frame.AddrStack.Offset = context.Rsp;
        frame.AddrStack.Mode   = AddrModeFlat;
        frame.AddrFrame.Offset = context.Rbp;
        frame.AddrFrame.Mode   = AddrModeFlat;
        // The first call returns the leaf frame itself, which is already recorded.
        for( int depth = 0; depth < m_opt.maxDepth; depth++ )
        {
          if( !StackWalk64( IMAGE_FILE_MACHINE_AMD64, m_process, thread, &frame, &context, readWalkMemory,
                            SymFunctionTableAccess64, SymGetModuleBase64, nullptr ) ) break;
          if( depth > 0 )
          {
            if( frame.AddrPC.Offset == 0 ) break;
            stack.push_back( frame.AddrPC.Offset );
          }
        }
      }
      m_stacks[stack]++;
    }
    ResumeThread( thread );
    return true;
  }

  // Symbolizes an address into its inline frames, innermost first. Return addresses point after
  // the call instruction, so callers are looked up one byte earlier.
  const std::vector<Frame>& symbolize( uint64_t address, bool isReturnAddress )
  {
    const uint64_t key = address * 2 + ( isReturnAddress ? 1 : 0 );
    auto found = m_symbolCache.find( key );
    if( found != m_symbolCache.end() ) return found->second;

    std::vector<Frame>& frames = m_symbolCache[key];
    const ModuleInfo* module     = findModule( address );
    const std::string moduleName = module ? module->name : "?";
    const uint64_t    lookup     = isReturnAddress ? address - 1 : address;

    alignas( SYMBOL_INFO ) char storage[sizeof( SYMBOL_INFO ) + 1024];
    SYMBOL_INFO*    symbol   = reinterpret_cast<SYMBOL_INFO*>( storage );
    DWORD64         offset   = 0;
    DWORD           lineOff  = 0;
    IMAGEHLP_LINE64 lineInfo = { sizeof( lineInfo ) };

    const DWORD inlineDepth   = SymAddrIncludeInlineTrace( m_process, lookup );
    DWORD       inlineContext = 0, frameIndex = 0;
    if( inlineDepth > 0 && SymQueryInlineTrace( m_process, lookup, 0, lookup, lookup, &inlineContext, &frameIndex ) )
    {
      for( DWORD i = 0; i < inlineDepth; i++, inlineContext++ )
      {
        memset( storage, 0, sizeof( storage ) );
        symbol->SizeOfStruct = sizeof( SYMBOL_INFO );
        symbol->MaxNameLen   = 1000;
        Frame f;
        f.module   = moduleName;
        f.function = SymFromInlineContext( m_process, lookup, inlineContext, &offset, symbol ) ? symbol->Name : "?";
        f.line     = SymGetLineFromInlineContext( m_process, lookup, inlineContext, 0, &lineOff, &lineInfo )
                     ? lowerBaseName( lineInfo.FileName ) + ":" + std::to_string( lineInfo.LineNumber ) : "?";
        frames.push_back( f );
      }
    }

    memset( storage, 0, sizeof( storage ) );
    symbol->SizeOfStruct = sizeof( SYMBOL_INFO );
    symbol->MaxNameLen   = 1000;
    Frame physical;
    physical.module   = moduleName;
    physical.function = SymFromAddr( m_process, lookup, &offset, symbol ) ? symbol->Name : "?";
    physical.line     = SymGetLineFromAddr64( m_process, lookup, &lineOff, &lineInfo )
                        ? lowerBaseName( lineInfo.FileName ) + ":" + std::to_string( lineInfo.LineNumber ) : "?";
    frames.push_back( physical );
    return frames;
  }

  const Options&                                    m_opt;
  HANDLE                                            m_process = nullptr;
  std::vector<ModuleInfo>                           m_modules;
  std::set<uint64_t>                                m_moduleBases;
  std::unordered_map<Stack, uint64_t, StackHash>    m_stacks;
  std::unordered_map<uint64_t, std::vector<Frame>>  m_symbolCache;
};

static void printTop( FILE* out, const char* title, const std::map<std::string, uint64_t>& counts, uint64_t total, int limit )
{
  std::vector<std::pair<std::string, uint64_t>> sorted( counts.begin(), counts.end() );
  std::sort( sorted.begin(), sorted.end(), []( const auto& a, const auto& b ) { return a.second > b.second; } );
  fprintf( out, "\n== %s\n", title );
  for( int i = 0; i < limit && i < ( int ) sorted.size(); i++ )
  {
    const double share = 100.0 * sorted[i].second / std::max<uint64_t>( total, 1 );
    fprintf( out, "%6.2f%%  %8llu  %s\n", share, ( unsigned long long ) sorted[i].second, sorted[i].first.c_str() );
  }
}

void Profiler::report( FILE* out )
{
  std::map<std::string, uint64_t> byModule, selfFunction, inclusive, selfLine, memoryCallers, memoryCallerLines;
  uint64_t total = 0, busy = 0, memory = 0;

  for( const auto& entry : m_stacks )
  {
    const Stack&      stack = entry.first;
    const uint64_t    count = entry.second;
    const ModuleInfo* leaf  = findModule( stack[0] );
    const std::string leafModule = leaf ? leaf->name : "?";
    byModule[leafModule] += count;
    total += count;
    if( isWaitModule( leafModule ) ) continue;
    busy += count;

    const std::vector<Frame>& leafFrames = symbolize( stack[0], false );
    selfFunction[leafFrames.front().module + "!" + leafFrames.front().function] += count;
    selfLine[leafFrames.front().line + "  (" + leafFrames.front().function + ")"] += count;

    std::set<std::string> seen;
    bool   memoryCallerFound = !isCrtMemoryModule( leafModule );
    if( !memoryCallerFound ) memory += count;
    for( size_t i = 0; i < stack.size(); i++ )
    {
      for( const Frame& f : symbolize( stack[i], i > 0 ) )
      {
        const std::string name = f.module + "!" + f.function;
        if( seen.insert( name ).second ) inclusive[name] += count;
        if( !memoryCallerFound && !isCrtMemoryModule( f.module ) )
        {
          memoryCallers[f.function] += count;
          memoryCallerLines[f.line + "  (" + f.function + ")"] += count;
          memoryCallerFound = true;
        }
      }
    }
  }

  fprintf( out, "samples: %llu total, %llu busy, %llu in C runtime memory functions\n",
           ( unsigned long long ) total, ( unsigned long long ) busy, ( unsigned long long ) memory );
  printTop( out, "modules (share of all samples)", byModule, total, 20 );
  printTop( out, "self time by function after inlining (share of busy samples)", selfFunction, busy, m_opt.topFunctions );
  printTop( out, "inclusive time by function (share of busy samples)", inclusive, busy, m_opt.topFunctions * 2 );
  printTop( out, "callers of C runtime memory functions (share of busy samples)", memoryCallers, busy, 40 );
  printTop( out, "caller lines of C runtime memory functions (share of busy samples)", memoryCallerLines, busy, 40 );
  printTop( out, "self time by source line (share of busy samples)", selfLine, busy, m_opt.topLines );
}

// Writes one line per unique busy stack in the folded format "root;...;leaf count", where each
// frame is "function@file:line" with inlined functions expanded. Waiting threads are skipped.
void Profiler::writeFoldedStacks( FILE* out )
{
  for( const auto& entry : m_stacks )
  {
    const Stack&      stack = entry.first;
    const ModuleInfo* leaf  = findModule( stack[0] );
    if( leaf && isWaitModule( leaf->name ) ) continue;

    std::vector<std::string> frames;  // leaf first
    for( size_t i = 0; i < stack.size(); i++ )
    {
      for( const Frame& f : symbolize( stack[i], i > 0 ) )
      {
        frames.push_back( f.function + "@" + f.line );
      }
    }
    std::string line;
    for( auto it = frames.rbegin(); it != frames.rend(); ++it )
    {
      if( !line.empty() ) line += ';';
      for( char c : *it ) line += ( c == ';' || c == ' ' ) ? '_' : c;
    }
    fprintf( out, "%s %llu\n", line.c_str(), ( unsigned long long ) entry.second );
  }
}

int main( int argc, char** argv )
{
  Options opt;
  if( !parseOptions( argc, argv, opt ) )
  {
    fprintf( stderr, "usage: vvprof [--interval-ms N] [--depth N] [--top N] [--lines N] [--report FILE] [--stacks FILE] [--affinity HEXMASK] -- command ...\n" );
    return 2;
  }

  STARTUPINFOA        si = { sizeof( si ) };
  PROCESS_INFORMATION pi = {};
  std::vector<char>   cmd( opt.commandLine.begin(), opt.commandLine.end() );
  cmd.push_back( 0 );
  // The child starts suspended so that its affinity applies before its first instruction.
  if( !CreateProcessA( nullptr, cmd.data(), nullptr, nullptr, TRUE, CREATE_SUSPENDED, nullptr, nullptr, &si, &pi ) )
  {
    fprintf( stderr, "vvprof: cannot start: %s (error %lu)\n", opt.commandLine.c_str(), GetLastError() );
    return 1;
  }

  if( opt.affinity ) SetProcessAffinityMask( pi.hProcess, ( DWORD_PTR ) opt.affinity );
  ResumeThread( pi.hThread );

  Profiler profiler( opt );
  profiler.run( pi );

  DWORD exitCode = 0;
  GetExitCodeProcess( pi.hProcess, &exitCode );

  FILE* out = stdout;
  if( !opt.reportPath.empty() )
  {
    out = fopen( opt.reportPath.c_str(), "w" );
    if( !out ) out = stdout;
  }
  fprintf( out, "command: %s\nexit code: %lu\n", opt.commandLine.c_str(), exitCode );
  profiler.report( out );
  if( out != stdout ) fclose( out );

  if( !opt.stacksPath.empty() )
  {
    if( FILE* stacks = fopen( opt.stacksPath.c_str(), "w" ) )
    {
      profiler.writeFoldedStacks( stacks );
      fclose( stacks );
    }
  }

  SymCleanup( pi.hProcess );
  CloseHandle( pi.hThread );
  CloseHandle( pi.hProcess );
  return ( int ) exitCode;
}
