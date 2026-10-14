"""Bit-exactness and speed/memory harness for the VVenC memory-layout work.

Every encode runs vvencapp as a child process on real clips. The harness compares the
bitstream hash of a candidate build with the unmodified baseline build, and measures the
child's wall time, CPU time and peak memory through the Win32 process APIs. It also measures
how much CPU other processes used during each run, so that disturbed timings are visible.

Examples:
  python vvbench.py check --cand C:/path/vvencapp.exe --suite quick
  python vvbench.py time  --cand C:/path/vvencapp.exe --configs bk600_1080p_medium bk600_2160p_medium
"""

import argparse
import ctypes
import ctypes.wintypes as wt
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time

BENCH_DIR = os.environ.get("VVBENCH_DIR", r"C:\code\VVenc\.claude\bench")
BASE_EXE = os.path.join(BENCH_DIR, "bin-base", "vvencapp.exe")
VIDEO_DIR = r"C:\Video encodings"

# Clips cut from the user's real videos (see make_clips in the README of this folder).
# name: (path, width, height, frames available, frame rate)
CLIPS = {
    "bk600_2160p": (os.path.join(BENCH_DIR, "in", "bk600_3840x2160.yuv"), 3840, 2160, 65, "24000/1001"),
    "bk600_1080p": (os.path.join(BENCH_DIR, "in", "bk600_1920x1080.yuv"), 1920, 1080, 65, "24000/1001"),
    "bk1500_1608p": (os.path.join(BENCH_DIR, "in", "bk1500_3840x1608.yuv"), 3840, 1608, 33, "24000/1001"),
    "bk1500_808p": (os.path.join(BENCH_DIR, "in", "bk1500_1920x808.yuv"), 1920, 808, 33, "24000/1001"),
    "honeybee_2160p": (os.path.join(VIDEO_DIR, "HoneyBee_3840x2160_120fps_420_10bit_YUV.yuv"), 3840, 2160, 600, "120/1"),
    "bk600_4320p": (os.path.join(BENCH_DIR, "in", "bk600_7680x4320.yuv"), 7680, 4320, 5, "24000/1001"),
}
for _size in ("1918x1078", "1366x766", "642x362", "208x120"):
    _w, _h = (int(v) for v in _size.split("x"))
    CLIPS["bk600_" + _size] = (os.path.join(BENCH_DIR, "in", f"bk600_{_size}.yuv"), _w, _h, 9, "24000/1001")


def config(clip, preset, qp, frames, threads=-1, extra=()):
    return {"clip": clip, "preset": preset, "qp": qp, "frames": frames, "threads": threads, "extra": list(extra)}


CONFIGS = {
    # Small runs for iterating: a few seconds each.
    "bk600_1080p_faster_q28_f9": config("bk600_1080p", "faster", 28, 9),
    "bk1500_808p_fast_q24_f9": config("bk1500_808p", "fast", 24, 9),
    "bk600_1080p_medium_q32_f5": config("bk600_1080p", "medium", 32, 5),
    # Larger runs for final checks and timing.
    "bk600_1080p_medium": config("bk600_1080p", "medium", 28, 33),
    "bk600_2160p_medium": config("bk600_2160p", "medium", 28, 17),
    "bk600_1080p_faster": config("bk600_1080p", "faster", 28, 65),
    "bk600_2160p_faster": config("bk600_2160p", "faster", 28, 33),
    "bk600_1080p_slow": config("bk600_1080p", "slow", 28, 17),
    "bk1500_808p_slower": config("bk1500_808p", "slower", 28, 5),
    "bk1500_1608p_fast": config("bk1500_1608p", "fast", 24, 17),
    "bk1500_808p_medium_t1": config("bk1500_808p", "medium", 28, 9, threads=1),
    "bk1500_808p_medium_t16": config("bk1500_808p", "medium", 28, 17, threads=16),
    "honeybee_2160p_fast": config("honeybee_2160p", "fast", 30, 17),
    "bk600_1080p_rc2pass": config("bk600_1080p", "fast", -1, 17, extra=["-b", "2M", "-p", "2"]),
    # Rate control runs a first pass or a look-ahead encoder that skips pictures. Two GOPs are
    # needed to cover references from the second GOP into the first.
    "bk600_1080p_rc1pass_f65": config("bk600_1080p", "fast", -1, 65, extra=["-b", "2M", "-p", "1"]),
    "bk600_1080p_faster_rc2pass_f65": config("bk600_1080p", "faster", -1, 65, extra=["-b", "2M", "-p", "2"]),
    # Memory stress: 8K, sizes that are not multiples of 8 or of the CTU size, many threads.
    "bk600_4320p_medium_t16": config("bk600_4320p", "medium", 30, 3, threads=16),
    "bk600_1918x1078_fast": config("bk600_1918x1078", "fast", 28, 9),
    "bk600_1366x766_medium": config("bk600_1366x766", "medium", 28, 5),
    "bk600_642x362_slow": config("bk600_642x362", "slow", 28, 5),
    "bk600_208x120_medium": config("bk600_208x120", "medium", 28, 9),
    "bk600_1080p_medium_t32": config("bk600_1080p", "medium", 28, 9, threads=32),
    "bk600_1080p_rc2pass_f9": config("bk600_1080p", "fast", -1, 9, extra=["-b", "2M", "-p", "2"]),
    # The tooltest preset enables every coding tool, including the ones no regular preset uses.
    "bk600_642x362_tooltest": config("bk600_642x362", "tooltest", 30, 5),
    "bk600_1366x766_tooltest_t8": config("bk600_1366x766", "tooltest", 32, 3, threads=8),
    # Benchmarks at all 32 hardware threads, where per-thread memory matters most.
    "bk600_1080p_medium_t32_f33": config("bk600_1080p", "medium", 28, 33, threads=32),
    "bk600_2160p_medium_t32": config("bk600_2160p", "medium", 28, 17, threads=32),
    # Two full GOPs: the GOP encoder holds a whole GOP of pictures, as in a long encode.
    "bk600_1080p_medium_f65": config("bk600_1080p", "medium", 28, 65),
    "bk600_1080p_medium_t32_f65": config("bk600_1080p", "medium", 28, 65, threads=32),
    "bk600_2160p_medium_f65": config("bk600_2160p", "medium", 28, 65),
    "bk600_2160p_medium_t32_f65": config("bk600_2160p", "medium", 28, 65, threads=32),
    # The slow and slower presets reuse CU results, which costs memory per worker thread.
    "bk600_1080p_slow_t32": config("bk600_1080p", "slow", 28, 17, threads=32),
    "bk600_1080p_slower_t32": config("bk600_1080p", "slower", 28, 9, threads=32),
    "bk600_2160p_slow_t32": config("bk600_2160p", "slow", 28, 9, threads=32),
}

SUITES = {
    "quick": ["bk600_1080p_faster_q28_f9", "bk1500_808p_fast_q24_f9", "bk600_1080p_medium_q32_f5"],
    "full": [
        "bk600_1080p_medium", "bk600_2160p_medium", "bk600_1080p_faster", "bk600_2160p_faster",
        "bk600_1080p_slow", "bk1500_808p_slower", "bk1500_1608p_fast", "bk1500_808p_medium_t1",
        "bk1500_808p_medium_t16", "honeybee_2160p_fast", "bk600_1080p_rc2pass",
        "bk600_1080p_rc1pass_f65", "bk600_1080p_faster_rc2pass_f65",
    ],
    "guard": [
        "bk600_4320p_medium_t16", "bk600_1918x1078_fast", "bk600_1366x766_medium", "bk600_642x362_slow",
        "bk600_208x120_medium", "bk600_1080p_medium_t32", "bk600_1080p_rc2pass_f9",
        "bk600_642x362_tooltest", "bk600_1366x766_tooltest_t8",
    ],
}


class ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", wt.DWORD), ("PageFaultCount", wt.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def filetime_seconds(ft):
    return ((ft.dwHighDateTime << 32) | ft.dwLowDateTime) / 1e7


def system_busy_seconds():
    """Returns machine-wide busy CPU seconds, summed over all logical processors."""
    idle, kernel, user = wt.FILETIME(), wt.FILETIME(), wt.FILETIME()
    ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user))
    # Kernel time includes idle time on Windows.
    return filetime_seconds(kernel) + filetime_seconds(user) - filetime_seconds(idle)


def child_stats(handle):
    """Returns (cpu seconds, peak working set bytes, peak commit bytes) of a finished child."""
    creation, exit_, kernel, user = wt.FILETIME(), wt.FILETIME(), wt.FILETIME(), wt.FILETIME()
    ctypes.windll.kernel32.GetProcessTimes(wt.HANDLE(handle), ctypes.byref(creation), ctypes.byref(exit_),
                                           ctypes.byref(kernel), ctypes.byref(user))
    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    ctypes.windll.psapi.GetProcessMemoryInfo(wt.HANDLE(handle), ctypes.byref(counters), counters.cb)
    cpu = filetime_seconds(kernel) + filetime_seconds(user)
    return cpu, counters.PeakWorkingSetSize, counters.PeakPagefileUsage, counters.PageFaultCount


def encoder_args(cfg):
    path, width, height, frames_available, fps = CLIPS[cfg["clip"]]
    if cfg["frames"] > frames_available:
        raise ValueError(f"{cfg['clip']} has only {frames_available} frames")
    args = ["-i", path, "-s", f"{width}x{height}", "-c", "yuv420_10", "--fps", fps,
            "--preset", cfg["preset"], "-f", str(cfg["frames"]), "-t", str(cfg["threads"])]
    if cfg["qp"] >= 0:
        args += ["-q", str(cfg["qp"])]
    return args + cfg["extra"]


def config_key(cfg):
    """Identifies an encode by everything that can change its output, so baseline hashes can be cached."""
    return json.dumps(cfg, sort_keys=True)


CHILD_ENV = None  # environment of child encoders; None keeps ours


def run_encode(exe, cfg, out_path):
    """Runs one encode in a child process and returns its measurements and output hash."""
    cmd = [exe] + encoder_args(cfg) + ["-o", out_path]
    busy_before = system_busy_seconds()
    start = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=CHILD_ENV)
    output, _ = proc.communicate()
    wall = time.perf_counter() - start
    busy_after = system_busy_seconds()
    cpu, peak_ws, peak_commit, page_faults = child_stats(int(proc._handle))
    if proc.returncode != 0:
        # Sanitizer reports can be longer than the tail kept here, so their error lines come first.
        errors = [line for line in output.splitlines() if "ERROR: AddressSanitizer" in line or "SUMMARY: AddressSanitizer" in line]
        raise RuntimeError(f"encode failed ({proc.returncode}): {' '.join(cmd)}\n" + "\n".join(errors) + f"\n{output[-3000:]}")

    with open(out_path, "rb") as f:
        digest = hashlib.md5(f.read()).hexdigest()
    summary = re.search(r"Total Frames.*\n.*?(\d+)\s+a\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", output)
    cores = os.cpu_count()
    others = max(0.0, (busy_after - busy_before) - cpu)
    return {
        "md5": digest,
        "wall": wall,
        "fps": cfg["frames"] / wall,
        "cpu_percent": 100.0 * cpu / (wall * cores),
        "others_percent": 100.0 * others / (wall * cores),
        "peak_ws_mb": peak_ws / 2**20,
        "peak_commit_mb": peak_commit / 2**20,
        "page_faults": page_faults,
        "summary": summary.group(0).splitlines()[-1].strip() if summary else "",
    }


def baseline_cache_path():
    return os.path.join(BENCH_DIR, "out", "baseline_hashes.json")


def save_cache_entry(key, value):
    """Merges one entry into the cache file. Writing a temporary file and renaming it keeps the
    file valid when two harness runs add entries at the same time."""
    cache = load_cache()
    cache[key] = value
    tmp_path = f"{baseline_cache_path()}.{os.getpid()}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(cache, f, indent=1)
    os.replace(tmp_path, baseline_cache_path())


def baseline_result(name, cfg, cache):
    key = config_key(cfg)
    if key not in cache:
        result = run_encode(BASE_EXE, cfg, os.path.join(BENCH_DIR, "out", f"base_{name}.266"))
        cache[key] = {"md5": result["md5"], "summary": result["summary"]}
        save_cache_entry(key, cache[key])
    return cache[key]


def load_cache():
    try:
        with open(baseline_cache_path()) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def cmd_check(args):
    names = args.configs or SUITES[args.suite]
    cache = load_cache()
    failures = 0
    for name in names:
        cfg = CONFIGS[name]
        base = baseline_result(name, cfg, cache)
        # The build's folder is part of the name, so that checks of other builds running at the same
        # time cannot overwrite this bitstream before its hash is taken.
        cand = run_encode(args.cand, cfg, os.path.join(BENCH_DIR, "out", f"cand_{label_of(args.cand)}_{name}.266"))
        same = cand["md5"] == base["md5"]
        failures += 0 if same else 1
        print(f"{'OK  ' if same else 'DIFF'} {name:28s} {cand['md5']} {cand['wall']:7.1f}s  peak commit {cand['peak_commit_mb']:6.0f} MB  working set {cand['peak_ws_mb']:6.0f} MB"
              + ("" if same else f"\n     base: {base['summary']}\n     cand: {cand['summary']}"), flush=True)
    print(f"{len(names) - failures}/{len(names)} identical")
    return 1 if failures else 0


def cmd_time(args):
    """Times the baseline and one or more candidate builds in alternation, so that every build
    sees the same machine state. Each candidate is reported against the baseline."""
    exes = [("base", BASE_EXE)] + [(label_of(exe), exe) for exe in args.cand]
    for name in args.configs:
        cfg = CONFIGS[name]
        results = {label: [] for label, _ in exes}
        for _ in range(args.rounds):
            for label, exe in exes:
                results[label].append(run_encode(exe, cfg, os.path.join(BENCH_DIR, "out", f"time_{label}_{name}.266")))
        base = results["base"]
        fps_base = statistics.median(r["fps"] for r in base)
        print(f"{name}: base fps {fps_base:.3f}, cpu {statistics.median(r['cpu_percent'] for r in base):.1f}%,"
              f" commit {statistics.median(r['peak_commit_mb'] for r in base):.0f} MB,"
              f" working set {statistics.median(r['peak_ws_mb'] for r in base):.0f} MB", flush=True)
        for label, _ in exes[1:]:
            cand = results[label]
            same = "" if cand[0]["md5"] == base[0]["md5"] else "  OUTPUT DIFFERS"
            fps_cand = statistics.median(r["fps"] for r in cand)
            others = max(r["others_percent"] for r in base + cand)
            print(f"  {label:12s} fps {fps_cand:6.3f} ({100 * (fps_cand / fps_base - 1):+5.1f}%)"
                  f"  cpu {statistics.median(r['cpu_percent'] for r in cand):5.1f}%"
                  f"  commit {statistics.median(r['peak_commit_mb'] for r in cand):6.0f} MB"
                  f"  working set {statistics.median(r['peak_ws_mb'] for r in cand):6.0f} MB"
                  f"  others max {others:4.1f}%{same}"
                  f"  rounds {' '.join('%.3f' % r['fps'] for r in cand)}", flush=True)
    return 0


def label_of(exe):
    """Names a build after the folder that holds it, e.g. bin-step1."""
    return os.path.basename(os.path.dirname(os.path.abspath(exe)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="compare candidate bitstreams with the baseline")
    check.add_argument("--cand", required=True)
    check.add_argument("--suite", default="quick", choices=sorted(SUITES))
    check.add_argument("--configs", nargs="*")
    timing = sub.add_parser("time", help="time baseline and candidate in alternation")
    timing.add_argument("--cand", required=True, nargs="+")
    timing.add_argument("--configs", nargs="+", required=True)
    timing.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--path-prefix", help="directory put first on the PATH of child encoders, e.g. for sanitizer runtimes")
    args = parser.parse_args()
    if args.path_prefix:
        global CHILD_ENV
        CHILD_ENV = dict(os.environ, PATH=args.path_prefix + os.pathsep + os.environ.get("PATH", ""))
    os.makedirs(os.path.join(BENCH_DIR, "out"), exist_ok=True)
    return cmd_check(args) if args.command == "check" else cmd_time(args)


if __name__ == "__main__":
    sys.exit(main())
