# Memory-layout work: verification and measurement tools

These scripts check that an encoder build produces byte-identical bitstreams to the unmodified
encoder on real video, and measure its speed and peak memory. Every encode runs the real
`vvencapp` in a child process.

| File | Purpose |
|---|---|
| `make_clips.py` | Cuts the raw 10-bit test clips from The Beekeeper UHD remux: 4K, 1080p, letterbox-cropped 3840x1608 and 1920x808, 8K and sizes that are not multiples of 8 or of the CTU size. |
| `vvbench.py` | `check` compares bitstream MD5s of a candidate build with the baseline build on a suite of configs (`quick`, `full`, `guard`). `time` runs the baseline and one or more candidates in alternation and reports fps, CPU, peak commit and peak working set. |
| `vvprof.cpp` | Sampling profiler for Windows without administrator rights. Prints self time, inclusive time, callers of memcpy and memset, and hot source lines from the PDBs. `--stacks` writes folded call stacks. |
| `stackq.py` | Queries folded stacks: self, inclusive, callers and callees of a function. |
| `profdiff.py` | Compares the self-time sections of two `vvprof` reports. |
| `thrash.cpp` | Prints which logical processors share a core and an L3 cache, and streams memory on chosen processors to evict a shared L3 during an experiment. |

## Typical use

```
python make_clips.py
python vvbench.py check --cand path\to\vvencapp.exe --suite quick
python vvbench.py check --cand path\to\vvencapp.exe --suite full
python vvbench.py --path-prefix "<MSVC bin dir>" check --cand path\to\asan\vvencapp.exe --suite guard
python vvbench.py time --cand bin-step1\vvencapp.exe bin-step2\vvencapp.exe --configs bk600_1080p_medium bk600_2160p_medium
```

The baseline is `bin-base\vvencapp.exe` in the bench folder (`VVBENCH_DIR`, default
`C:\code\VVenc\.claude\bench`). Baseline hashes are cached in `out\baseline_hashes.json`.

Build the profiler and the cache tool with `cl /O2 /EHsc vvprof.cpp` and `cl /O2 /EHsc thrash.cpp`.
