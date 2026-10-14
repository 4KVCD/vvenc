"""Compares the self-time sections of two vvprof reports.

Each report counts samples per function. The script scales both to milliseconds of busy
thread time using the sampling interval, then lists the functions whose time changed most.

Usage: python profdiff.py BASE_REPORT OTHER_REPORT [--interval-ms 2] [--section "self time by function"] [--top 40]
"""

import argparse
import re


def read_section(path, title):
    counts = {}
    inside = False
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("== "):
                inside = line[3:].startswith(title)
                continue
            match = re.match(r"\s*[\d.]+%\s+(\d+)\s+(.*)$", line.rstrip("\n"))
            if inside and match:
                counts[match.group(2)] = int(match.group(1))
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base")
    parser.add_argument("other")
    parser.add_argument("--interval-ms", type=float, default=2.0)
    parser.add_argument("--section", default="self time by function")
    parser.add_argument("--top", type=int, default=40)
    args = parser.parse_args()

    base = read_section(args.base, args.section)
    other = read_section(args.other, args.section)
    total_base = sum(base.values()) * args.interval_ms / 1000
    total_other = sum(other.values()) * args.interval_ms / 1000
    print(f"listed busy thread time: {total_base:.1f} s -> {total_other:.1f} s ({100 * (total_other / total_base - 1):+.1f}%)")

    rows = []
    for name in set(base) | set(other):
        b = base.get(name, 0) * args.interval_ms / 1000
        o = other.get(name, 0) * args.interval_ms / 1000
        rows.append((o - b, b, o, name))
    rows.sort(reverse=True)
    print(f"{'delta s':>8} {'base s':>8} {'other s':>8}  function")
    for delta, b, o, name in rows[: args.top]:
        print(f"{delta:+8.2f} {b:8.2f} {o:8.2f}  {name}")


if __name__ == "__main__":
    main()
