"""Queries folded call stacks written by vvprof --stacks.

Each input line is "root;...;leaf count" with frames "function@file:line".

  python stackq.py STACKS self [--top N]              self time by leaf function
  python stackq.py STACKS inclusive [--top N]         time of every function, counted once per stack
  python stackq.py STACKS callers PATTERN [--depth N] who calls the innermost frame matching PATTERN
  python stackq.py STACKS callees PATTERN             what the outermost frame matching PATTERN spends time in

Percentages are shares of all busy samples in the file.
"""

import argparse
import collections


def load(path):
    stacks = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            body, _, count = line.rstrip("\n").rpartition(" ")
            if body:
                stacks.append((body.split(";"), int(count)))
    return stacks


def name(frame, with_line=False):
    return frame if with_line else frame.split("@", 1)[0]


def print_top(counter, total, top):
    for key, count in counter.most_common(top):
        print(f"{100.0 * count / total:6.2f}%  {count:7d}  {key}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stacks")
    parser.add_argument("query", choices=["self", "inclusive", "callers", "callees"])
    parser.add_argument("pattern", nargs="?")
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--lines", action="store_true", help="keep file:line in frame names")
    args = parser.parse_args()

    stacks = load(args.stacks)
    total = sum(count for _, count in stacks)
    result = collections.Counter()

    for frames, count in stacks:
        names = [name(f, args.lines) for f in frames]
        if args.query == "self":
            result[names[-1]] += count
        elif args.query == "inclusive":
            for n in set(names):
                result[n] += count
        elif args.query == "callers":
            hits = [i for i, f in enumerate(frames) if args.pattern in name(f)]
            if not hits:
                continue
            innermost = hits[-1]
            chain = []
            for f in reversed(frames[:innermost]):
                n = name(f, args.lines)
                if args.pattern in n or (chain and chain[-1] == n):
                    continue
                chain.append(n)
                if len(chain) == args.depth:
                    break
            result[" < ".join([name(frames[innermost], args.lines)] + chain)] += count
        elif args.query == "callees":
            hits = [i for i, f in enumerate(frames) if args.pattern in name(f)]
            if not hits:
                continue
            outermost = hits[0]
            below = [name(f, args.lines) for f in frames[outermost + 1:] if args.pattern not in name(f)]
            result[below[0] if below else "(self)"] += count

    print(f"busy samples: {total}")
    print_top(result, total, args.top)


if __name__ == "__main__":
    main()
