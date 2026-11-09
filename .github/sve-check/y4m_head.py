#!/usr/bin/env python3
# Copies the y4m header and the first N frames (argv[1]) of a 4:2:0 8-bit y4m stream from stdin to stdout, then stops
# reading, so that `curl ... | y4m_head.py N | encoder` streams only N pictures. Nothing is written to disk.
import sys
n = int(sys.argv[1])
inp, out = sys.stdin.buffer, sys.stdout.buffer
header = inp.readline()
fields = dict((f[:1], f[1:]) for f in header.split()[1:])
frame = int(fields[b"W"]) * int(fields[b"H"]) * 3 // 2
out.write(header)
for _ in range(n):
    out.write(inp.readline())
    out.write(inp.read(frame))
out.flush()
