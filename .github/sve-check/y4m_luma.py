#!/usr/bin/env python3
# Reads a 1920x1080 4:2:0 8-bit y4m stream on stdin and writes the luma planes of frames 0 and 4 to stdout (raw,
# 8-bit), so that a bench can read two pictures four apart from a pipe. Nothing is written to disk.
import sys
W, H = 1920, 1080
inp, out = sys.stdin.buffer, sys.stdout.buffer
header = inp.readline()
assert header.startswith(b"YUV4MPEG2") and b" W1920 " in header and b" H1080 " in header, header[:80]
for n in range(5):
    assert inp.readline().startswith(b"FRAME")
    y = inp.read(W * H)
    inp.read(W * H // 2)
    if n in (0, 4):
        out.write(y)
out.flush()
