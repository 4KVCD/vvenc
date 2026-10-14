"""Cuts the raw test clips used by vvbench.py from real videos.

Sources (from the user's test video folder):
  The Beekeeper UHD Blu-ray remux: 3840x2160, 10-bit HEVC, 23.976 fps.

Clips (yuv420p10le, written to BENCH_DIR/in):
  bk600_3840x2160   65 frames from 10:00, uncropped 4K
  bk600_1920x1080   the same frames scaled to 1080p
  bk1500_3840x1608  33 frames from 25:00, letterbox removed as in the user's own encodes
  bk1500_1920x808   the same frames scaled to 1920x808
  bk600_7680x4320   5 frames scaled up to 8K, for memory stress tests
  bk600_WxH         9 frames cropped to sizes that are not multiples of 8 or of the CTU size

Usage: python make_clips.py [--only NAME ...]
"""

import argparse
import os
import subprocess

BENCH_DIR = os.environ.get("VVBENCH_DIR", r"C:\code\VVenc\.claude\bench")
SOURCE = r"C:\Video encodings\The.Beekeeper.2024.UHD.BluRay.2160p.TrueHD.Atmos.7.1.DV.HEVC.HYBRID.REMUX-FraMeSToR.mkv"
RAW = ["-pix_fmt", "yuv420p10le", "-f", "rawvideo"]


def clip_path(name):
    return os.path.join(BENCH_DIR, "in", name + ".yuv")


def from_source(start_seconds, frames, video_filter=None):
    args = ["-ss", str(start_seconds), "-i", SOURCE, "-map", "0:v:0", "-frames:v", str(frames)]
    return args + (["-vf", video_filter] if video_filter else [])


def from_clip(name, size, frames, video_filter):
    return ["-s", size] + RAW + ["-i", clip_path(name), "-frames:v", str(frames), "-vf", video_filter]


CLIPS = {
    "bk600_3840x2160": lambda: from_source(600, 65),
    "bk600_1920x1080": lambda: from_clip("bk600_3840x2160", "3840x2160", 65, "scale=1920:1080:flags=lanczos"),
    "bk1500_3840x1608": lambda: from_source(1500, 33, "crop=3840:1608:0:276"),
    "bk1500_1920x808": lambda: from_clip("bk1500_3840x1608", "3840x1608", 33, "scale=1920:808:flags=lanczos"),
    "bk600_7680x4320": lambda: from_clip("bk600_3840x2160", "3840x2160", 5, "scale=7680:4320:flags=lanczos"),
}
for size in ("1918x1078", "1366x766", "642x362", "208x120"):
    width, height = size.split("x")
    CLIPS["bk600_" + size] = (lambda w=width, h=height: from_clip("bk600_3840x2160", "3840x2160", 9, f"crop={w}:{h}:1000:500"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="*", help="clip names to make; default: all missing clips")
    args = parser.parse_args()
    os.makedirs(os.path.join(BENCH_DIR, "in"), exist_ok=True)
    for name, make_args in CLIPS.items():
        if args.only and name not in args.only:
            continue
        if not args.only and os.path.exists(clip_path(name)):
            continue
        cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y"] + make_args() + RAW + [clip_path(name)]
        print("making", name, flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
