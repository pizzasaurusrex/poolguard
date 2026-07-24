"""P0 bench test: measure sustained frame-read FPS from the camera's RTSP
stream. Establishes the capture baseline before any inference is added.

Usage:
    poolguard-benchmark --rtsp-url rtsp://user:pass@cam/h264Preview_01_main

Requires the `vision` extra: uv sync --extra vision
"""

import argparse
import time


def measure_fps(rtsp_url: str, duration_seconds: float) -> tuple[float, int, int]:
    """Return (fps, frames_read, frames_dropped) over the sample window."""
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("opencv not installed — run: uv sync --extra vision") from exc

    capture = cv2.VideoCapture(rtsp_url)
    if not capture.isOpened():
        raise SystemExit(f"could not open stream: {rtsp_url}")

    frames_read = 0
    frames_dropped = 0
    start = time.monotonic()
    try:
        while time.monotonic() - start < duration_seconds:
            ok, _frame = capture.read()
            if ok:
                frames_read += 1
            else:
                frames_dropped += 1
    finally:
        capture.release()

    elapsed = time.monotonic() - start
    return frames_read / elapsed, frames_read, frames_dropped


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure RTSP capture FPS.")
    parser.add_argument("--rtsp-url", required=True)
    parser.add_argument("--duration", type=float, default=30.0, help="sample window in seconds")
    args = parser.parse_args()

    print(f"sampling {args.rtsp_url} for {args.duration:.0f}s ...")
    fps, read, dropped = measure_fps(args.rtsp_url, args.duration)
    print(f"fps={fps:.1f} frames_read={read} frames_dropped={dropped}")
    if fps < 15:
        print("WARNING: below the 15 FPS P0 exit criterion (PRD §10)")


if __name__ == "__main__":
    main()
