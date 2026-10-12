"""P0 bench test: measure sustained frame-read FPS from the camera's RTSP
stream. Establishes the capture baseline before any inference is added.

Usage:
    POOLGUARD_CAMERA_RTSP_URL=rtsp://user:pass@cam/h264Preview_01_main poolguard-benchmark

The URL carries the camera's credentials, so it is read from the environment
or .env (never a command-line argument, which lands in shell history and `ps`)
and is never printed.

Requires the `vision` extra: uv sync --extra vision
"""

import argparse
import time

from pydantic import SecretStr, ValidationError

from poolguard.config import CameraSettings

RTSP_URL_ENV_VAR = "POOLGUARD_CAMERA_RTSP_URL"
MIN_ACCEPTABLE_FPS = 15
"""P0 exit criterion (PRD §10)."""


def measure_fps(rtsp_url: SecretStr, duration_seconds: float) -> tuple[float, int, int]:
    """Return (fps, frames_read, frames_dropped) over the sample window."""
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("opencv not installed — run: uv sync --extra vision") from exc

    capture = cv2.VideoCapture(rtsp_url.get_secret_value())
    if not capture.isOpened():
        raise SystemExit(f"could not open the camera stream; check {RTSP_URL_ENV_VAR}")

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
    fps = frames_read / elapsed if elapsed > 0 else 0.0
    return fps, frames_read, frames_dropped


def _load_rtsp_url() -> SecretStr:
    try:
        return CameraSettings().rtsp_url
    except ValidationError:
        # `from None`: the validation error can echo the rejected input.
        raise SystemExit(
            f"camera URL not configured — set {RTSP_URL_ENV_VAR} in the environment or .env"
        ) from None


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure RTSP capture FPS.")
    parser.add_argument("--duration", type=float, default=30.0, help="sample window in seconds")
    args = parser.parse_args()

    rtsp_url = _load_rtsp_url()
    print(f"sampling the camera stream for {args.duration:.0f}s ...")
    fps, read, dropped = measure_fps(rtsp_url, args.duration)
    print(f"fps={fps:.1f} frames_read={read} frames_dropped={dropped}")
    if fps < MIN_ACCEPTABLE_FPS:
        print(f"WARNING: below the {MIN_ACCEPTABLE_FPS} FPS P0 exit criterion (PRD §10)")


if __name__ == "__main__":
    main()
