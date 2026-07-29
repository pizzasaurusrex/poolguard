"""Replay a video file through the pose pipeline and report detections.

Usage:
    poolguard-replay --video clips/swim.mp4 [--model yolo11n-pose.pt] [--per-frame]

Requires the `vision` extra: uv sync --extra vision
"""

import argparse
import time

try:
    from poolguard.vision.ultralytics_pose import DEFAULT_MODEL, UltralyticsPoseEstimator
    from poolguard.vision.video import VideoFileFrameSource
except ImportError as exc:
    raise SystemExit("vision deps not installed — run: uv sync --extra vision") from exc

from poolguard.replay import ReplaySummary, run_replay


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a video through pose detection.")
    parser.add_argument("--video", required=True, help="path to a video file")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="ultralytics pose model")
    parser.add_argument("--min-confidence", type=float, default=0.25)
    parser.add_argument("--per-frame", action="store_true", help="print each frame's detections")
    args = parser.parse_args()

    source = VideoFileFrameSource(args.video)
    estimator = UltralyticsPoseEstimator(model=args.model, min_confidence=args.min_confidence)

    summary = ReplaySummary()
    start = time.monotonic()
    for result in run_replay(source, estimator):
        summary = summary.fold(result)
        if args.per_frame:
            confidences = ", ".join(f"{d.confidence:.2f}" for d in result.detections)
            people = len(result.detections)
            print(f"frame {result.frame_index:5d}  people={people}  [{confidences}]")
    elapsed = time.monotonic() - start

    if summary.frames == 0:
        raise SystemExit(f"no frames decoded from {args.video}")

    fps = summary.frames / elapsed if elapsed > 0 else 0.0
    print(
        f"frames={summary.frames} frames_with_people={summary.frames_with_people} "
        f"detections={summary.detections} elapsed={elapsed:.1f}s fps={fps:.1f}"
    )


if __name__ == "__main__":
    main()
