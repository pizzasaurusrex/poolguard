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

from poolguard.config import TrackingSettings
from poolguard.events import TrackedPerson
from poolguard.replay import ReplaySummary, run_replay, run_tracked_replay


def _describe(person: TrackedPerson) -> str:
    unseen = person.seconds_since_last_seen
    suffix = f" unseen {unseen:.1f}s" if unseen else ""
    return f"#{person.track_id}{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a video through pose detection.")
    parser.add_argument("--video", required=True, help="path to a video file")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="ultralytics pose model")
    parser.add_argument("--min-confidence", type=float, default=0.25)
    parser.add_argument("--per-frame", action="store_true", help="print each frame's detections")
    parser.add_argument("--track", action="store_true", help="run the tracking stage")
    parser.add_argument(
        "--render", metavar="OUT_MP4", help="write an annotated video (implies --track)"
    )
    args = parser.parse_args()
    if args.render:
        args.track = True

    source = VideoFileFrameSource(args.video)
    estimator = UltralyticsPoseEstimator(model=args.model, min_confidence=args.min_confidence)

    summary = ReplaySummary()
    track_ids: set[int] = set()
    start = time.monotonic()

    if args.track:
        settings = TrackingSettings()
        if args.render:
            from poolguard.vision.render import render_tracked_replay

            tracked_results = render_tracked_replay(source, estimator, settings, args.render)
        else:
            tracked_results = run_tracked_replay(source, estimator, settings)
        for tracked in tracked_results:
            summary = summary.fold(tracked)
            track_ids.update(p.track_id for p in tracked.people)
            if args.per_frame:
                people = ", ".join(_describe(p) for p in tracked.people)
                print(f"frame {tracked.frame_index:5d}  tracks=[{people}]")
    else:
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
    tracks = f" tracks={len(track_ids)}" if args.track else ""
    print(
        f"frames={summary.frames} frames_with_people={summary.frames_with_people} "
        f"detections={summary.detections}{tracks} elapsed={elapsed:.1f}s fps={fps:.1f}"
    )


if __name__ == "__main__":
    main()
