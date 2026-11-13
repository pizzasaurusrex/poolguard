"""Replay a video file through the pose pipeline and report detections.

Usage:
    poolguard-replay --video clips/swim.mp4 [--model yolo11n-pose.pt] [--per-frame]

Requires the `vision` extra: uv sync --extra vision
(--labels also needs the `labels` extra: uv sync --extra vision --extra labels)
"""

import argparse
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from poolguard.labels import CameraProfile, ShotLabel

try:
    from poolguard.vision.ultralytics_pose import (
        DEFAULT_MAX_INFERENCE_SIZE,
        DEFAULT_MODEL,
        UltralyticsPoseEstimator,
    )
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


def _find_label(args: argparse.Namespace) -> "tuple[ShotLabel | None, CameraProfile | None]":
    """Look up the manifest entry matching the replayed file, and its camera profile.

    The profile is None when the shot names no camera (scraped footage has no
    stable geometry), as well as when there is no label.
    """
    if not args.labels:
        return None, None
    from pathlib import Path

    try:
        from poolguard.labels import load_shot_labels
    except ImportError as exc:
        raise SystemExit("--labels needs PyYAML — run: uv sync --extra labels") from exc

    wanted = Path(args.video).name
    manifest = load_shot_labels(Path(args.labels))
    for shot in manifest.shots:
        if shot.shot == wanted:
            return shot, manifest.camera_for(shot)
    print(f"warning: no label for {wanted} in {args.labels}; replaying without expectations")
    return None, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a video through pose detection.")
    parser.add_argument("--video", required=True, help="path to a video file")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="ultralytics pose model")
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=None,
        help="detection floor; defaults to the tracker's low_confidence",
    )
    parser.add_argument(
        "--max-inference-size",
        type=int,
        default=DEFAULT_MAX_INFERENCE_SIZE,
        help="cap on the model input size in pixels; lower is faster but misses distant people",
    )
    parser.add_argument("--per-frame", action="store_true", help="print each frame's detections")
    parser.add_argument("--track", action="store_true", help="run the tracking and rules stages")
    parser.add_argument(
        "--render", metavar="OUT_MP4", help="write an annotated video (implies --track)"
    )
    parser.add_argument(
        "--labels",
        metavar="SHOTS_YAML",
        help="shot manifest; the entry matching --video supplies the expected event for the "
        "overlay and, when its camera has a profile, the pool zone",
    )
    args = parser.parse_args()
    if args.render:
        args.track = True

    label, camera = _find_label(args)
    settings = TrackingSettings()
    if camera is not None:
        # Already validated by CameraProfile, so a plain copy is safe.
        settings = settings.model_copy(update={"pool_zone": camera.water_polygon})
    min_confidence = (
        args.min_confidence if args.min_confidence is not None else settings.low_confidence
    )

    source = VideoFileFrameSource(args.video)
    estimator = UltralyticsPoseEstimator(
        model=args.model,
        min_confidence=min_confidence,
        max_inference_size=args.max_inference_size,
    )

    summary = ReplaySummary()
    track_ids: set[int] = set()
    event_count = 0
    start = time.monotonic()

    if args.track:
        if args.render:
            from poolguard.vision.render import render_tracked_replay

            tracked_results = render_tracked_replay(
                source, estimator, settings, args.render, label=label
            )
        else:
            tracked_results = run_tracked_replay(source, estimator, settings)
        for tracked in tracked_results:
            summary = summary.fold(tracked)
            track_ids.update(p.track_id for p in tracked.people)
            for event in tracked.events:
                event_count += 1
                seconds_in = (tracked.ts - source.start_time).total_seconds()
                print(
                    f"EVENT t={seconds_in:.1f}s {event.tier.value} {event.event_type.value} "
                    f"track #{event.track_id}: {event.detail}"
                )
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
    tracks = f" tracks={len(track_ids)} events={event_count}" if args.track else ""
    print(
        f"frames={summary.frames} frames_with_people={summary.frames_with_people} "
        f"detections={summary.detections}{tracks} elapsed={elapsed:.1f}s fps={fps:.1f}"
    )


if __name__ == "__main__":
    main()
