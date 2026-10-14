"""Check that the camera still sees the view the pool zone was traced for.

Usage:
    poolguard-view-check save  --video clips/today.mp4 --reference calibration/backyard.viewref.npz
    poolguard-view-check check --video clips/later.mp4 --reference calibration/backyard.viewref.npz

`save` records the fixed landmarks around the pool zone. Run it once, right
after tracing the zone. `check` compares another recording against them and
exits non-zero if the camera has moved or the view cannot be confirmed, so the
zone is never trusted silently after a re-mount.

The pool zone comes from POOLGUARD_TRACKING_POOL_ZONE, or from a camera
profile with --labels and --camera. The reference describes a private scene:
keep it outside the repository (calibration/ is gitignored).

Requires the `vision` extra: uv sync --extra vision
"""

import argparse
from datetime import UTC, datetime
from pathlib import Path

try:
    from poolguard.vision.video import VideoFileFrameSource
    from poolguard.vision.view_compare import (
        build_reference,
        compare_view,
        load_reference,
        save_reference,
        steady_image,
    )
except ImportError as exc:
    raise SystemExit("vision deps not installed — run: uv sync --extra vision") from exc

import numpy as np

from poolguard.camera_view import degraded_event
from poolguard.config import RulesSettings, TrackingSettings, ViewSettings
from poolguard.geometry import WHOLE_FRAME, Polygon

SAMPLE_FRAMES = 25
SAMPLE_EVERY = 10
"""Frames blended into one steady image, and the gap between them. At 20 fps
that spans about 12 s, long enough for a swimmer to move out of any one spot."""


def _pool_zone(args: argparse.Namespace) -> Polygon:
    if args.labels and args.camera:
        try:
            from poolguard.labels import load_shot_labels
        except ImportError as exc:
            raise SystemExit("--labels needs PyYAML — run: uv sync --extra labels") from exc
        cameras = load_shot_labels(Path(args.labels)).cameras
        if args.camera not in cameras:
            raise SystemExit(f"no camera named {args.camera!r} in {args.labels}")
        return cameras[args.camera].water_polygon

    zone = TrackingSettings().pool_zone
    if zone == WHOLE_FRAME:
        raise SystemExit(
            "no pool zone is set: landmarks are taken from outside the zone, and the default "
            "zone is the whole frame. Set POOLGUARD_TRACKING_POOL_ZONE or pass --labels and "
            "--camera."
        )
    return zone


def _steady_view(video: str) -> np.ndarray:
    """One image of the scene with moving things (swimmers) blended out."""
    sampled = []
    for frame in VideoFileFrameSource(video).frames():
        if frame.index % SAMPLE_EVERY == 0:
            sampled.append(frame.image)
        if len(sampled) == SAMPLE_FRAMES:
            break
    if not sampled:
        raise SystemExit(f"no frames decoded from {video}")
    return steady_image(sampled)


def _save(args: argparse.Namespace) -> None:
    settings = ViewSettings()
    reference = build_reference(_steady_view(args.video), _pool_zone(args))
    if len(reference.points) < settings.min_landmarks:
        raise SystemExit(
            f"found only {len(reference.points)} landmarks outside the pool zone; "
            f"need at least {settings.min_landmarks}. Is the view mostly blank?"
        )
    save_reference(reference, Path(args.reference))
    print(f"saved {len(reference.points)} landmarks to {args.reference}")


def _check(args: argparse.Namespace) -> None:
    reference = load_reference(Path(args.reference))
    check = compare_view(reference, _steady_view(args.video), _pool_zone(args), ViewSettings())
    shift = "not measurable" if check.zone_shift is None else f"{check.zone_shift:.1%} of frame"
    print(
        f"view {check.verdict.value}: {check.matched_landmarks} landmarks agree, "
        f"spread {check.landmark_spread:.2f}, zone shift {shift}"
    )
    event = degraded_event(check, datetime.now(UTC), RulesSettings().default_mode)
    if event is not None:
        print(f"EVENT {event.tier.value} {event.event_type.value}: {event.detail}")
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the camera view against a reference.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, handler, summary in (
        ("save", _save, "record the landmarks of the current view"),
        ("check", _check, "compare a recording against the saved landmarks"),
    ):
        command = commands.add_parser(name, help=summary)
        command.add_argument("--video", required=True, help="recording from the camera")
        command.add_argument("--reference", required=True, help="landmark file (kept local)")
        command.add_argument("--labels", metavar="SHOTS_YAML", help="manifest with profiles")
        command.add_argument("--camera", help="camera profile in --labels supplying the zone")
        command.set_defaults(handler=handler)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
