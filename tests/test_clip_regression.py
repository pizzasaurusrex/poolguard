"""Replay regression suite over labeled shots.

Opt-in — these decode real video through a real pose model, which is far too
slow for the normal test loop:

    uv run pytest -m regression

Every shot in the manifest becomes its own test case, so a failure names the
shot that broke. A shot whose file is not present locally skips rather than
fails: the footage is gitignored, so a fresh checkout legitimately has none of
it, and a red suite for a missing file would train everyone to ignore red.

Two things are asserted per shot: tracker behavior (does it find the right
number of people, and does it hold their IDs) and the rules engine's verdict
(a quiet shot raises nothing; a positive raises its event inside the labeled
onset window). Each shot is replayed once per run and shared by its tests.
"""

from pathlib import Path
from typing import NamedTuple

import pytest

from poolguard.config import TrackingSettings
from poolguard.events import EventType
from poolguard.labels import ShotLabel, ShotLabels, load_shot_labels

REPO_ROOT = Path(__file__).parent.parent
MANIFEST = Path(__file__).parent / "labels" / "shots.yaml"
SHOTS_DIR = REPO_ROOT / "clips"

MAX_FRAGMENTATION = 2.0
"""Allowed ratio of distinct track IDs to real people before a shot fails.

Tuning knob, not a law: 1.0 would mean flawless ID retention, which no IoU
tracker achieves on splashing swimmers. Tighten it as the tracker improves —
a loosened value here is a regression that should be argued for in review.
"""

pytestmark = pytest.mark.regression


def _load_manifest() -> ShotLabels | None:
    """Read the manifest at collection time so each shot is its own test id."""
    try:
        return load_shot_labels(MANIFEST)
    except (FileNotFoundError, ValueError):
        # test_labels.py reports the real diagnosis; don't fail collection here.
        return None


LABELS = _load_manifest()
SHOTS = LABELS.shots if LABELS is not None else ()


class RaisedEvent(NamedTuple):
    event_type: EventType
    seconds_in: float


class ShotReplay(NamedTuple):
    """What one shot produced: the facts every test below asserts on."""

    frames: int
    track_count: int
    events: tuple[RaisedEvent, ...]


@pytest.fixture(scope="session")
def estimator():  # type: ignore[no-untyped-def]
    ultralytics_pose = pytest.importorskip(
        "poolguard.vision.ultralytics_pose",
        reason="needs the vision extra: uv sync --extra vision",
    )
    return ultralytics_pose.UltralyticsPoseEstimator(
        min_confidence=TrackingSettings().low_confidence
    )


def _tracking_settings_for(shot: ShotLabel) -> TrackingSettings:
    """Default settings, with the pool zone from the shot's camera profile if it has one."""
    settings = TrackingSettings()
    camera = LABELS.camera_for(shot) if LABELS is not None else None
    if camera is None:
        return settings
    return settings.model_copy(update={"pool_zone": camera.water_polygon})


def _replay(shot: ShotLabel, estimator) -> ShotReplay:  # type: ignore[no-untyped-def]
    """Run one shot through detect, track and rules."""
    from poolguard.replay import run_tracked_replay
    from poolguard.vision.video import VideoFileFrameSource

    source = VideoFileFrameSource(str(SHOTS_DIR / shot.shot))

    frames = 0
    track_ids: set[int] = set()
    events: list[RaisedEvent] = []
    for result in run_tracked_replay(source, estimator, _tracking_settings_for(shot)):
        frames += 1
        track_ids.update(person.track_id for person in result.people)
        seconds_in = (result.ts - source.start_time).total_seconds()
        for event in result.events:
            events.append(RaisedEvent(event.event_type, seconds_in))
    return ShotReplay(frames=frames, track_count=len(track_ids), events=tuple(events))


@pytest.fixture(scope="session")
def replay_cache() -> dict[str, ShotReplay]:
    """One replay per shot per run; every test for that shot reads the same result."""
    return {}


@pytest.fixture
def replayed(request, estimator, replay_cache):  # type: ignore[no-untyped-def]
    shot: ShotLabel = request.param
    path = SHOTS_DIR / shot.shot
    if not path.is_file():
        pytest.skip(f"footage not present locally: {path}")
    if shot.shot not in replay_cache:
        replay_cache[shot.shot] = _replay(shot, estimator)
    return shot, replay_cache[shot.shot]


def _cases() -> list[object]:
    """One parametrized case per shot, ided by file name so failures name it."""
    return [pytest.param(shot, id=shot.shot) for shot in SHOTS]


@pytest.mark.parametrize("replayed", _cases(), indirect=True)
class TestTrackerOnLabeledShots:
    def test_decodes_frames(self, replayed) -> None:  # type: ignore[no-untyped-def]
        shot, replay = replayed
        assert replay.frames > 0, f"{shot.shot} decoded no frames"

    def test_finds_everyone(self, replayed) -> None:  # type: ignore[no-untyped-def]
        shot, replay = replayed
        assert replay.track_count >= shot.people, (
            f"{shot.shot}: tracked {replay.track_count} people, expected at least "
            f"{shot.people} — a miss, not fragmentation"
        )

    def test_does_not_fragment(self, replayed) -> None:  # type: ignore[no-untyped-def]
        shot, replay = replayed
        if shot.people == 0:
            assert replay.track_count == 0, f"{shot.shot}: {replay.track_count} tracks, no people"
            return
        ratio = replay.track_count / shot.people
        assert ratio <= MAX_FRAGMENTATION, (
            f"{shot.shot}: {replay.track_count} track IDs for {shot.people} people "
            f"(ratio {ratio:.1f} > {MAX_FRAGMENTATION}) — ID churn"
        )


@pytest.mark.parametrize("replayed", _cases(), indirect=True)
class TestEventsOnLabeledShots:
    def test_event_verdict_matches_the_label(self, replayed) -> None:  # type: ignore[no-untyped-def]
        """A quiet shot raises nothing; a positive raises its event inside the onset window.

        Only the submersion rule exists, so a shot labeled `distress` fails
        here until that rule is written.
        """
        shot, replay = replayed
        raised = ", ".join(f"{e.event_type.value} at {e.seconds_in:.1f}s" for e in replay.events)

        if shot.expected_event is None:
            assert replay.events == (), f"{shot.shot}: must stay quiet but raised {raised}"
            return

        assert shot.onset_window is not None  # the manifest enforces both-or-neither
        start, end = shot.onset_window
        on_time = [
            event
            for event in replay.events
            if event.event_type is shot.expected_event and start <= event.seconds_in <= end
        ]
        assert on_time, (
            f"{shot.shot}: expected {shot.expected_event.value} between {start}s and {end}s, "
            f"raised: {raised or 'nothing'}"
        )
        # An alarm outside the window is a false alarm even on a clip that has a real event.
        outside_window = [e for e in replay.events if not start <= e.seconds_in <= end]
        assert not outside_window, (
            f"{shot.shot}: event outside the {start}s to {end}s onset window; raised: {raised}"
        )
