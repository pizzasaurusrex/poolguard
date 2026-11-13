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
(a quiet shot raises nothing; a positive raises each labeled event inside its
onset window, and nothing else). A file may be labeled once per operating
mode. Detection and tracking run once per file per run; the rules, which are
cheap, run once per labeled mode over that cached tracker output.
"""

from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import pytest

from poolguard.config import RulesSettings, TrackingSettings
from poolguard.events import EventType, OperatingMode, TrackedPerson
from poolguard.labels import ShotLabel, ShotLabels, load_shot_labels
from poolguard.rules import RulesState, evaluate

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


class TrackedFrame(NamedTuple):
    seconds_in: float
    ts: datetime
    people: tuple[TrackedPerson, ...]


class ShotReplay(NamedTuple):
    """What one file produced through detection and tracking, independent of mode."""

    frames: tuple[TrackedFrame, ...]
    track_count: int


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
    """Run one shot through detection and tracking; the rules run per mode afterwards."""
    from poolguard.replay import run_tracked_replay
    from poolguard.vision.video import VideoFileFrameSource

    source = VideoFileFrameSource(str(SHOTS_DIR / shot.shot))
    frames: list[TrackedFrame] = []
    track_ids: set[int] = set()
    for result in run_tracked_replay(source, estimator, _tracking_settings_for(shot)):
        track_ids.update(person.track_id for person in result.people)
        seconds_in = (result.ts - source.start_time).total_seconds()
        frames.append(TrackedFrame(seconds_in, result.ts, result.people))
    return ShotReplay(frames=tuple(frames), track_count=len(track_ids))


def _events_in_mode(replay: ShotReplay, mode: OperatingMode) -> tuple[RaisedEvent, ...]:
    """Apply the rules engine to the cached tracker output, as the live loop would."""
    rules = RulesSettings(_env_file=None)
    state = RulesState()
    raised: list[RaisedEvent] = []
    for frame in replay.frames:
        state, events = evaluate(state, frame.people, frame.ts, mode, rules)
        raised.extend(RaisedEvent(event.event_type, frame.seconds_in) for event in events)
    return tuple(raised)


@pytest.fixture(scope="session")
def replay_cache() -> dict[str, ShotReplay]:
    """One detection-and-tracking pass per file per run, shared by every mode's label."""
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
    """One parametrized case per label, ided by file and mode so failures name both."""
    return [pytest.param(shot, id=f"{shot.shot}[{shot.mode.value}]") for shot in SHOTS]


@pytest.mark.parametrize("replayed", _cases(), indirect=True)
class TestTrackerOnLabeledShots:
    def test_decodes_frames(self, replayed) -> None:  # type: ignore[no-untyped-def]
        shot, replay = replayed
        assert replay.frames, f"{shot.shot} decoded no frames"

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


def _is_expected(event: RaisedEvent, shot: ShotLabel) -> bool:
    return any(
        expected.event is event.event_type
        and expected.onset_window[0] <= event.seconds_in <= expected.onset_window[1]
        for expected in shot.expected_events
    )


@pytest.mark.parametrize("replayed", _cases(), indirect=True)
class TestEventsOnLabeledShots:
    def test_event_verdict_matches_the_label(self, replayed) -> None:  # type: ignore[no-untyped-def]
        """Each labeled event fires inside its window, and nothing else fires at all.

        There is no distress rule yet, so a shot expecting distress fails here
        until that rule is written.
        """
        shot, replay = replayed
        events = _events_in_mode(replay, shot.mode)
        raised = ", ".join(f"{e.event_type.value} at {e.seconds_in:.1f}s" for e in events)

        for expected in shot.expected_events:
            start, end = expected.onset_window
            on_time = [
                e for e in events if e.event_type is expected.event and start <= e.seconds_in <= end
            ]
            assert on_time, (
                f"{shot.shot} [{shot.mode.value}]: expected {expected.event.value} between "
                f"{start}s and {end}s, raised: {raised or 'nothing'}"
            )
        # Anything not covered by a labeled window is a false alarm, even on a clip with a
        # real event.
        unexpected = [e for e in events if not _is_expected(e, shot)]
        assert not unexpected, (
            f"{shot.shot} [{shot.mode.value}]: unexpected events; raised: {raised}"
        )
