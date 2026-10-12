"""Replay regression suite over labeled shots.

Opt-in — these decode real video through a real pose model, which is far too
slow for the normal test loop:

    uv run pytest -m regression

Every shot in the manifest becomes its own test case, so a failure names the
shot that broke. A shot whose file is not present locally skips rather than
fails: the footage is gitignored, so a fresh checkout legitimately has none of
it, and a red suite for a missing file would train everyone to ignore red.

What is asserted today is tracker behavior (does it find the right number of
people, and does it hold their IDs). The event-level assertions land with the
rules engine; their labels are already carried in the manifest.
"""

from pathlib import Path

import pytest

from poolguard.config import TrackingSettings
from poolguard.labels import ShotLabel, load_shot_labels

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


def _load_shots() -> tuple[ShotLabel, ...]:
    """Read the manifest at collection time so each shot is its own test id."""
    try:
        return load_shot_labels(MANIFEST).shots
    except (FileNotFoundError, ValueError):
        # test_labels.py reports the real diagnosis; don't fail collection here.
        return ()


SHOTS = _load_shots()


@pytest.fixture(scope="session")
def estimator():  # type: ignore[no-untyped-def]
    ultralytics_pose = pytest.importorskip(
        "poolguard.vision.ultralytics_pose",
        reason="needs the vision extra: uv sync --extra vision",
    )
    return ultralytics_pose.UltralyticsPoseEstimator()


def _replay(shot: ShotLabel, estimator) -> tuple[int, int]:  # type: ignore[no-untyped-def]
    """Run one shot through detect + track. Returns (frames, distinct track IDs)."""
    from poolguard.replay import run_tracked_replay
    from poolguard.vision.video import VideoFileFrameSource

    source = VideoFileFrameSource(str(SHOTS_DIR / shot.shot))
    settings = TrackingSettings()

    frames = 0
    track_ids: set[int] = set()
    for result in run_tracked_replay(source, estimator, settings):
        frames += 1
        track_ids.update(person.track_id for person in result.people)
    return frames, len(track_ids)


@pytest.fixture
def replayed(request, estimator):  # type: ignore[no-untyped-def]
    shot: ShotLabel = request.param
    path = SHOTS_DIR / shot.shot
    if not path.is_file():
        pytest.skip(f"footage not present locally: {path}")
    return shot, *_replay(shot, estimator)


def _cases() -> list[object]:
    """One parametrized case per shot, ided by file name so failures name it."""
    return [pytest.param(shot, id=shot.shot) for shot in SHOTS]


@pytest.mark.parametrize("replayed", _cases(), indirect=True)
class TestTrackerOnLabeledShots:
    def test_decodes_frames(self, replayed) -> None:  # type: ignore[no-untyped-def]
        shot, frames, _ = replayed
        assert frames > 0, f"{shot.shot} decoded no frames"

    def test_finds_everyone(self, replayed) -> None:  # type: ignore[no-untyped-def]
        shot, _, track_count = replayed
        assert track_count >= shot.people, (
            f"{shot.shot}: tracked {track_count} people, expected at least "
            f"{shot.people} — a miss, not fragmentation"
        )

    def test_does_not_fragment(self, replayed) -> None:  # type: ignore[no-untyped-def]
        shot, _, track_count = replayed
        if shot.people == 0:
            assert track_count == 0, f"{shot.shot}: {track_count} tracks in empty water"
            return
        ratio = track_count / shot.people
        assert ratio <= MAX_FRAGMENTATION, (
            f"{shot.shot}: {track_count} track IDs for {shot.people} people "
            f"(ratio {ratio:.1f} > {MAX_FRAGMENTATION}) — ID churn"
        )
