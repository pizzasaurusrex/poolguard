"""Ground-truth labels for the replay regression suite.

The footage itself is gitignored (clips/ is local-only), so the *labels* are
the committed artifact: a manifest describing what each shot contains and what
the pipeline is expected to conclude about it. A checkout without the clips
still type-checks and unit-tests the manifest; the regression run skips the
shots whose files are absent.

One label describes one *shot* — a single continuous camera take. Clips edited
from a longer source are split at their cut boundaries first, because a cut
teleports every person in frame and forces the tracker to spawn new IDs,
inflating fragmentation metrics for reasons that have nothing to do with the
tracker.

This module is never imported by the core pipeline (it needs PyYAML, installed
by the `labels` extra and the dev group), following the same isolation rule as
the vision backends.
"""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from poolguard.events import EventType, OperatingMode
from poolguard.geometry import Polygon, validate_polygon

MIN_ONSET_WINDOW_SECONDS = 0.1
"""Narrowest onset window a label may claim.

A window tighter than this asserts frame-level precision that a human eyeballing
a clip cannot actually deliver, so it is far more likely a typo than a real
intent — and it would fail the rules engine for annotation jitter, not for
being wrong.
"""


class CameraProfile(BaseModel):
    """Fixed-camera geometry, shared by every shot recorded from that camera.

    Keyed by name in the manifest so twenty clips from one mount declare their
    water region once. Only meaningful for a camera that does not move; footage
    scraped from the internet has no stable geometry and carries no profile.
    """

    model_config = ConfigDict(frozen=True)

    water_polygon: Polygon = Field(
        description="Waterline as normalized (x, y) vertices, clockwise from any start. "
        "A polygon rather than a rect because real pools are not rectangular — the "
        "bounding box of a kidney pool is roughly a third deck, which is exactly where "
        "shadow false positives land",
    )
    crop_margin: float = Field(
        default=0.05,
        ge=0.0,
        le=0.5,
        description="Fraction of frame added around the polygon bounds when deriving the "
        "inference crop. A swimmer's head clears the waterline and people stand on the "
        "coping, so a crop tight to the water clips the people you most need to see",
    )
    notes: str = ""

    @model_validator(mode="after")
    def _polygon_encloses_an_area(self) -> "CameraProfile":
        validate_polygon(self.water_polygon, name="water_polygon")
        return self

    @property
    def crop_rect(self) -> tuple[float, float, float, float]:
        """Normalized (x, y, width, height) region worth running inference over.

        Everything outside it is roof, sky, or fence and cannot contain a swimmer.
        """
        xs = [x for x, _ in self.water_polygon]
        ys = [y for _, y in self.water_polygon]
        x0 = max(0.0, min(xs) - self.crop_margin)
        y0 = max(0.0, min(ys) - self.crop_margin)
        x1 = min(1.0, max(xs) + self.crop_margin)
        y1 = min(1.0, max(ys) + self.crop_margin)
        return (x0, y0, x1 - x0, y1 - y0)


class ExpectedEvent(BaseModel):
    """One event a shot must raise, and when.

    The window is part of the verdict: a verdict with no timing would pass even
    if the rule fired 40 seconds late, which is not a save.
    """

    model_config = ConfigDict(frozen=True)

    event: EventType
    onset_window: tuple[float, float] = Field(
        description="Seconds from shot start within which the event must fire; a fire outside "
        "this window is a failure even though the verdict is right",
    )

    @model_validator(mode="after")
    def _window_is_ordered_and_plausible(self) -> "ExpectedEvent":
        start, end = self.onset_window
        if start < 0:
            raise ValueError(f"onset_window start must be >= 0, got {start}")
        if end <= start:
            raise ValueError(f"onset_window must be ordered, got {self.onset_window}")
        if end - start < MIN_ONSET_WINDOW_SECONDS:
            raise ValueError(
                f"onset_window is too narrow ({end - start:.3f}s); must be at least "
                f"{MIN_ONSET_WINDOW_SECONDS}s to allow for human annotation error"
            )
        return self


class ShotLabel(BaseModel):
    """What a human asserts is true about one continuous shot in one operating mode."""

    model_config = ConfigDict(frozen=True)

    shot: str = Field(description="File name of the shot, relative to the clips root")
    source: str = Field(description="Original clip this shot was cut from")
    duration_seconds: float = Field(gt=0)
    people: int = Field(
        ge=0,
        description="Distinct humans visible in the shot; the tracker should converge to this "
        "many track IDs, so track_count > people is fragmentation",
    )
    mode: OperatingMode = Field(
        description="Mode the shot is replayed in. Required: people in the water are a swim "
        "session in one mode and an emergency in the other",
    )
    camera: str | None = Field(
        default=None,
        description="Name of a profile in the manifest's `cameras` block. None for footage "
        "with no known geometry — the scraped clips, where every shot is a different mount",
    )
    expected_events: tuple[ExpectedEvent, ...] = Field(
        default=(),
        description="Every event the rules engine must raise; empty if the shot must stay quiet",
    )
    notes: str = ""

    @model_validator(mode="after")
    def _windows_end_inside_the_shot(self) -> "ShotLabel":
        for expected in self.expected_events:
            end = expected.onset_window[1]
            if end > self.duration_seconds:
                raise ValueError(
                    f"{self.shot}: {expected.event.value} onset_window end ({end}) exceeds "
                    f"duration_seconds ({self.duration_seconds})"
                )
        return self


class ShotLabels(BaseModel):
    """The whole manifest."""

    model_config = ConfigDict(frozen=True)

    shots: tuple[ShotLabel, ...]
    cameras: dict[str, CameraProfile] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _camera_references_resolve(self) -> "ShotLabels":
        """A typo in a camera name must fail loudly, not silently drop the zone."""
        dangling = {
            shot.shot: shot.camera
            for shot in self.shots
            if shot.camera is not None and shot.camera not in self.cameras
        }
        if dangling:
            known = sorted(self.cameras) or ["<none defined>"]
            raise ValueError(
                f"shots reference unknown cameras: {dangling}; defined cameras are {known}"
            )
        return self

    def camera_for(self, shot: ShotLabel) -> CameraProfile | None:
        """Profile backing this shot, or None when its geometry is unknown."""
        return self.cameras.get(shot.camera) if shot.camera is not None else None

    @model_validator(mode="after")
    def _one_label_per_shot_and_mode(self) -> "ShotLabels":
        keys = [(shot.shot, shot.mode.value) for shot in self.shots]
        duplicates = {key for key in keys if keys.count(key) > 1}
        if duplicates:
            raise ValueError(f"duplicate (shot, mode) labels in manifest: {sorted(duplicates)}")
        return self

    def label_for(self, shot_name: str, mode: OperatingMode) -> ShotLabel | None:
        """The label for this file in this mode, or None if it has none."""
        for shot in self.shots:
            if shot.shot == shot_name and shot.mode is mode:
                return shot
        return None

    def positives(self) -> tuple[ShotLabel, ...]:
        """Shots where an event must fire."""
        return tuple(shot for shot in self.shots if shot.expected_events)

    def negatives(self) -> tuple[ShotLabel, ...]:
        """Shots that must stay quiet — the false-alarm budget."""
        return tuple(shot for shot in self.shots if not shot.expected_events)


def load_shot_labels(path: Path) -> ShotLabels:
    """Read and validate the YAML manifest.

    Fails loudly on a malformed manifest rather than silently regressing to an
    empty suite, which would make every regression test vacuously pass.
    """
    if not path.is_file():
        raise FileNotFoundError(f"shot label manifest not found: {path}")

    raw = yaml.safe_load(path.read_text())
    if raw is None:
        raise ValueError(f"shot label manifest is empty: {path}")
    if not isinstance(raw, dict):
        raise ValueError(f"shot label manifest must be a mapping, got {type(raw).__name__}: {path}")

    return ShotLabels.model_validate(raw)
