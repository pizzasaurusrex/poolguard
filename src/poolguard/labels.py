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

This module is never imported by the core pipeline (it needs PyYAML, a dev-only
dependency), following the same isolation rule as the vision backends.
"""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from poolguard.events import EventType

MIN_POLYGON_POINTS = 3
"""Fewest vertices that enclose an area; below this there is no inside."""

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

    water_polygon: tuple[tuple[float, float], ...] = Field(
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
        if len(self.water_polygon) < MIN_POLYGON_POINTS:
            raise ValueError(
                f"water_polygon needs at least {MIN_POLYGON_POINTS} vertices, "
                f"got {len(self.water_polygon)}"
            )
        for x, y in self.water_polygon:
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError(f"water_polygon vertex out of normalized range: {(x, y)}")
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


class ShotLabel(BaseModel):
    """What a human asserts is true about one continuous shot."""

    model_config = ConfigDict(frozen=True)

    shot: str = Field(description="File name of the shot, relative to the clips root")
    source: str = Field(description="Original clip this shot was cut from")
    duration_seconds: float = Field(gt=0)
    people: int = Field(
        ge=0,
        description="Distinct humans visible in the shot; the tracker should converge to this "
        "many track IDs, so track_count > people is fragmentation",
    )
    camera: str | None = Field(
        default=None,
        description="Name of a profile in the manifest's `cameras` block. None for footage "
        "with no known geometry — the scraped clips, where every shot is a different mount",
    )
    expected_event: EventType | None = Field(
        default=None,
        description="Event the rules engine must raise, or None if the shot must stay quiet",
    )
    onset_window: tuple[float, float] | None = Field(
        default=None,
        description="Seconds from shot start within which the event must fire; a fire outside "
        "this window is a failure even though the verdict is right",
    )
    notes: str = ""

    @model_validator(mode="after")
    def _window_is_ordered_and_in_bounds(self) -> "ShotLabel":
        if self.onset_window is None:
            return self
        start, end = self.onset_window
        if start < 0:
            raise ValueError(f"{self.shot}: onset_window start must be >= 0, got {start}")
        if end <= start:
            raise ValueError(f"{self.shot}: onset_window must be ordered, got {self.onset_window}")
        if end > self.duration_seconds:
            raise ValueError(
                f"{self.shot}: onset_window end ({end}) exceeds "
                f"duration_seconds ({self.duration_seconds})"
            )
        return self

    @model_validator(mode="after")
    def _event_and_window_agree(self) -> "ShotLabel":
        """Require both fields or neither: a verdict is not a label until it is timed."""
        if self.expected_event is None and self.onset_window is not None:
            raise ValueError(
                f"{self.shot}: onset_window is set but expected_event is None; "
                "must either set both or neither"
            )
        if self.expected_event is not None and self.onset_window is None:
            raise ValueError(
                f"{self.shot}: expected_event is set but onset_window is None; "
                "must either set both or neither"
            )
        if self.onset_window is not None:
            start, end = self.onset_window
            if end - start < MIN_ONSET_WINDOW_SECONDS:
                raise ValueError(
                    f"{self.shot}: onset_window is too narrow ({end - start:.3f}s); "
                    f"must be at least {MIN_ONSET_WINDOW_SECONDS}s to allow for "
                    "human annotation error"
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
    def _shot_names_are_unique(self) -> "ShotLabels":
        names = [shot.shot for shot in self.shots]
        duplicates = {name for name in names if names.count(name) > 1}
        if duplicates:
            raise ValueError(f"duplicate shot names in manifest: {sorted(duplicates)}")
        return self

    def positives(self) -> tuple[ShotLabel, ...]:
        """Shots where an event must fire."""
        return tuple(shot for shot in self.shots if shot.expected_event is not None)

    def negatives(self) -> tuple[ShotLabel, ...]:
        """Shots that must stay quiet — the false-alarm budget."""
        return tuple(shot for shot in self.shots if shot.expected_event is None)


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
