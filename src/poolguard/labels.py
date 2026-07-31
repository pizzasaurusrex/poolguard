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

MIN_ONSET_WINDOW_SECONDS = 0.1
"""Narrowest onset window a label may claim.

A window tighter than this asserts frame-level precision that a human eyeballing
a clip cannot actually deliver, so it is far more likely a typo than a real
intent — and it would fail the rules engine for annotation jitter, not for
being wrong.
"""


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
