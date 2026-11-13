"""Is the camera still looking where the pool zone says it is?

The pool zone is traced for one camera view. If the camera is re-mounted at a
different angle the outline no longer sits on the water, and `in_water` is
quietly wrong: a missed submersion or a false alarm, with nothing failing.

This module is the dependency-free half of the check: it turns measurements
into a verdict, and a bad verdict into a SYSTEM_DEGRADED event. Taking the
measurements needs OpenCV and lives in vision/view_compare.py.
"""

from enum import StrEnum

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from poolguard.config import ViewSettings
from poolguard.events import AlertTier, EventType, OperatingMode, SafetyEvent


class ViewVerdict(StrEnum):
    ALIGNED = "aligned"
    MOVED = "moved"
    UNKNOWN = "unknown"


class ViewCheck(BaseModel):
    """Result of comparing the live view against the saved reference."""

    model_config = ConfigDict(frozen=True)

    verdict: ViewVerdict
    matched_landmarks: int = Field(ge=0)
    """Fixed points found in both views that agree on how the view moved."""
    landmark_spread: float = Field(ge=0.0, le=1.0)
    """How much of the frame those points span: the smaller of the normalized
    width and height of the box around them."""
    zone_shift: float | None = Field(default=None, ge=0.0)
    """Farthest any pool-zone vertex moved, as a fraction of frame width.
    None when it could not be measured."""


def classify_view(
    matched_landmarks: int,
    landmark_spread: float,
    zone_shift: float | None,
    settings: ViewSettings,
) -> ViewCheck:
    """Decide whether the pool zone can still be trusted.

    UNKNOWN is deliberate, and treated as untrusted: too few landmarks, or
    landmarks bunched in one spot (a burned-in logo matches at zero shift
    after any move), cannot vouch for the view.
    """
    evidence_is_weak = (
        zone_shift is None
        or matched_landmarks < settings.min_landmarks
        or landmark_spread < settings.min_landmark_spread
    )
    if evidence_is_weak:
        verdict = ViewVerdict.UNKNOWN
    elif zone_shift > settings.max_zone_shift:
        verdict = ViewVerdict.MOVED
    else:
        verdict = ViewVerdict.ALIGNED
    return ViewCheck(
        verdict=verdict,
        matched_landmarks=matched_landmarks,
        landmark_spread=landmark_spread,
        zone_shift=zone_shift,
    )


def degraded_event(check: ViewCheck, now: AwareDatetime, mode: OperatingMode) -> SafetyEvent | None:
    """The alert for a view that is not confirmed aligned, or None if it is."""
    if check.verdict is ViewVerdict.ALIGNED:
        return None
    if check.verdict is ViewVerdict.MOVED and check.zone_shift is not None:
        detail = (
            f"The camera has moved: the pool outline is off by about "
            f"{check.zone_shift:.0%} of the frame and is not trusted. Re-trace it."
        )
    else:
        detail = (
            "The camera view could not be confirmed against its reference "
            f"({check.matched_landmarks} matching landmarks). The pool outline is not "
            "trusted. Check the camera, then re-trace the outline."
        )
    return SafetyEvent(
        event_type=EventType.SYSTEM_DEGRADED,
        tier=AlertTier.WARN,
        started_at=now,
        mode=mode,
        confidence=1.0,
        detail=detail,
    )
