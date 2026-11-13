"""Replay: run the detection pipeline over a FrameSource offline.

This is the PD-phase test harness (ADR-010): detect, then track, then rules.
It depends only on the seam protocols, so tests drive it with fakes and the
CLI drives it with real backends.
"""

from collections.abc import Iterator

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from poolguard.config import RulesSettings, TrackingSettings
from poolguard.events import Detection, SafetyEvent, TrackedPerson
from poolguard.rules import RulesState, evaluate
from poolguard.tracking import TrackerState, advance
from poolguard.vision.frames import FrameSource
from poolguard.vision.pose import PoseEstimator


class FrameResult(BaseModel):
    """Detections for one replayed frame."""

    model_config = ConfigDict(frozen=True)

    frame_index: int = Field(ge=0)
    ts: AwareDatetime
    detections: tuple[Detection, ...]


class ReplaySummary(BaseModel):
    """Running totals over a replay, folded immutably per frame."""

    model_config = ConfigDict(frozen=True)

    frames: int = 0
    frames_with_people: int = 0
    detections: int = 0

    def fold(self, result: "FrameResult | TrackedFrameResult") -> "ReplaySummary":
        return ReplaySummary(
            frames=self.frames + 1,
            frames_with_people=self.frames_with_people + (1 if result.detections else 0),
            detections=self.detections + len(result.detections),
        )


def run_replay(source: FrameSource, estimator: PoseEstimator) -> Iterator[FrameResult]:
    for frame in source.frames():
        detections = estimator.estimate(frame)
        yield FrameResult(frame_index=frame.index, ts=frame.ts, detections=detections)


class TrackedFrameResult(BaseModel):
    """Tracked people (including coasting tracks) and any events raised on one frame."""

    model_config = ConfigDict(frozen=True)

    frame_index: int = Field(ge=0)
    ts: AwareDatetime
    detections: tuple[Detection, ...]
    people: tuple[TrackedPerson, ...]
    events: tuple[SafetyEvent, ...] = ()


def run_tracked_replay(
    source: FrameSource,
    estimator: PoseEstimator,
    settings: TrackingSettings,
    rules: RulesSettings | None = None,
) -> Iterator[TrackedFrameResult]:
    """run_replay with the tracking and rules stages folded in.

    The frame timestamp is the clock for both stages, so replay behaves
    identically at any processing speed — coasting timers and rule timeouts
    measure video time, not wall-clock time. The operating mode is the
    rules settings' default_mode.
    """
    rules = rules or RulesSettings()
    tracker_state = TrackerState()
    rules_state = RulesState()
    for result in run_replay(source, estimator):
        tracker_state, people = advance(tracker_state, result.detections, result.ts, settings)
        rules_state, events = evaluate(rules_state, people, result.ts, rules.default_mode, rules)
        yield TrackedFrameResult(
            frame_index=result.frame_index,
            ts=result.ts,
            detections=result.detections,
            people=people,
            events=events,
        )
