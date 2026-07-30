"""Replay: run the detection pipeline over a FrameSource offline.

This is the PD-phase test harness (ADR-010): the same loop later gains
tracking and rules stages. It depends only on the seam protocols, so tests
drive it with fakes and the CLI drives it with real backends.
"""

from collections.abc import Iterator

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from poolguard.config import TrackingSettings
from poolguard.events import Detection, TrackedPerson
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
    """Tracked people (including coasting tracks) for one replayed frame."""

    model_config = ConfigDict(frozen=True)

    frame_index: int = Field(ge=0)
    ts: AwareDatetime
    detections: tuple[Detection, ...]
    people: tuple[TrackedPerson, ...]


def run_tracked_replay(
    source: FrameSource, estimator: PoseEstimator, settings: TrackingSettings
) -> Iterator[TrackedFrameResult]:
    """run_replay with the tracking stage folded in.

    The frame timestamp is the tracker's clock, so replay behaves identically
    at any processing speed — coasting timers measure video time, not
    wall-clock time.
    """
    state = TrackerState()
    for result in run_replay(source, estimator):
        state, people = advance(state, result.detections, result.ts, settings)
        yield TrackedFrameResult(
            frame_index=result.frame_index,
            ts=result.ts,
            detections=result.detections,
            people=people,
        )
