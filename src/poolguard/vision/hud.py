"""Heads-up overlay: the pipeline's running state drawn onto each frame.

The per-track boxes in render.py answer "where does the system think people
are". This answers "what does the system currently believe, and is that what
the label says should be happening" — the view you scrub to find the exact
frame where a rule should have fired and didn't.

Requires the `vision` extra (opencv).
"""

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

_PANEL_COLOR = (28, 28, 28)
_TEXT_COLOR = (235, 235, 235)
_MUTED_COLOR = (150, 150, 150)
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_PANEL_ALPHA = 0.65

_ALERT_COLOR = (60, 60, 230)  # red — the rule would fire now
_WARN_COLOR = (40, 220, 220)  # yellow — a clock is running
_BLIND_COLOR = (200, 160, 40)  # teal — detector sees nothing
_CALM_COLOR = (80, 220, 60)  # green — nominal


class HudState(BaseModel):
    """Everything the overlay knows about one frame."""

    model_config = ConfigDict(frozen=True)

    frame_index: int = Field(ge=0)
    elapsed_seconds: float = Field(ge=0.0)
    live_tracks: int = Field(ge=0)
    coasting_tracks: int = Field(ge=0)
    longest_unseen_seconds: float = Field(ge=0.0)
    """Longest time a track last seen *in the water* has gone unseen. A track
    lost on the deck (someone walking out of frame) coasts too, but it is not
    a submersion signal and does not count here."""
    detection_rate: float = Field(ge=0.0, le=1.0)
    """Share of frames so far that produced at least one detection."""

    expected_event: str | None = None
    onset_window: tuple[float, float] | None = None
    submersion_timeout_seconds: float = Field(gt=0)

    @property
    def in_onset_window(self) -> bool:
        """True when the clock is inside the window the label expects a fire."""
        if self.onset_window is None:
            return False
        start, end = self.onset_window
        return start <= self.elapsed_seconds <= end

    @property
    def submersion_progress(self) -> float:
        """How far the longest-unseen in-water track is toward the submersion timeout."""
        return min(self.longest_unseen_seconds / self.submersion_timeout_seconds, 1.0)


def status_line(state: HudState) -> tuple[str, tuple[int, int, int]]:
    """The one line an operator reads: what the system believes right now.

    Ordered most-urgent first, so the highest-priority true condition wins and
    the rest never draw. Returns the text and its BGR color.
    """
    if state.submersion_progress >= 1.0:
        return f"SUBMERSION — unseen {state.longest_unseen_seconds:.1f}s", _ALERT_COLOR
    if state.longest_unseen_seconds > 0:
        return f"in-water track lost {state.longest_unseen_seconds:.1f}s", _WARN_COLOR
    if state.live_tracks == 0:
        return "NO ONE DETECTED", _BLIND_COLOR
    if state.in_onset_window:
        return f"in onset window — expecting {state.expected_event}", _WARN_COLOR
    return f"tracking {state.live_tracks}", _CALM_COLOR


def draw_hud(canvas: np.ndarray, state: HudState) -> np.ndarray:
    """Draw the HUD panel onto a copy of the frame."""
    out = canvas.copy()
    height, width = out.shape[:2]
    panel_height = 74

    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (width, panel_height), _PANEL_COLOR, -1)
    cv2.addWeighted(overlay, _PANEL_ALPHA, out, 1 - _PANEL_ALPHA, 0, out)

    clock = f"t={state.elapsed_seconds:5.2f}s  f{state.frame_index:04d}"
    cv2.putText(out, clock, (8, 20), _FONT, 0.5, _TEXT_COLOR, 1, cv2.LINE_AA)

    tracks = (
        f"live {state.live_tracks}  coasting {state.coasting_tracks}  "
        f"det {state.detection_rate * 100:3.0f}%"
    )
    cv2.putText(out, tracks, (8, 40), _FONT, 0.45, _MUTED_COLOR, 1, cv2.LINE_AA)

    text, color = status_line(state)
    if text:
        cv2.putText(out, text, (8, 62), _FONT, 0.55, color, 2, cv2.LINE_AA)

    _draw_expectation(out, state, width)
    if state.longest_unseen_seconds > 0:
        _draw_submersion_meter(out, state, width, panel_height)
    return out


def _draw_expectation(canvas: np.ndarray, state: HudState, width: int) -> None:
    """Right-aligned reminder of what the label expects, so drift is obvious."""
    if state.expected_event is None:
        label = "label: must stay quiet"
    else:
        start, end = state.onset_window or (0.0, 0.0)
        label = f"label: {state.expected_event} @ {start:.1f}-{end:.1f}s"
    color = _TEXT_COLOR if state.in_onset_window else _MUTED_COLOR
    (text_width, _), _ = cv2.getTextSize(label, _FONT, 0.45, 1)
    cv2.putText(canvas, label, (width - text_width - 8, 20), _FONT, 0.45, color, 1, cv2.LINE_AA)


def _draw_submersion_meter(
    canvas: np.ndarray, state: HudState, width: int, panel_height: int
) -> None:
    """A bar filling toward the submersion timeout — the rule's clock, visible.

    Coasting time is the submersion signal, so watching this bar fill is the
    fastest way to see whether a track was lost long enough to matter.
    """
    bar_width = 160
    left = width - bar_width - 8
    top = panel_height - 26
    cv2.rectangle(canvas, (left, top), (left + bar_width, top + 10), _MUTED_COLOR, 1)
    filled = int(bar_width * state.submersion_progress)
    color = (60, 60, 230) if state.submersion_progress >= 1.0 else (40, 220, 220)
    if filled > 0:
        cv2.rectangle(canvas, (left, top), (left + filled, top + 10), color, -1)
    cv2.putText(
        canvas,
        f"unseen {state.longest_unseen_seconds:4.1f}s / {state.submersion_timeout_seconds:.0f}s",
        (left, top + 24),
        _FONT,
        0.4,
        _MUTED_COLOR,
        1,
        cv2.LINE_AA,
    )
