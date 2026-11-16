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
HUD_REFERENCE_WIDTH = 960
"""Frame width the overlay's sizes are given for; wider frames scale up."""

_ALERT_COLOR = (60, 60, 230)  # red — the rule would fire now
_WARN_COLOR = (40, 220, 220)  # yellow — a clock is running
_EDGE_COLOR = (0, 150, 255)  # orange — armed, someone at the coping
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

    armed: bool = False
    in_water_tracks: int = Field(default=0, ge=0)
    """Live tracks past the edge margin inside the pool; while armed, each is an entry."""
    edge_tracks: int = Field(default=0, ge=0)
    """Live tracks within the edge margin of the outline; while armed, a warning."""

    raised: tuple[tuple[str, float], ...] = ()
    """Every event raised so far as (event name, seconds in), oldest first."""

    expected: tuple[tuple[str, tuple[float, float]], ...] = ()
    """The label's expected events as (event name, onset window) pairs."""
    submersion_timeout_seconds: float = Field(gt=0)

    @property
    def expecting_now(self) -> str | None:
        """The labeled event whose onset window the clock is inside, if any."""
        for event, (start, end) in self.expected:
            if start <= self.elapsed_seconds <= end:
                return event
        return None

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
    if state.armed and state.in_water_tracks > 0:
        return f"ENTRY — {state.in_water_tracks} in the water while armed", _ALERT_COLOR
    if state.longest_unseen_seconds > 0:
        return f"in-water track lost {state.longest_unseen_seconds:.1f}s", _WARN_COLOR
    if state.armed and state.edge_tracks > 0:
        return f"EDGE — {state.edge_tracks} at the pool edge while armed", _EDGE_COLOR
    if state.live_tracks == 0:
        return "NO ONE DETECTED", _BLIND_COLOR
    if state.expecting_now is not None:
        return f"in onset window — expecting {state.expecting_now}", _WARN_COLOR
    return f"tracking {state.live_tracks}", _CALM_COLOR


_ALARM_NAMES = {
    "unsupervised_entry": "ENTRY",
    "submersion": "SUBMERSION",
    "person_at_edge": "EDGE",
    "distress": "DISTRESS",
    "system_degraded": "DEGRADED",
}
_WARNING_EVENTS = frozenset({"person_at_edge"})


def alarm_line(state: HudState) -> tuple[str, tuple[int, int, int]] | None:
    """Everything raised so far, so an alarm stays on screen after its frame passes.

    One entry per event type at its first time, with a count of repeats (a person
    whose track fragments raises the same warning once per track). Red if any
    emergency was raised, orange if only warnings.
    """
    if not state.raised:
        return None
    first: dict[str, float] = {}
    count: dict[str, int] = {}
    for event, seconds in state.raised:
        first.setdefault(event, seconds)
        count[event] = count.get(event, 0) + 1
    parts = []
    for event, seconds in first.items():
        repeats = f" (+{count[event] - 1})" if count[event] > 1 else ""
        parts.append(f"{_ALARM_NAMES.get(event, event.upper())} {seconds:.1f}s{repeats}")
    only_warnings = all(event in _WARNING_EVENTS for event in first)
    return "RAISED  " + "  |  ".join(parts), _EDGE_COLOR if only_warnings else _ALERT_COLOR


def hud_scale(width: int) -> float:
    """Size multiplier for overlay text and lines: 1 at HUD_REFERENCE_WIDTH, never below."""
    return max(1.0, width / HUD_REFERENCE_WIDTH)


def draw_hud(canvas: np.ndarray, state: HudState) -> np.ndarray:
    """Draw the HUD panel onto a copy of the frame, scaled to the frame's width."""
    out = canvas.copy()
    height, width = out.shape[:2]
    k = hud_scale(width)
    panel_height = round(96 * k)

    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (width, panel_height), _PANEL_COLOR, -1)
    cv2.addWeighted(overlay, _PANEL_ALPHA, out, 1 - _PANEL_ALPHA, 0, out)

    clock = f"t={state.elapsed_seconds:5.2f}s  f{state.frame_index:04d}"
    _text(out, clock, (8, 20), 0.5, _TEXT_COLOR, 1, k)

    tracks = (
        f"live {state.live_tracks}  coasting {state.coasting_tracks}  "
        f"det {state.detection_rate * 100:3.0f}%"
    )
    _text(out, tracks, (8, 40), 0.45, _MUTED_COLOR, 1, k)

    text, color = status_line(state)
    if text:
        _text(out, text, (8, 62), 0.55, color, 2, k)
    alarms = alarm_line(state)
    if alarms is not None:
        _text(out, alarms[0], (8, 86), 0.55, alarms[1], 2, k)

    _draw_expectation(out, state, width, k)
    if state.longest_unseen_seconds > 0:
        _draw_submersion_meter(out, state, width, panel_height, k)
    return out


def _text(
    canvas: np.ndarray,
    text: str,
    origin: tuple[int, int],
    font_scale: float,
    color: tuple[int, int, int],
    thickness: int,
    k: float,
) -> None:
    """putText with the position, size and stroke given at 1x and multiplied by k."""
    x, y = origin
    cv2.putText(
        canvas,
        text,
        (round(x * k), round(y * k)),
        _FONT,
        font_scale * k,
        color,
        max(1, round(thickness * k)),
        cv2.LINE_AA,
    )


def _draw_expectation(canvas: np.ndarray, state: HudState, width: int, k: float) -> None:
    """Right-aligned reminder of what the label expects, so drift is obvious."""
    if not state.expected:
        label = "label: must stay quiet"
    else:
        label = "label: " + ", ".join(
            f"{event} @ {start:.1f}-{end:.1f}s" for event, (start, end) in state.expected
        )
    color = _TEXT_COLOR if state.expecting_now is not None else _MUTED_COLOR
    thickness = max(1, round(k))
    (text_width, _), _ = cv2.getTextSize(label, _FONT, 0.45 * k, thickness)
    x = width - text_width - round(8 * k)
    cv2.putText(canvas, label, (x, round(20 * k)), _FONT, 0.45 * k, color, thickness, cv2.LINE_AA)


def _draw_submersion_meter(
    canvas: np.ndarray, state: HudState, width: int, panel_height: int, k: float
) -> None:
    """A bar filling toward the submersion timeout — the rule's clock, visible.

    Coasting time is the submersion signal, so watching this bar fill is the
    fastest way to see whether a track was lost long enough to matter.
    """
    bar_width = round(160 * k)
    bar_height = round(10 * k)
    left = width - bar_width - round(8 * k)
    top = round(48 * k)
    line = max(1, round(k))
    cv2.rectangle(canvas, (left, top), (left + bar_width, top + bar_height), _MUTED_COLOR, line)
    filled = int(bar_width * state.submersion_progress)
    color = _ALERT_COLOR if state.submersion_progress >= 1.0 else _WARN_COLOR
    if filled > 0:
        cv2.rectangle(canvas, (left, top), (left + filled, top + bar_height), color, -1)
    cv2.putText(
        canvas,
        f"unseen {state.longest_unseen_seconds:4.1f}s / {state.submersion_timeout_seconds:.0f}s",
        (left, top + round(24 * k)),
        _FONT,
        0.4 * k,
        _MUTED_COLOR,
        line,
        cv2.LINE_AA,
    )
