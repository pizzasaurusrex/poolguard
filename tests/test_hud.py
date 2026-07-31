"""Tests for the HUD overlay state and status priority.

The status line shows one message, so what matters is which condition wins
when several are true at once — an operator scrubbing footage must not see
"calm" while a track is coasting.
"""

import pytest

pytest.importorskip("cv2", reason="needs the vision extra: uv sync --extra vision")

import numpy as np  # noqa: E402

from poolguard.vision.hud import (  # noqa: E402
    _ALERT_COLOR,
    _BLIND_COLOR,
    _CALM_COLOR,
    _WARN_COLOR,
    HudState,
    draw_hud,
    status_line,
)


def state(**overrides: object) -> HudState:
    defaults: dict[str, object] = {
        "frame_index": 0,
        "elapsed_seconds": 0.0,
        "live_tracks": 1,
        "coasting_tracks": 0,
        "longest_unseen_seconds": 0.0,
        "detection_rate": 1.0,
        "submersion_timeout_seconds": 12.0,
    }
    return HudState(**(defaults | overrides))  # type: ignore[arg-type]


class TestHudState:
    def test_submersion_progress_scales_to_timeout(self) -> None:
        assert state(longest_unseen_seconds=6.0).submersion_progress == 0.5

    def test_submersion_progress_clamps_at_one(self) -> None:
        assert state(longest_unseen_seconds=99.0).submersion_progress == 1.0

    def test_in_onset_window_is_false_without_a_window(self) -> None:
        assert state(elapsed_seconds=5.0).in_onset_window is False

    def test_in_onset_window_is_inclusive(self) -> None:
        assert state(elapsed_seconds=5.0, onset_window=(5.0, 7.0)).in_onset_window is True
        assert state(elapsed_seconds=7.0, onset_window=(5.0, 7.0)).in_onset_window is True
        assert state(elapsed_seconds=7.1, onset_window=(5.0, 7.0)).in_onset_window is False


class TestStatusPriority:
    def test_submersion_outranks_everything(self) -> None:
        text, color = status_line(
            state(
                live_tracks=0,
                coasting_tracks=1,
                longest_unseen_seconds=12.0,
                onset_window=(0.0, 99.0),
                expected_event="submersion",
            )
        )
        assert "SUBMERSION" in text
        assert color == _ALERT_COLOR

    def test_coasting_outranks_blind_and_calm(self) -> None:
        text, color = status_line(
            state(live_tracks=0, coasting_tracks=1, longest_unseen_seconds=3.0)
        )
        assert "lost 3.0s" in text
        assert color == _WARN_COLOR

    def test_blind_when_nothing_detected(self) -> None:
        text, color = status_line(state(live_tracks=0))
        assert text == "NO ONE DETECTED"
        assert color == _BLIND_COLOR

    def test_onset_window_when_otherwise_nominal(self) -> None:
        text, color = status_line(
            state(elapsed_seconds=6.0, onset_window=(5.0, 7.0), expected_event="distress")
        )
        assert "distress" in text
        assert color == _WARN_COLOR

    def test_calm_is_the_fallback(self) -> None:
        text, color = status_line(state(live_tracks=2))
        assert text == "tracking 2"
        assert color == _CALM_COLOR


class TestDrawHud:
    def test_does_not_mutate_the_input_frame(self) -> None:
        frame = np.zeros((120, 200, 3), dtype=np.uint8)
        before = frame.copy()
        draw_hud(frame, state())
        assert np.array_equal(frame, before)

    def test_returns_same_shape(self) -> None:
        frame = np.zeros((120, 200, 3), dtype=np.uint8)
        assert draw_hud(frame, state()).shape == frame.shape

    def test_draws_something(self) -> None:
        frame = np.zeros((120, 200, 3), dtype=np.uint8)
        assert draw_hud(frame, state()).any(), "HUD drew nothing onto the frame"
