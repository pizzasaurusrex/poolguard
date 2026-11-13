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

    def test_expecting_nothing_without_a_window(self) -> None:
        assert state(elapsed_seconds=5.0).expecting_now is None

    def test_expecting_now_is_inclusive(self) -> None:
        window = (("distress", (5.0, 7.0)),)
        assert state(elapsed_seconds=5.0, expected=window).expecting_now == "distress"
        assert state(elapsed_seconds=7.0, expected=window).expecting_now == "distress"
        assert state(elapsed_seconds=7.1, expected=window).expecting_now is None

    def test_expecting_now_picks_the_open_window(self) -> None:
        windows = (("unsupervised_entry", (0.0, 5.0)), ("submersion", (34.0, 39.0)))
        assert state(elapsed_seconds=35.0, expected=windows).expecting_now == "submersion"


class TestStatusPriority:
    def test_submersion_outranks_everything(self) -> None:
        text, color = status_line(
            state(
                live_tracks=0,
                coasting_tracks=1,
                longest_unseen_seconds=12.0,
                armed=True,
                in_water_tracks=1,
                expected=(("submersion", (0.0, 99.0)),),
            )
        )
        assert "SUBMERSION" in text
        assert color == _ALERT_COLOR

    def test_entry_while_armed_outranks_coasting(self) -> None:
        text, color = status_line(
            state(armed=True, in_water_tracks=2, coasting_tracks=1, longest_unseen_seconds=3.0)
        )
        assert text == "ENTRY — 2 in the water while armed"
        assert color == _ALERT_COLOR

    def test_people_in_the_water_in_swim_mode_are_not_an_entry(self) -> None:
        text, _ = status_line(state(armed=False, in_water_tracks=2, live_tracks=2))
        assert text == "tracking 2"

    def test_coasting_outranks_blind_and_calm(self) -> None:
        text, color = status_line(
            state(live_tracks=0, coasting_tracks=1, longest_unseen_seconds=3.0)
        )
        assert "lost 3.0s" in text
        assert color == _WARN_COLOR

    def test_track_lost_outside_the_pool_is_not_a_warning(self) -> None:
        """Someone walking off the deck coasts too; that is not a submersion clock."""
        text, color = status_line(
            state(live_tracks=1, coasting_tracks=1, longest_unseen_seconds=0.0)
        )
        assert text == "tracking 1"
        assert color == _CALM_COLOR

    def test_blind_when_nothing_detected(self) -> None:
        text, color = status_line(state(live_tracks=0))
        assert text == "NO ONE DETECTED"
        assert color == _BLIND_COLOR

    def test_onset_window_when_otherwise_nominal(self) -> None:
        text, color = status_line(state(elapsed_seconds=6.0, expected=(("distress", (5.0, 7.0)),)))
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
