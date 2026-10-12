"""Tests for the shot-label manifest schema and loader.

These run everywhere — no footage, no model, no vision extra. They guard the
manifest itself, so a typo in shots.yaml fails fast instead of silently
shrinking the regression suite.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from poolguard.events import EventType
from poolguard.labels import CameraProfile, ShotLabel, ShotLabels, load_shot_labels

MANIFEST = Path(__file__).parent / "labels" / "shots.yaml"


def label(**overrides: object) -> ShotLabel:
    defaults: dict[str, object] = {
        "shot": "drowning_01.mp4",
        "source": "drowning.mp4",
        "duration_seconds": 10.0,
        "people": 1,
    }
    return ShotLabel(**(defaults | overrides))  # type: ignore[arg-type]


class TestShotLabel:
    def test_quiet_shot_needs_no_event(self) -> None:
        shot = label()
        assert shot.expected_event is None
        assert shot.onset_window is None

    def test_accepts_event_with_window(self) -> None:
        shot = label(expected_event=EventType.DISTRESS, onset_window=(2.0, 7.0))
        assert shot.expected_event is EventType.DISTRESS
        assert shot.onset_window == (2.0, 7.0)

    def test_rejects_unordered_window(self) -> None:
        with pytest.raises(ValidationError, match="ordered"):
            label(expected_event=EventType.DISTRESS, onset_window=(7.0, 2.0))

    def test_rejects_negative_window_start(self) -> None:
        with pytest.raises(ValidationError, match="must be >= 0"):
            label(expected_event=EventType.DISTRESS, onset_window=(-1.0, 5.0))

    def test_rejects_window_past_end_of_shot(self) -> None:
        with pytest.raises(ValidationError, match="exceeds"):
            label(
                duration_seconds=6.0,
                expected_event=EventType.DISTRESS,
                onset_window=(2.0, 9.0),
            )

    def test_rejects_window_without_event(self) -> None:
        with pytest.raises(ValidationError, match="must either set both or neither"):
            label(onset_window=(2.0, 7.0))

    def test_rejects_event_without_window(self) -> None:
        # A verdict with no timing is a half-finished label: it would pass even
        # if the rule fired 40 seconds late, which is not a save.
        with pytest.raises(ValidationError, match="must either set both or neither"):
            label(expected_event=EventType.DISTRESS)

    def test_rejects_window_too_narrow_to_be_real(self) -> None:
        with pytest.raises(ValidationError, match="too narrow"):
            label(expected_event=EventType.DISTRESS, onset_window=(5.0, 5.001))

    def test_is_frozen(self) -> None:
        shot = label()
        with pytest.raises(ValidationError):
            shot.people = 2  # type: ignore[misc]


class TestShotLabels:
    def test_rejects_duplicate_shot_names(self) -> None:
        with pytest.raises(ValidationError, match="duplicate shot names"):
            ShotLabels(shots=(label(), label()))

    def test_partitions_positives_and_negatives(self) -> None:
        quiet = label(shot="calm_01.mp4")
        loud = label(
            shot="loud_01.mp4",
            expected_event=EventType.SUBMERSION,
            onset_window=(1.0, 5.0),
        )
        labels = ShotLabels(shots=(quiet, loud))
        assert labels.positives() == (loud,)
        assert labels.negatives() == (quiet,)


class TestLoader:
    def test_loads_the_committed_manifest(self) -> None:
        labels = load_shot_labels(MANIFEST)
        assert labels.shots, "manifest must not be empty — an empty suite passes vacuously"

    def test_missing_file_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_shot_labels(tmp_path / "nope.yaml")

    def test_empty_file_is_an_error(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.yaml"
        path.write_text("")
        with pytest.raises(ValueError, match="empty"):
            load_shot_labels(path)

    def test_non_mapping_is_an_error(self, tmp_path: Path) -> None:
        path = tmp_path / "list.yaml"
        path.write_text("- just\n- a\n- list\n")
        with pytest.raises(ValueError, match="must be a mapping"):
            load_shot_labels(path)


SQUARE = ((0.2, 0.2), (0.8, 0.2), (0.8, 0.6), (0.2, 0.6))


class TestCameraProfile:
    def test_derives_crop_rect_from_polygon_bounds(self) -> None:
        profile = CameraProfile(water_polygon=SQUARE, crop_margin=0.0)
        assert profile.crop_rect == pytest.approx((0.2, 0.2, 0.6, 0.4))

    def test_crop_margin_expands_the_rect(self) -> None:
        profile = CameraProfile(water_polygon=SQUARE, crop_margin=0.05)
        x, y, w, h = profile.crop_rect
        assert (x, y) == pytest.approx((0.15, 0.15))
        assert (w, h) == pytest.approx((0.7, 0.5))

    def test_crop_rect_clamps_to_frame(self) -> None:
        """A pool touching the frame edge must not produce a crop outside it."""
        profile = CameraProfile(water_polygon=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)), crop_margin=0.2)
        x, y, w, h = profile.crop_rect
        assert (x, y, w, h) == pytest.approx((0.0, 0.0, 1.0, 1.0))

    def test_rejects_polygon_without_area(self) -> None:
        with pytest.raises(ValidationError, match="at least 3 vertices"):
            CameraProfile(water_polygon=((0.1, 0.1), (0.5, 0.5)))

    def test_rejects_vertex_outside_normalized_range(self) -> None:
        with pytest.raises(ValidationError, match="out of normalized range"):
            CameraProfile(water_polygon=((0.1, 0.1), (1.4, 0.5), (0.2, 0.9)))


class TestCameraReferences:
    def test_shot_without_camera_is_valid(self) -> None:
        """Scraped clips have no stable geometry and must still load."""
        manifest = ShotLabels(shots=(label(),))
        assert manifest.camera_for(manifest.shots[0]) is None

    def test_resolves_named_camera(self) -> None:
        manifest = ShotLabels(
            shots=(label(camera="backyard"),),
            cameras={"backyard": CameraProfile(water_polygon=SQUARE)},
        )
        profile = manifest.camera_for(manifest.shots[0])
        assert profile is not None
        assert profile.water_polygon == SQUARE

    def test_rejects_dangling_camera_reference(self) -> None:
        """A typo must fail loudly rather than silently drop the zone."""
        with pytest.raises(ValidationError, match="unknown cameras"):
            ShotLabels(
                shots=(label(camera="bakyard"),),
                cameras={"backyard": CameraProfile(water_polygon=SQUARE)},
            )
