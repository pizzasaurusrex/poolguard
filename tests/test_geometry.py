"""Tests for normalized-coordinate polygon geometry."""

import pytest

from poolguard.geometry import WHOLE_FRAME, point_in_polygon, validate_polygon

SQUARE = ((0.2, 0.2), (0.6, 0.2), (0.6, 0.6), (0.2, 0.6))

# An L: the full left column plus the bottom row. The top-right quarter is a
# notch, inside the bounding box but outside the shape, like deck inside the
# bounding box of a kidney-shaped pool.
L_SHAPE = ((0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (1.0, 0.5), (1.0, 1.0), (0.0, 1.0))


class TestPointInPolygon:
    def test_point_inside_square(self) -> None:
        assert point_in_polygon(0.4, 0.4, SQUARE) is True

    @pytest.mark.parametrize(("x", "y"), [(0.1, 0.4), (0.7, 0.4), (0.4, 0.1), (0.4, 0.7)])
    def test_point_outside_square(self, x: float, y: float) -> None:
        assert point_in_polygon(x, y, SQUARE) is False

    def test_point_in_the_notch_of_a_concave_shape_is_outside(self) -> None:
        assert point_in_polygon(0.75, 0.25, L_SHAPE) is False

    @pytest.mark.parametrize(("x", "y"), [(0.25, 0.25), (0.25, 0.75), (0.75, 0.75)])
    def test_points_in_the_arms_of_a_concave_shape_are_inside(self, x: float, y: float) -> None:
        assert point_in_polygon(x, y, L_SHAPE) is True

    def test_vertex_order_does_not_matter(self) -> None:
        reversed_square = tuple(reversed(SQUARE))
        assert point_in_polygon(0.4, 0.4, reversed_square) is True
        assert point_in_polygon(0.1, 0.4, reversed_square) is False

    def test_whole_frame_contains_any_interior_point(self) -> None:
        assert point_in_polygon(0.01, 0.99, WHOLE_FRAME) is True


class TestValidatePolygon:
    def test_accepts_a_triangle(self) -> None:
        validate_polygon(((0.0, 0.0), (1.0, 0.0), (0.5, 1.0)))

    def test_rejects_fewer_than_three_vertices(self) -> None:
        with pytest.raises(ValueError, match="at least 3"):
            validate_polygon(((0.0, 0.0), (1.0, 1.0)))

    def test_rejects_a_vertex_outside_the_frame(self) -> None:
        with pytest.raises(ValueError, match="out of normalized range"):
            validate_polygon(((0.0, 0.0), (1.5, 0.0), (0.5, 1.0)))

    def test_rejects_a_polygon_enclosing_no_area(self) -> None:
        with pytest.raises(ValueError, match="encloses no area"):
            validate_polygon(((0.2, 0.2), (0.5, 0.2), (0.8, 0.2)))

    def test_error_names_the_field_it_was_given(self) -> None:
        with pytest.raises(ValueError, match="water_polygon"):
            validate_polygon(((0.0, 0.0), (1.0, 1.0)), name="water_polygon")
