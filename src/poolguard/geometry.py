"""Polygon geometry in normalized [0, 1] frame coordinates.

Dependency-free, so settings, the tracker and the label manifest can share one
definition of what a valid zone is and what "inside" means.
"""

Point = tuple[float, float]
Polygon = tuple[Point, ...]

MIN_POLYGON_POINTS = 3
"""Fewest vertices that enclose an area; below this there is no inside."""

MIN_POLYGON_AREA = 1e-9
"""Below this the vertices are collinear or coincident, to rounding error."""

WHOLE_FRAME: Polygon = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))


def polygon_area(polygon: Polygon) -> float:
    """Unsigned area by the shoelace formula."""
    doubled_area = 0.0
    previous_x, previous_y = polygon[-1]
    for x, y in polygon:
        doubled_area += previous_x * y - x * previous_y
        previous_x, previous_y = x, y
    return abs(doubled_area) / 2


def validate_polygon(polygon: Polygon, name: str = "polygon") -> None:
    """Raise ValueError unless the polygon is a usable zone.

    `name` is the field being validated, so the message points at it.
    """
    if len(polygon) < MIN_POLYGON_POINTS:
        raise ValueError(f"{name} needs at least {MIN_POLYGON_POINTS} vertices, got {len(polygon)}")
    for x, y in polygon:
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError(f"{name} vertex out of normalized range: {(x, y)}")
    if polygon_area(polygon) < MIN_POLYGON_AREA:
        raise ValueError(f"{name} encloses no area: {polygon}")


def point_in_polygon(x: float, y: float, polygon: Polygon) -> bool:
    """True when the point lies inside the polygon (ray casting).

    Cast a ray from the point towards +x and count the edges it crosses: an
    odd count means inside. Works for concave shapes and either vertex order.
    """
    inside = False
    previous_x, previous_y = polygon[-1]
    for current_x, current_y in polygon:
        edge_spans_y = (previous_y > y) != (current_y > y)
        if edge_spans_y:
            slope = (current_x - previous_x) / (current_y - previous_y)
            edge_x_at_y = previous_x + (y - previous_y) * slope
            if x < edge_x_at_y:
                inside = not inside
        previous_x, previous_y = current_x, current_y
    return inside


def depth_inside(x: float, y: float, polygon: Polygon) -> float:
    """Signed distance from the point to the polygon's outline.

    Positive inside, negative outside, zero on the outline. In normalized
    units, so a horizontal and a vertical distance of the same value are
    different numbers of pixels on a non-square frame.
    """
    nearest = min(
        _distance_to_segment(x, y, start, end)
        for start, end in zip(polygon, polygon[1:] + polygon[:1], strict=True)
    )
    return nearest if point_in_polygon(x, y, polygon) else -nearest


def _distance_to_segment(x: float, y: float, start: Point, end: Point) -> float:
    start_x, start_y = start
    dx, dy = end[0] - start_x, end[1] - start_y
    length_squared = dx * dx + dy * dy
    along = (
        0.0 if length_squared == 0 else ((x - start_x) * dx + (y - start_y) * dy) / length_squared
    )
    along = min(max(along, 0.0), 1.0)
    closest_x, closest_y = start_x + along * dx, start_y + along * dy
    return ((x - closest_x) ** 2 + (y - closest_y) ** 2) ** 0.5
