import math

from ella.utils.polygon import (
    Point2D,
    normalize_points_in_polygon,
    polar_angle,
    circular_interpolation,
)


def test_circular_interpolation() -> None:
    input_data = [None, None, 1.0, None, 3.0, None]
    interpolated = circular_interpolation(input_data)
    assert interpolated == [2.0, 1.5, 1.0, 2.0, 3.0, 2.5]


def visualize_normalized_points() -> None:
    # pylint: disable=import-outside-toplevel
    import matplotlib.pyplot as plt
    import numpy as np

    polygon_points = [
        Point2D(10, 0),
        Point2D(10, 10),
        Point2D(0, 10),
        Point2D(0, 0),
    ]
    center_pt = Point2D(5, 5)
    free_pts = [
        Point2D(6, 5),
        Point2D(5, 6),
        Point2D(4, 4),
        Point2D(9, 5),
        Point2D(1, 1),
    ]
    normalized_points = normalize_points_in_polygon(polygon_points, center_pt, free_pts, 8)

    _fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    ax1 = axes[0]
    polygon_x = [p.x for p in polygon_points] + [polygon_points[0].x]
    polygon_y = [p.y for p in polygon_points] + [polygon_points[0].y]

    ax1.plot(polygon_x, polygon_y, "k-", label="Polygon Boundary")
    ax1.plot(center_pt.x, center_pt.y, "ro", label="Center Point")

    for point in free_pts:
        ax1.plot(point.x, point.y, "bo")
        ax1.text(point.x, point.y, f"({point.x}, {point.y})", fontsize=8, ha="center")

    ax1.set_title("Original Polygon with Free Points")
    ax1.set_xlabel("X-axis")
    ax1.set_ylabel("Y-axis")
    ax1.axis("equal")
    ax1.legend()
    ax1.grid(True)

    ax2 = axes[1]
    theta = np.linspace(0, 2 * np.pi, 100)
    ax2.plot(np.cos(theta), np.sin(theta), "k--", label="Unit Circle")

    for point in normalized_points:
        ax2.plot(point.x, point.y, "bo")
        ax2.text(point.x, point.y, f"({point.x:.2f}, {point.y:.2f})", fontsize=8, ha="center")

    ax2.plot(0, 0, "ro", label="Normalized Center")
    ax2.set_xlim(-1.1, 1.1)
    ax2.set_ylim(-1.1, 1.1)
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_title("Normalized Points in Unit Circle")
    ax2.set_xlabel("Normalized X")
    ax2.set_ylabel("Normalized Y")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize_normalized_points()
