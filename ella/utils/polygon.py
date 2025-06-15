import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class Point2D:
    x: float
    y: float
    relative_position: float = field(default=1.0, compare=False, repr=True)


def polar_angle(center: Point2D, point: Point2D) -> float:
    return math.atan2(point.y - center.y, point.x - center.x) % (2 * math.pi)


def distance(a: Point2D, b: Point2D) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def circular_interpolation(data: List[Optional[float]]) -> List[float]:
    arr = np.array(data, dtype=float)
    n = len(arr)
    if n == 0:
        raise ValueError("data is empty.")

    known = ~np.isnan(arr)
    if not known.any():
        raise ValueError("At least one value must be non-None for interpolation.")
    if known.all():
        return arr.tolist()

    known_indices = np.where(known)[0]
    known_values = arr[known_indices]

    extended_indices = np.concatenate([known_indices - n, known_indices, known_indices + n])
    extended_values = np.concatenate([known_values, known_values, known_values])

    all_indices = np.arange(n)
    interpolated = np.interp(all_indices, extended_indices, extended_values)
    return interpolated.tolist()


def normalize_points_in_polygon(
    polygon: List[Point2D],
    center: Point2D,
    free_points: List[Point2D],
    num_sectors: int,
) -> List[Point2D]:
    sector_step: float = 2 * math.pi / num_sectors
    sectors_bps: List[List[Point2D]] = [[] for _ in range(num_sectors)]
    for bp in polygon:
        sector_id: int = int(polar_angle(center=center, point=bp) // sector_step)
        sectors_bps[sector_id].append(bp)
    sectors_radii: List[List[float]] = [[distance(center, bp) for bp in bps] for bps in sectors_bps]
    sectors_radius: List[Optional[float]] = [max(x) if x else None for x in sectors_radii]
    sectors_radius_interpolated = circular_interpolation(sectors_radius)

    results: List[Point2D] = []
    for fp in free_points:
        dist_cf = distance(center, fp)
        if math.isclose(dist_cf, 0.0, abs_tol=1e-6):
            results.append(Point2D(x=0.0, y=0.0, relative_position=0.0))
        sector_id = int(polar_angle(center=center, point=fp) // sector_step)
        sector_radius: float = sectors_radius_interpolated[sector_id]
        relative_position = dist_cf / sector_radius
        if math.isclose(dist_cf, 0.0, abs_tol=1e-6):
            normalized_x = 0.0
            normalized_y = 0.0
        else:
            normalized_x = (fp.x - center.x) / dist_cf * relative_position
            normalized_y = (fp.y - center.y) / dist_cf * relative_position
        results.append(Point2D(x=normalized_x, y=normalized_y, relative_position=relative_position))
    return results
