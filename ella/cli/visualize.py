import argparse
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel

from ella.utils.polygon import Point2D, polar_angle


class VisualizationConfig(BaseModel):
    data_dir: str
    cell_id: str
    gene_id: str

    @property
    def cells_polygon_path(self) -> str:
        return os.path.join(self.data_dir, "cells_polygon.json")

    @property
    def cells_center_path(self) -> str:
        return os.path.join(self.data_dir, "cells_center.json")

    @property
    def cells_point_infos_path(self) -> str:
        return os.path.join(self.data_dir, "cells_point_infos.json")


def parse_args() -> VisualizationConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data-dir", type=str, required=True)
    parser.add_argument("-c", "--cell-id", type=str, required=True)
    parser.add_argument("-g", "--gene-id", type=str, required=True)
    args = parser.parse_args()
    return VisualizationConfig(**vars(args))


def main() -> None:
    cfg: VisualizationConfig = parse_args()

    with open(cfg.cells_polygon_path, "r", encoding="utf-8") as f:
        cells_polygon_dict: Dict[str, List[Dict]] = json.load(f)
    cells_polygon: Dict[str, List[Point2D]] = {}
    for cell_id, polygon in cells_polygon_dict.items():
        cells_polygon[cell_id] = [Point2D(x=bp["x"], y=bp["y"]) for bp in polygon]
    polygon_points: List[Point2D] = cells_polygon[cfg.cell_id]

    with open(cfg.cells_center_path, "r", encoding="utf-8") as f:
        cells_center_dict: Dict[str, Dict] = json.load(f)
    cells_center: Dict[str, Point2D] = {}
    for cell_id, center in cells_center_dict.items():
        cells_center[cell_id] = Point2D(x=center["x"], y=center["y"])
    center_pt: Point2D = cells_center[cfg.cell_id]

    polygon_points = sorted(polygon_points, key=lambda bp: polar_angle(center_pt, bp))

    with open(cfg.cells_point_infos_path, "r", encoding="utf-8") as f:
        cells_point_infos: Dict[str, List[Dict]] = json.load(f)
    free_points: List[Point2D] = []
    normalized_points: List[Point2D] = []
    for point_info in cells_point_infos[cfg.cell_id]:
        if point_info["gene"] == cfg.gene_id:
            free_points.append(Point2D(x=point_info["x"], y=point_info["y"]))
            normalized_points.append(Point2D(x=point_info["normalized_x"], y=point_info["normalized_y"]))

    _fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    ax1 = axes[0]
    polygon_x = [p.x for p in polygon_points] + [polygon_points[0].x]
    polygon_y = [p.y for p in polygon_points] + [polygon_points[0].y]

    ax1.plot(polygon_x, polygon_y, "k-", label="Polygon Boundary")
    ax1.plot(center_pt.x, center_pt.y, "ro", label="Center Point")

    for point in free_points:
        ax1.plot(point.x, point.y, "bo")
        # ax1.text(point.x, point.y, f"({point.x}, {point.y})", fontsize=8, ha="center")

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
        # ax2.text(point.x, point.y, f"({point.x:.2f}, {point.y:.2f})", fontsize=8, ha="center")

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
    plt.savefig(f'plot_{cfg.gene_id}_{cfg.cell_id}.png', dpi=300, bbox_inches='tight') 
    plt.show()
