import argparse
import json
import math
import os
import pickle
import pandas as pd
from collections import defaultdict
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import DefaultDict, Dict, List, Set, Tuple

from pydantic import BaseModel

from ella.utils.polygon import Point2D, normalize_points_in_polygon


class PreparationConfig(BaseModel):
    input_path: str
    output_dir: str

    @property
    def out_training_data_path(self) -> str:
        return os.path.join(self.output_dir, "training_data.jsonl")

    @property
    def out_cells_polygon_path(self) -> str:
        return os.path.join(self.output_dir, "cells_polygon.json")

    @property
    def out_cells_center_path(self) -> str:
        return os.path.join(self.output_dir, "cells_center.json")

    @property
    def out_cells_point_infos_path(self) -> str:
        return os.path.join(self.output_dir, "cells_point_infos.json")


def parse_args() -> PreparationConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-path", type=str, required=True)
    parser.add_argument("-o", "--output-dir", type=str, required=True)
    args = parser.parse_args()
    return PreparationConfig(**vars(args))


def load_raw_data(
    data_path: str,
) -> Tuple[Dict[str, List[Point2D]], Dict[str, Point2D], Dict[str, int], Dict[str, List[Dict]]]:
    print(f"loading {data_path}")
    # with open(data_path, "rb") as f:
    # raw_data = pickle.load(f)
    raw_data = pd.read_pickle(data_path)
    df_cell_seg = raw_data["cell_seg"] 
    df_expr = raw_data["expr"] 

    if pd.api.types.is_categorical_dtype(df_expr["cell"]):
        df_expr["cell"] = df_expr["cell"].cat.remove_unused_categories()

    cells_polygon: Dict[str, List[Point2D]] = {}
    grouped = df_cell_seg.groupby("cell")
    for cell_id, group in grouped:
        points = [Point2D(x=float(row["x"]), y=float(row["y"])) for _, row in group.iterrows()]
        cells_polygon[cell_id] = points

    cells_center: Dict[str, Point2D] = {}
    cells_sc_total: Dict[str, int] = {}
    cells_point_infos: DefaultDict[str, List[Dict]] = defaultdict(list)
    grouped = df_expr.groupby("cell", observed=True)
    for cell_id, group in grouped:
        unique_centers = group[["centerX", "centerY"]].drop_duplicates()
        assert len(unique_centers) == 1
        center_row = unique_centers.iloc[0]
        cells_center[cell_id] = Point2D(x=float(center_row["centerX"]), y=float(center_row["centerY"]))

        unique_sc_totals = group[["sc_total"]].drop_duplicates()
        # assert len(unique_sc_totals) == 1
        sc_total_row = unique_sc_totals.iloc[0]
        cells_sc_total[cell_id] = int(sc_total_row["sc_total"])

        for _, row in group.iterrows():
            cells_point_infos[cell_id].append(row.to_dict())

    return cells_polygon, cells_center, cells_sc_total, dict(cells_point_infos)


def update_point_infos(point_infos: List[Dict], polygon: List[Point2D], center: Point2D) -> List[Dict]:
    free_points: List[Point2D] = [Point2D(x=v["x"], y=v["y"]) for v in point_infos]
    normalized_points = normalize_points_in_polygon(
        polygon=polygon,
        center=center,
        free_points=free_points,
        num_sectors=40,  # <<<!!! 360
    )
    updated_point_infos: List[Dict] = []
    for point_info, normalized_point in zip(point_infos, normalized_points):
        updated_point_infos.append(
            {
                **point_info,
                "normalized_x": normalized_point.x,
                "normalized_y": normalized_point.y,
                "relative_position": normalized_point.relative_position,
            }
        )
    return updated_point_infos


def get_genes_info(cells_point_infos: Dict[str, List[Dict]], cells_sc_total: Dict[str, int]) -> Dict[str, Dict]:
    gene_id_to_cell_ids: DefaultDict[str, Set[str]] = defaultdict(set)
    for cell_id, point_infos in cells_point_infos.items():
        for point_info in point_infos:
            gene_id_to_cell_ids[point_info["gene"]].add(cell_id)
    gene_infos: DefaultDict[str, Dict] = defaultdict(dict)
    for gene_id, cell_ids in gene_id_to_cell_ids.items():
        cells: List[Dict] = []
        gene_num_points: float = 0.0
        sum_sc_total: float = 0.0
        for cell_id in cell_ids:
            points = []
            for row in cells_point_infos[cell_id]:
                if row['gene'] == gene_id:
                    for i in range(row['umi']):
                        points.append(row['relative_position'])
            gene_num_points = gene_num_points + len(points)
            # sum_sc_total = sum_sc_total + cells_sc_total[cell_id] / 1000
            cells.append(
                {
                    "cell_id": cell_id,
                    "points": points, 
                    # "sc_total": cells_sc_total[cell_id],
                    "sc_total": len(points),
                }
            )
        gene_infos[gene_id]["gene_id"] = gene_id
        gene_infos[gene_id]["cells"] = cells
        # gene_infos[gene_id]["lam_null"] = gene_num_points / (math.pi * sum_sc_total)
        gene_infos[gene_id]["lam_null"] = gene_num_points / (math.pi * gene_num_points)
    return dict(gene_infos)


def save_results(
    cfg: PreparationConfig,
    gene_infos: Dict[str, Dict],
    cells_polygon: Dict[str, List[Point2D]],
    cells_center: Dict[str, Point2D],
    cells_point_infos: Dict[str, List[Dict]],
) -> None:
    print(f"saving results to {cfg.output_dir}")

    gene_id_point_count: List[Tuple[str, int]] = []
    for gene_id, gene_info in gene_infos.items():
        total_counts_per_gene = 0
        for cell_info in gene_info["cells"]:
            total_counts_per_gene = total_counts_per_gene + len(cell_info["points"])
        gene_infos[gene_id]["total_counts_per_gene"] = total_counts_per_gene
        gene_id_point_count.append((gene_id, total_counts_per_gene))
    gene_id_point_count = sorted(gene_id_point_count, key=lambda x: x[1], reverse=True)

    output_path = Path(cfg.out_training_data_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for gene_id, _ in gene_id_point_count:
            f.write(f"{json.dumps(gene_infos[gene_id])}\n")

    output_path = Path(cfg.out_cells_polygon_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cells_polygon_dict: Dict[str, List[Dict]] = {}
    for cell_id, polygon in cells_polygon.items():
        cells_polygon_dict[cell_id] = [{"x": bp.x, "y": bp.y} for bp in polygon]
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(cells_polygon_dict, f, indent=4)

    output_path = Path(cfg.out_cells_center_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cells_center_dict: Dict[str, Dict] = {}
    for cell_id, center in cells_center.items():
        cells_center_dict[cell_id] = {"x": center.x, "y": center.y}
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(cells_center_dict, f, indent=4)

    output_path = Path(cfg.out_cells_point_infos_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(cells_point_infos, f, indent=4)


def main() -> None:
    cfg: PreparationConfig = parse_args()
    cells_polygon, cells_center, cells_sc_total, cells_point_infos = load_raw_data(cfg.input_path)
    cell_ids: List[str] = list(cells_center.keys())

    updated_cells_point_infos: Dict[str, List[Dict]] = {}
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures: List[Future] = []
        for cell_id in cell_ids:
            print(f"submitted {cell_id=}")
            futures.append(
                executor.submit(
                    update_point_infos,
                    cells_point_infos[cell_id],
                    cells_polygon[cell_id],
                    cells_center[cell_id],
                )
            )
        for future in as_completed(futures):
            result = future.result()
            cell_id = result[0]["cell"]
            print(f"completed {cell_id=}")
            updated_cells_point_infos[cell_id] = result

    gene_infos: Dict[str, Dict] = get_genes_info(
        cells_point_infos=updated_cells_point_infos,
        cells_sc_total=cells_sc_total,
    )

    save_results(
        cfg=cfg,
        gene_infos=gene_infos,
        cells_polygon=cells_polygon,
        cells_center=cells_center,
        cells_point_infos=updated_cells_point_infos,
    )


if __name__ == "__main__":
    main()
