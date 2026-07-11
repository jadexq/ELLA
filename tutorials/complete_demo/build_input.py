"""
Provenance / regeneration utility for the complete-demo input pickle.

The current ELLA v1 `load_data` requires cell POLYGONS (`cell_poly`: dict
cell_id -> (M, 2) native-frame boundary vertices), not raster `cell_seg` masks.
This script derives `cell_poly` from the raster `cell_seg` using the same alphashape
recipe the demo notebook uses to draw cell outlines (resolution reduced to a 10 px
grid, unique points, `alphashape(pts, 0.1)`) and writes it back into the input pickle
IN PLACE. `cell_seg` is retained, so this is idempotent: it can be re-run against the
shipped pickle to regenerate `cell_poly` at any time.

The demo notebook loads the pre-built pickle directly and does NOT need to run this.
Run in the `ella1` env:  python build_input.py
"""
import os
import pickle
import numpy as np
import pandas as pd
import alphashape
from shapely.geometry import Polygon, MultiPolygon

HERE = os.path.dirname(os.path.abspath(__file__))
PKL = os.path.join(HERE, "input", "complete_demo_data.pkl")
GRID = 10       # px grid for resolution reduction (matches the notebook's `//10 * 10`)
ALPHA = 0.1     # alphashape alpha (matches the notebook)


def seg_to_polygon(seg_c):
    """cell_seg rows for one cell -> (M, 2) boundary vertices via the demo's recipe."""
    x = (seg_c.x.values // GRID) * GRID
    y = (seg_c.y.values // GRID) * GRID
    pts = np.unique(np.stack((x, y)).T, axis=0)
    shp = alphashape.alphashape(pts, ALPHA)
    # alphashape can return a MultiPolygon/GeometryCollection for tricky masks;
    # take the largest-area polygon component so we always get a simple ring.
    if isinstance(shp, MultiPolygon):
        shp = max(shp.geoms, key=lambda g: g.area)
    if not isinstance(shp, Polygon) or shp.is_empty:
        shp = Polygon(pts).convex_hull  # fallback: convex hull of the reduced points
    xs, ys = shp.exterior.xy
    return np.column_stack([np.asarray(xs), np.asarray(ys)])


def main():
    d = pd.read_pickle(PKL)
    gb = d["cell_seg"].groupby("cell", observed=True)

    cell_poly = {}
    for c in d["cells_all"]:
        poly = seg_to_polygon(gb.get_group(c))
        cell_poly[c] = poly
        print(f"  cell {c:>6}: polygon with {poly.shape[0]:3d} vertices")

    d["cell_poly"] = cell_poly
    with open(PKL, "wb") as f:
        pickle.dump(d, f)
    print(f"\nwrote {PKL}")
    print(f"keys: {list(d.keys())}")


if __name__ == "__main__":
    main()
