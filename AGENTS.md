# AGENTS.md — running ELLA

Instructions for a coding agent driving **ELLA** on a user's behalf. Scope: how to
install ELLA and run an analysis. Follow the rules literally; they encode the current
API and the places stale ELLA knowledge goes wrong.

## What ELLA is

ELLA detects **subcellular RNA localization**: for each gene in each cell type it tests
whether transcripts are non-uniformly distributed along the radial axis of the cell
(nuclear center → membrane) and estimates the localization pattern. Input is
single-molecule spatial data (molecule coordinates + cell boundary polygons); output is,
per gene, a significance p-value and an intensity-vs-relative-position curve.

This is **ELLA Newton** (branch `main`, the current and recommended version): the
alternative model is fit with a deterministic bounded-Newton solver. Two older variants
(**ELLA Adam**, the original release; **ELLA-Cox**, a Cox-process rewrite) are archived on
other branches — do not use them unless the user explicitly asks.

## Install

Requires **Python ≥ 3.9**. All dependencies (numpy, pandas, scipy, scikit-learn,
matplotlib, tqdm, alphashape) install automatically. There is **no torch and no R/rpy2** —
if you see instructions to install those, they describe an old version; ignore them.

```bash
pip install "git+https://github.com/jadexq/ELLA.git"   # or, from a local clone:
pip install -e .                                        # editable/dev install
```

Verify the install:

```python
from ELLA import ELLA        # `from ELLA.ELLA import ELLA` also works
ELLA()                       # constructs with defaults, no error
```

## Run recipe

ELLA is a plain Python library — **there is no command-line interface.** Run the pipeline
by calling the object's methods in order. This is the whole analysis:

```python
from ELLA import ELLA

ella = ELLA(dataset='my_run')          # names the run
ella.load_data(data_path='input/my_data.pkl')   # or load_data(data_dict=...)

ella.register_cells()   # map each cell to the unit disk (geometry only, gene-independent)
ella.nhpp_prepare()     # prepare per-gene data for fitting (gene-independent)
ella.nhpp_fit()         # fit the NHPP model per gene (bounded Newton; the compute step)
ella.weighted_density_est()   # model-averaged intensity curve per gene
ella.compute_pv()       # likelihood-ratio test -> per-gene p-values (+ BY-FDR)

# optional: group the significant genes into localization patterns
ella.pattern_clustering()      # picks K by an elbow heuristic
ella.pattern_labeling(K=5)     # assign each sig gene a pattern label
```

Each stage writes a checkpoint pickle into `output/` (created if missing):
`df_registered.pkl`, `df_nhpp_prepared.pkl`, `nhpp_fit_results.pkl`, `lam_est.pkl`,
`pv_est.pkl`. To skip a completed stage, reload its checkpoint instead of recomputing:

```python
ella.load_registered_cells(registered_path='output/df_registered.pkl')
ella.load_nhpp_prepared(prepared_path='output/df_nhpp_prepared.pkl')
ella.load_nhpp_fit_results(res_path='output/nhpp_fit_results.pkl')
```

The constructor knobs are rarely changed; the two Newton-specific ones are
`newton_max_iter=100` and `newton_ftol=1e-12`. There are **no** `adam_*`, `max_iter`,
`min_iter`, `hpp_solution`, or `optimizer` arguments — passing them raises `TypeError`.

## Input contract

`load_data` takes a dict (or a pickle path to one) with these keys:

| key | type | required | meaning |
|---|---|---|---|
| `expr` | DataFrame | yes | one row per molecule (see columns below) |
| `cell_poly` | dict `cell_id → (M,2) array` | **yes** | cell boundary vertices, native coords |
| `types` | list | yes | cell type names |
| `cells` | dict `type → [cell_id]` | yes | cells in each type |
| `genes` | dict `type → [gene]` | yes | genes in each type |
| `cells_all` | list | yes | all cell ids |
| `nucleus_seg` | DataFrame | no | nucleus outline, **visualization only** |

`expr` columns used: `x`, `y` (molecule coords), `umi` (count), `centerX`, `centerY`
(cell/nuclear center), `type`, `cell`, `gene`, `sc_total` (per-cell total count; equal for
all rows of a cell).

Rules:
- **`cell_poly` is mandatory.** If it is missing, `load_data` raises `KeyError`. ELLA
  registers each cell by ray-casting these polygons.
- A raster `cell_seg` point cloud is **not consumed** by ELLA (viz-only, optional). Do not
  rely on it as the cell geometry — build `cell_poly` polygons.
- To see the exact expected shape, load the shipped example
  `tutorials/mini_demo/input/mini_demo_data.pkl` (5 cells, 4 genes) and inspect its keys.
  Field-by-field detail is at https://jadexq.github.io/ELLA/inputs.html.

## Reading results

Results live on the object as dicts keyed by cell type:

- `ella.pv_fdr_tl[type]` — per-gene BY-FDR-adjusted p-values. **A gene is significant when
  `pv_fdr_tl[type][i] <= 0.05`** (`ella.sig_cutoff`), i.e. it has a non-uniform
  subcellular pattern. Gene order matches `ella.gene_list_dict[type]`.
- `ella.weighted_lam_est[type]` — per gene, the estimated intensity as a length-100 curve
  over relative position, where **0 = nuclear center and 1 = cell membrane**. A peak near 0
  = nuclear, near 1 = membrane-localized.
- `ella.scores[type]` — per gene, the relative position of the intensity peak.
- `ella.labels_dict[type]` — pattern cluster label per gene (after `pattern_labeling`).
- Also available: `pv_cauchy_tl` (pre-FDR combined p), `pv_raw_tl`, `ts_tl` (LRT stats).

## Pitfalls (do not get these wrong)

- No CLI. Drive ELLA from Python, not a shell command.
- Do not pass Adam-era arguments (`adam_*`, `max_iter`, `hpp_solution`, ...) — they were
  removed and will error.
- `cell_poly` is required; raster masks are not a substitute.
- The reload methods take `*_path=` keywords (`registered_path`, `prepared_path`,
  `res_path`), not a bare positional path.
- Every result is keyed by cell type — index with `[type]` before indexing by gene.

## Smoke test

Before touching the user's data, confirm the environment end-to-end. From
`tutorials/mini_demo/`:

```python
from ELLA import ELLA
ella = ELLA(dataset='smoke')
ella.load_data(data_path='input/mini_demo_data.pkl')
ella.register_cells(); ella.nhpp_prepare(); ella.nhpp_fit()
ella.weighted_density_est(); ella.compute_pv()
assert 'fibroblast' in ella.pv_fdr_tl and len(ella.pv_fdr_tl['fibroblast']) == 4
print('ELLA OK:', ella.pv_fdr_tl['fibroblast'])
```

If that runs and prints four p-values, ELLA is installed correctly.
