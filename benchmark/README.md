# ELLA Newton vs ELLA Adam benchmark

Compares the current shipped ELLA (bounded-Newton alternative fit) against the
original ELLA-v1 optimizer (per-kernel Adam) on two 100-gene semi-synthetic
datasets. The two arms differ **only in the alternative-model fit** — registration
(ray-cast), `nhpp_prepare`, `compute_pv`, `weighted_density_est`, and the p-value /
FDR path are byte-identical — so any difference is attributable to the optimizer.

Not part of the shipped ELLA package; benchmark-internal.

## Arms

- **ELLA Newton** — the shipped `ELLA/ELLA.py` (installed package).
- **ELLA Adam** — `ella_adam/ELLA.py`: a copy of the shipped file with **only** the
  `nhpp_fit` alternative fit reverted to the original ELLA-v1 Adam optimizer
  (restored from git `d96eb76`: `model_beta`/`loss_ll`, the `adam_*` constructor
  args, and the per-kernel Adam loop with adaptive-LR / delta-loss early stopping).
  Original ELLA defaults (`max_iter=5000`, lr ∈ [1e-3, 1e-2]). Loaded by file path,
  so it never shadows the installed package.

## Datasets (xenium scaffold, 116 cells, KL=0.02, µ=5)

- **null2** — uniform density in the registration frame (calibrated null; a correct
  method returns uniform p-values). First 100 genes of `null2_ng1000_mu5`.
- **cyto_symmetric** — pure-radial planted pattern peaking mid-cytoplasm (r≈0.67);
  `cyto_symmetric_skl0.02_ng100_mu5` (all 100 genes). A fair target for ELLA's
  radial-only model. Planted truth curve precomputed to `output/cyto_truth.csv`.

## Run

```bash
# ella1 conda env; N_JOBS defaults to 64 (set to available cores)
N_JOBS=32 conda run -n ella1 --no-capture-output python -u benchmark/step1_null2.py
N_JOBS=32 conda run -n ella1 --no-capture-output python -u benchmark/step2_cyto_symmetric.py
```

Both arms register/prepare once, then fit genes sharded across `N_JOBS` processes
with per-gene seeding (BLAS pinned to 1 thread/worker). Heavy intermediates go to
`/scratch/.../tmp_ella1_eval/benchmark/`; only the light outputs below land here.

## Metrics

- **null2**: `resources.csv` (wall-clock, cores, peak memory per arm) + `null2_qq.png`
  (uniform QQ of raw Cauchy p-values, both arms overlaid — calibration check).
- **cyto_symmetric**: `resources.csv` + `cyto_power.csv` (power = fraction with
  BH-FDR < 0.05) + `cyto_kld.png`/`cyto_kld.csv` (per-gene profile accuracy,
  KL(true ‖ estimated radial profile) in nats; each length-100 curve normalized to
  a pmf on ELLA's `linspace(0.001,0.999,100)` grid).

## Files

| file | contents |
|---|---|
| `ella_adam/ELLA.py` | Adam variant (shipped file, fit-only reverted) |
| `runner.py` | selectable-ELLA run wrapper over `bridge.ella1` + peak-RSS sampler |
| `step1_null2.py` | null2 run + QQ |
| `step2_cyto_symmetric.py` | cyto run + power + KLD |
| `output/resources.csv` | runtime + compute, both experiments/arms |
| `output/null2_qq.png` | calibration QQ |
| `output/cyto_power.csv`, `cyto_kld.png`, `cyto_kld.csv` | power + accuracy |
| `output/cyto_truth.csv` | planted cyto_symmetric radial profile |

## Results

Run: 100 genes, 32 cores, node `arseven` (login node; cluster was saturated).
Newton wins on every axis — faster, better calibrated-or-equal, higher power, more
accurate — consistent with Newton finding the true global optimum while Adam
under-optimizes.

**Runtime + compute** (`resources.csv`)

| experiment | method | wall (s) | peak mem (GB) |
|---|---|---|---|
| null2 | Newton | 28.2 | 11.9 |
| null2 | Adam | 171.6 | 12.5 |
| cyto_symmetric | Newton | 11.5 | 10.9 |
| cyto_symmetric | Adam | 149.0 | 11.3 |

Newton is ~6× faster on null2 and ~13× faster on cyto_symmetric; the per-gene fit
itself is far faster (Adam runs ~1.5 s/gene of optimization vs Newton's near-instant
solve), but the shared one-time register+prepare compresses the wall-clock ratio.
Peak memory is essentially equal across arms (dominated by the 32 worker processes,
each carrying a numpy/torch interpreter), so the speedup is free. Wall-clock varies
run-to-run on the shared node; Adam (~150-170 s) is the stable, fit-bound arm.

**null2 calibration** (`null2_qq.png`, raw Cauchy p): neither arm inflates —
fraction of raw p < 0.05 is Newton 0.04, Adam 0.02; both hug/under the diagonal.

**cyto_symmetric power** (`cyto_power.csv`, BH-FDR < 0.05): **Newton 0.97, Adam 0.91.**

**cyto_symmetric accuracy** (`cyto_kld.png`/`.csv`, median KL(true ‖ est) in nats):
**Newton 0.0026, Adam 0.0095** — Newton recovers the planted radial profile ~3.6×
more accurately (median), with a tighter spread.
