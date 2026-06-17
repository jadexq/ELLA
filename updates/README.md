# ELLA1 updates

Scratch space for planning and testing changes to the **v1** ELLA codebase
(`competing_methods/ELLA1/`, `ella1` branch). Mirrors `competing_methods/ELLA2/updates/`.

Use this for:
- `bottlenecks.md` / design notes for proposed v1 changes
- equivalence / regression tests that prove a change does not alter v1 outputs
- benchmark + comparison scripts (baseline vs candidate)
- `output/` for run artifacts (gitignored: `_work/`, `run.log`)

## Environment

Run under conda env `ella1` (Python 3.10, torch 1.13.0, rpy2 3.5.6 + R 4.6.0):

```
conda activate ella1   # activation hook sets LD_LIBRARY_PATH so rpy2 can load R
```

v1 is the monolithic `ELLA/ELLA.py` implementation and depends on R via rpy2,
unlike the optimized v2 in `ELLA2/`.

## Baseline speed: v1 vs v2 (mini demo)

Same mini demo (4 genes x 5 cells, 23 fits/gene = null + 22 kernels, CPU,
single process). v1 from `output/baseline/timing.json`; v2 from
`../../ELLA2/updates/output/{baseline,optimized}/timing.json`.

| run | fit time (4 genes) | per gene | total |
|---|---|---|---|
| **ELLA1 (v1)** | **35.6s** | ~8.9s | 36.6s |
| ELLA2 baseline (unoptimized v2) | 303.3s | ~75.9s | 303.7s |
| ELLA2 optimized (current `main`) | 48.0s | ~12.0s | 48.4s |

On this toy data v1 is a touch faster than even the optimized v2 (~1.35x) and
about 8.5x faster than the unoptimized v2 baseline. v1's torch forward is already
vectorized over transcripts and it has no Lightning/Hydra per-fit overhead (no
fresh `Trainer` per kernel, no checkpoint reload), which is most of the gap.

Caveats: (1) v1 and v2 fit different models (v1 = a 2-parameter beta intensity;
v2 = the over-dispersed COX model with per-iteration policy-network sampling), so
this is wall-clock on identical toy input, not a fair algorithmic comparison.
(2) The runs may have been on different nodes, so treat the absolute seconds as
approximate and the ratios as order-of-magnitude.
