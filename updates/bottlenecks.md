# ELLA v1 speed bottlenecks

Anchor: mini-demo baseline (`output/baseline`, 4 genes x 5 cells, CPU, 1 process)
= 36.6s total, `nhpp_fit` = 35.6s (97%). Everything below is about `nhpp_fit`,
whose work scales as `genes x 23 fits x iters(<=max_iter) x O(N)` (N = pooled UMI
per gene; 23 = 1 null + 22 beta kernels).

Ranked by expected speedup on a real many-gene panel (the benchmark use case). At
the demo scale actually measured, item 3 dominates the 35.6s; items 1-2 dominate
once genes/transcripts grow.

Tags: `[BI]` bit-identical, `[BC]` behavior-changing. Behavior-preservation is a
hard constraint. Two RNG touch points make naive batching/parallel reordering
non-identical: `torch.rand` in every model `__init__` (`ELLA/ELLA.py:35-36,54`)
and `np.random.uniform` in `compute_pv` (`:928`).

v1 has no per-transcript / per-bin Python loop (its forward is already
vectorized), so v2's two biggest wins do not apply; v1 also has no Lightning and
no built-in parallelism.

## 1. No parallelism at all  `[BI with per-gene seeding]`
`:22` imports `ProcessPoolExecutor`, unused. Gene loop (`:609`) and kernel loop
(`:697`) are both serial; no SLURM/gene-split script (v2 had one). Genes are
independent, so a process pool over genes scales near-linearly with cores: the
largest lever for real panels. `nhpp_fit` already exposes `ig_start`/`ng_demo`
(`:566`) for manual gene-range splitting. Behavior-preserving only if each worker
is seeded from its gene index (current code shares one global RNG).

## 2. 23 sequential Adam fits per gene  `[BC batched / BI parallel]`
Null (`:649-693`) + 22 kernels (`:697-743`), all serial, all on identical data,
differing only in scalar `(a,b)`. Largest single-process structural multiplier.
Kernel-batching (one model, `[22]`-shaped params) is the biggest win but `[BC]`
(stacking reorders the `torch.rand` init draws). Running the 23 fits concurrently
with fixed per-fit seeds is `[BI]`.

## 3. Per-iteration Python / CPU-sync overhead  `[BI]`
Adam loops `:664-689` (null), `:713-736` (kernel). Each iteration pays `.item()`
twice (`:721,728`) for the early-stop test (CPU sync), `.detach().numpy()`
appended to a loss history (`:717,668`) that is kept in memory even under
`save='less'`, and `torch.isnan`. At demo scale (tiny N) this overhead, not the
math, is the 35.6s. Fix: one `.item()` per iter, skip the unused loss history
when `save='less'`.

## 4. Constant work redone every iteration in `model_beta.forward`  `[BI if op order kept]`
`:38-45`. Per iteration recomputes quantities that depend only on data+kernel, not
the trainable params: `BB` (3 `gamma` calls, `:42`), `clamp` (`:43`), two
`torch.pow` N-vectors and `log(c0i*2pi)` (`:44`). Only `exp(A)`, `exp(B)` change.
Precompute them once per fit. Leverage grows with N; small on the demo.

## 5. Config levers  `[BC]`
- `max_iter` (5000 default / 1000 demo, `:713,664`), full batch each iter; null
  forced `>= min_iter=100` (`:672`). Sets the iter multiplier on items 3-4.
- `hpp_solution='numerical'` (`:76`; the docstring's 'analytical' is stale)
  Adam-fits the null though an exact closed form exists (`:627-628,649-652`).
  Switching removes 1 of 23 fits and is arguably more exact.

## 6. Secondary (non-fit), <3% now, scale later  `[BI]`
- `weighted_density_est` (`:864-874`): recomputes `beta.pdf(x,a,b)` per gene; the
  22 curves are constant, precompute once.
- `register_cells` (`:313-378`): pandas-per-cell, ~0.15s/cell, no parallelism.
- `nhpp_prepare` (`:441-487`): per-transcript list build + list concat.

## Recommended order
Land 3 + 4 first (bit-identical, validate against `output/baseline`), then add 1
(gene parallelism with per-gene seeds) for real panels. 2 (kernel-batching) and 5
change outputs, so adopt only after re-validating. Profile one gene first to
confirm the 3-vs-4 split before editing.
