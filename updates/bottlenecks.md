# ELLA v1 speed bottlenecks (Newton-era)

**Status of this file.** Rewritten after the Adam->Newton port of the alternative
fit landed (see `ELLA/ELLA.py` `newton_fit_gene`, and git history on the `ella1`
branch for the removed Adam machinery). The previous version of this file ranked
the *Adam-era* `nhpp_fit` as ~97% of runtime; that is no longer true. The old
anchor profile (`updates/output/baseline/`, mini demo) was deleted, so the ranking
below is **structural** (from code complexity + where the per-transcript / per-cell
work lives), **not** a fresh measured profile. Capture a Newton-era profile on a
real-sized dataset (compute node, `ella1` env) before trusting the ordering for a
specific panel — see "Profiling TODO".

## What changed (why the old ranking is dead)

`nhpp_fit` (`:574`) no longer does 23 sequential Adam optimizations per gene. It now
does: (a) the **analytical closed-form null** (`hatB`, `maxll`, `:623-632`), and (b)
one **bounded-Newton** call `newton_fit_gene` (`:637`) that is **vectorized over all
22 beta kernels** and converges in ~3-11 numpy iterations. No torch, no per-kernel
loop, no early-stop `.item()` sync, no loss history. So the whole former hot path
(old items 2-5) is gone, and the cost has shifted to what the old file called
"secondary" work: data prep, registration, and the p-value assembly.

**Resolved by the port (kept for context, do not re-list as bottlenecks):**
- 23 sequential Adam fits / kernel-batching  -> one vectorized Newton call.
- Per-iteration `.item()` / CPU-sync / loss-history overhead  -> gone (numpy).
- Constant work redone in `model_beta.forward`  -> `newton_fit_gene` precomputes
  the kernel matrix `K`, baseline `w`, and penalties once (`:61-65`).
- `hpp_solution='numerical'` Adam-fitting the null  -> null is analytical (`:631`).
- "No parallelism at all"  -> **now provided externally** by `bridge/ella1.py`
  (gene-sharded `ProcessPoolExecutor`, per-gene seeding `seed_base + g`). `ELLA.py`
  itself is still serial, and its `ProcessPoolExecutor` import (`:19`) is dead.

## Conventions

Tags: `[BI]` bit-identical, `[BC]` behavior-changing. Behavior-preservation is a
hard constraint (the bridge shards genes and seeds per gene, expecting identical
per-gene output regardless of sharding).

RNG touch points in the Newton era (only one is in the fit path):
- `np.random.uniform` in `compute_pv` (`:803`) -- the `T <= 0` boundary atom, drawn
  per (gene, kernel). Seeded per gene by the bridge (`np.random.seed`,
  `bridge/ella1.py:211`). **This is the only RNG that affects results.**
- `bridge/ella1.py:212` still calls `torch.manual_seed` -- **vestigial** now
  (Newton has no torch RNG); it seeds nothing in the fit.
- `:908 random.seed(2024)` (kmeans) and `:1116 np.random.uniform` (plot jitter) are
  outside the fit/pv path.

Newton itself is deterministic (no RNG), so parallel/reordered execution of the fit
is `[BI]` as long as the per-gene `compute_pv` seed is preserved.

## Ranked bottlenecks (structural, real-panel)

### 1. `nhpp_prepare` is quadratic in transcripts-per-cell  `[BI]`
`:449-495`. Triple-nested Python loop (type -> gene -> cell -> transcript) that grows
Python lists with `+`: `r_c = r_c + [rj]*umi` (`:479`) inside the per-transcript
loop, then `r_g = r_g + r_c`, `n_g = n_g + n_c`, `c0_g = c0_g + c0_c` (`:486-488`)
per cell. List concatenation with `+` is O(len) each time, so building a gene's
pooled arrays is O(N^2) in its total UMI. With the fit now cheap, this is the most
likely single largest CPU cost per dataset. Fix: preallocate / `np.repeat(rj, umi)`
+ `np.concatenate` once per gene (or build a flat list and convert once). Pure
data reshaping -> bit-identical.

### 2. `register_cells` per-cell pandas, no parallelism  `[BI / parallelizable]`
`:349-389`. Serial loop over cells; each does `df_gbC.get_group(c).copy()` and
per-cell column math. Now ray-cast (`polygon_boundary_radius` + `searchsorted`,
`:362-371`), correct but still one pandas group per cell. Runs once per dataset in
each bridge worker. Fix: vectorize across cells, or parallelize (the bridge only
parallelizes genes, and registration precedes the gene split). Bit-identical if the
per-cell math is preserved.

### 3. `compute_pv` scalar double-loop over (gene x kernel)  `[BI if RNG order kept]`
`:797-807`. `for ig: for ik:` computes `T = -2*(mll_null - mll_k)` and
`scipy.stats.chi2.cdf(T, 1)` one scalar at a time over genes x 22 kernels. Trivially
vectorizable (matrix `T`, vectorized `chi2.cdf`, `np.where` for the `T<=0` branch).
Was negligible under Adam; visible now. **Careful:** the `T<=0` branch draws
`np.random.uniform` per entry (`:803`); a vectorized rewrite must draw for exactly
the same entries in the same order to stay `[BI]` (otherwise `[BC]`).

### 4. `weighted_density_est` recomputes constant kernel curves per gene  `[BI]`
`:739-746`. Inside the per-gene loop, `beta.pdf(x, a, b)` is evaluated for each of
the 22 kernels on a fixed `x = linspace(0.001, 0.999, 100)` (`:738`). The 22 curves
are constant across genes -> precompute a `(22, 100)` matrix once, then per gene just
do the weighted sum. Bit-identical.

### 5. Cleanup (not speed, but dead weight)  `[BI]`
- Dead imports in `ELLA.py`: `timeit` (`:3`), `ipdb` (`:18`),
  `ProcessPoolExecutor` (`:19`) -- all unused after the port.
- Vestigial `torch.manual_seed` in `bridge/ella1.py:212` (and the torch import that
  exists only to serve it) -- Newton has no torch RNG.

## Recommended order

Land 1 (`nhpp_prepare` O(N^2) -> linear) first: biggest structural lever, and
bit-identical so it validates against a saved run. Then 3 and 4 (both `[BI]`
reshapes/precompute). 2 (`register_cells`) is the next real cost but a larger rewrite;
do it after profiling confirms its share. All of 1-4 are behavior-preserving; none
touch the fit or the seeded boundary atom.

## Profiling TODO

The old mini-demo anchor was deleted and, at 4 genes x 5 cells, was dominated by
fixed overhead anyway. Before optimizing, capture a Newton-era stage breakdown
(`register_cells` / `nhpp_prepare` / `nhpp_fit` / `weighted_density_est` /
`compute_pv`) on a **real-sized** dataset (e.g. a simulation scaffold panel, 100
genes x ~116 cells) in the `ella1` env on a compute node, so the ranking above is
backed by measured shares rather than code structure. Expectation to confirm:
`nhpp_fit` is now a small fraction and items 1-2 dominate.
