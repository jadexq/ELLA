# ELLA speed bottlenecks

Ranked by reasoning (how many times each runs x cost per hit). Not yet profiled.
Frequencies below: `genes x kernels (23) x epochs (<=200) x cells x ...`.

## 1. Per-transcript Python loop in `COX.forward`  [FIXED]
`ella/models/cox.py:119-124`. Runs `points x cells x epochs x kernels x genes`.
Every molecule is one Python iteration doing tensor `floor` / index / `log`.
Scales with total transcript count; largest multiplier in the pipeline.

**Fix applied.** Replaced the per-molecule loop with one vectorized clamp +
gather + sum: `bin_idx = clamp(floor(cell_points * n_bins), max=n_bins-1)`,
`sum_log = log(lam_vec[bin_idx] + 1e-10).sum()`. Same math (same binning, clamp,
`+1e-10`), N python iterations -> ~3 tensor ops.

**Result** (4-gene / 5-cell mini demo, seeded, `updates/run_fit.py`):
lam(r) bit-identical to baseline (`max|dlam| = 0` all genes);
total 303.7s -> 205.7s = **1.48x (32% faster)**. The win is modest here only
because the demo has few transcripts/cell, so #2 (per-`cell x bin` loop)
dominates this scale; #1's payoff grows with transcripts/cell on real panels.

## 2. Per-(cell x bin) Python loop in `PolicyNetwork.forward`  [FIXED]
`ella/models/cox.py:46-65`. Runs `cells x 20 bins x epochs x kernels x genes`,
each building a separate `torch.distributions.Normal` + `rsample()`. No
vectorization. Hit as often as #1 but without the per-point factor.

**Fix applied (with #3).** `forward` now builds `mu [n_cells, n_bins]` and
`sigma [n_bins]` in one tensor op; `get_lambda_star_i` draws one batched
`Normal(mu, sigma).rsample()` instead of `cells x bins` scalar distributions;
`COX.forward` indexes the `[n_cells, n_bins]` lam / log_prob matrices (the
`>0 points` drop and the #1 per-transcript gather are unchanged).

## 3. `scipy.stats.beta.pdf` inside the bin loop  [FIXED]
`ella/models/cox.py:53`. Same `cells x bins x epochs` frequency, but the result
is **constant** (kernel `a0,b0` + `r_mid` are fixed). Pure waste; should be
computed once per kernel. Also forces a NumPy roundtrip that blocks GPU use.

**Fix applied (with #2).** `r_mid` and `varphi = beta.pdf(r_mid, a0, b0)` are
precomputed once in `PolicyNetwork.__init__` as non-persistent buffers (out of
`state_dict`, which `on_train_end` assumes holds only scalar params). The scipy
call and NumPy roundtrip are gone from `forward`.

**Result for #2+#3** (4-gene / 5-cell mini demo, seeded, `updates/run_fit.py`):
total **205.7s -> 48.4s = 4.25x** incremental over #1; **303.7s -> 48.4s = 6.28x
(84% faster)** cumulative vs the original. #2/#3 is the dominant win at this scale,
as expected (few transcripts/cell, so #1's per-point factor is small here).

Correctness: unlike #1 this is NOT bit-identical, because the batched `rsample()`
consumes the RNG stream differently than `cells x bins` scalar draws, so the
seeded stochastic-gradient trajectory differs. Validated two ways:
`updates/test_vectorize_equiv.py` proves the math is bit-identical *given the same
noise* (`mu` 3.8e-6 float32, `sigma`/`log_prob` exact, per-cell reward 4.8e-7);
and a two-seed run shows base@42-vs-opt@42 lam(r) divergence (1-48% of curve span)
is within same-code seed-to-seed spread (opt@42-vs-opt@99 = 14-60%), i.e. seed
noise, not bias. Overlay in `updates/output/compare_lam_curves.png`.

## 4. 23 sequential trainings per gene
`ella/cli/train.py:91-99`. Null + 22 kernels in a serial Python loop, each a
fresh Lightning `Trainer`. Multiplies everything above by ~23. Only `varphi`
differs across kernels, so the forward could largely be shared.

**Status: DEFERRED (2026-06-16).** Not implementing now. Hard constraint: any #4
change must NOT alter ELLA pipeline outputs. Kernel-batching (one model, `varphi`
as a `[23, ...]` axis) is the largest speedup but is NOT bit-identical (a batched
`rsample()` reorders the RNG stream, same as #2), so it would change seeded
results and is off the table unless outputs are re-validated. The
behavior-preserving levers are (a) parallelizing the 23 independent fits across
cores, each with its own seeding, and (b) the #6 overhead fixes (skip the per-fit
best-checkpoint disk reload, cheaper logging). Practical urgency is low because
gene-level parallelism (#7) already saturates a node: the `null_ng1000_mu5` run
fit 100 genes in ~one gene's wall time (~2-3 min/gene, 116 cells / ~5
transcripts/cell). **Revisit only if per-gene speed is still insufficient;
profile first** (single-gene, per-fit setup/IO vs epoch compute) to decide
parallelize vs #6 before touching any code.

## 5. Up to 200 epochs, full dataset per epoch
`configs/mini_demo.yaml:68` (`max_epochs=200`). Single batch, `num_workers=0`,
CPU by default. Sets the epoch multiplier for #1-#3; early-stop patience 10
helps only if it converges early.

## 6. Per-fit Lightning / IO overhead
`ella/cli/train.py:71` (`log_every_n_steps=1`) + `ella/models/cox.py:165`
(best-checkpoint reload from disk per fit). Small individually, paid 23x per gene.

## 7. Parallelism only across genes
`scripts/demo/mimi_demo/run_mini_demo.sh`. SLURM array over `gene_idx` is the
only scaling; everything within a gene is serial.

## Highest leverage
#1, #2, #3 DONE: 6.28x on the mini demo (the per-fit forward is now vectorized
and scipy-free). #4 (the 23 kernel fits per gene) is DEFERRED by choice, not for
lack of value: its biggest form (kernel-batching) is not bit-identical, and
gene-level parallelism (#7) already covers the common case. Revisit #4/#5/#6
(behavior-preserving parallelize, epoch budget, per-fit Lightning/IO overhead)
only if per-gene speed is still insufficient.
