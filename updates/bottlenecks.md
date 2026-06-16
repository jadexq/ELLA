# ELLA speed bottlenecks

Ranked by reasoning (how many times each runs x cost per hit). Not yet profiled.
Frequencies below: `genes x kernels (23) x epochs (<=200) x cells x ...`.

## 1. Per-transcript Python loop in `COX.forward`
`ella/models/cox.py:119-124`. Runs `points x cells x epochs x kernels x genes`.
Every molecule is one Python iteration doing tensor `floor` / index / `log`.
Scales with total transcript count; largest multiplier in the pipeline.

## 2. Per-(cell x bin) Python loop in `PolicyNetwork.forward`
`ella/models/cox.py:46-65`. Runs `cells x 20 bins x epochs x kernels x genes`,
each building a separate `torch.distributions.Normal` + `rsample()`. No
vectorization. Hit as often as #1 but without the per-point factor.

## 3. `scipy.stats.beta.pdf` inside the bin loop
`ella/models/cox.py:53`. Same `cells x bins x epochs` frequency, but the result
is **constant** (kernel `a0,b0` + `r_mid` are fixed). Pure waste; should be
computed once per kernel. Also forces a NumPy roundtrip that blocks GPU use.

## 4. 23 sequential trainings per gene
`ella/cli/train.py:91-99`. Null + 22 kernels in a serial Python loop, each a
fresh Lightning `Trainer`. Multiplies everything above by ~23. Only `varphi`
differs across kernels, so the forward could largely be shared.

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
#1, #2, #3 (vectorize the loops, precompute `varphi`): likely an order-of-
magnitude per-fit win. Then #4 (share/parallelize kernels across the 23 fits).
