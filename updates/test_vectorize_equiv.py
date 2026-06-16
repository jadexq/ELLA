"""
Math-equivalence micro-test for bottleneck #2/#3 vectorization.

Compares the NEW vectorized PolicyNetwork.forward / COX path against a from-scratch
reimplementation of the ORIGINAL per-(cell, bin) loop, BOTH fed the same parameters
and the same injected eps noise. Isolates "did the math change" from "RNG stream
order changed" (the latter is expected once rsample() is batched).

Run from repo root:  conda run --no-capture-output -n ella python updates/test_vectorize_equiv.py
"""
import math

import torch
import torch.nn.functional as F
from scipy.stats import beta as beta_dist

from ella.models.cox import PolicyNetwork
from ella.options import KernelParam, ModelConfig

torch.manual_seed(0)

N_BINS = 20
A0, B0 = 2.52, 1.38           # a representative kernel
N_CELLS = 5

cfg = ModelConfig(
    is_debug=True, optimizer={"lr": 1e-3}, n_bins=N_BINS,
    beta_init=0.1, sigma02_init=0.01, beta_requires_grad=True,
)
net = PolicyNetwork(cfg=cfg, init_values={"lam_null": 0.3183098861837907},
                    kernel_param=KernelParam(a0=A0, b0=B0))

# arbitrary non-trivial sc_total per cell
sc_totals = torch.tensor([3.0, 7.0, 0.0, 12.0, 5.0], dtype=torch.float32)

# ---- NEW vectorized mu/sigma --------------------------------------------
mu_new, sigma_new = net(sc_totals)            # [n_cells, n_bins], [n_bins]

# ---- ORIGINAL loop mu/sigma (reimplemented from the pre-edit code) -------
alpha = float(torch.square(net.sqrt_alpha))
beta = float(torch.square(net.sqrt_beta))
sigma02 = float(torch.square(net.sqrt_sigma02))
rho = 0.0
mu_ref = torch.zeros(N_CELLS, N_BINS)
sigma_ref = torch.zeros(N_BINS)
for c, sc in enumerate(sc_totals):
    sc = float(sc) / 1.0
    for b in range(N_BINS):
        r_mid = (b + 0.5) / N_BINS
        varphi = beta_dist.pdf(r_mid, A0, B0)
        mu_ref[c, b] = (alpha + beta * varphi) * sc * 2 * math.pi * r_mid
        sigma_ref[b] = math.sqrt(sigma02 + rho * varphi)

dmu = (mu_new - mu_ref).abs().max().item()
dsigma = (sigma_new - sigma_ref).abs().max().item()
print(f"mu:    max|new - loop| = {dmu:.3e}")
print(f"sigma: max|new - loop| = {dsigma:.3e}")

# ---- sampling + per-transcript gather, given identical injected eps ------
eps = torch.randn(N_CELLS, N_BINS)

# NEW path: relu(mu + eps*sigma), log_prob, then the #1 gather per cell
lam_new = F.relu(mu_new + eps * sigma_new)
logp_new = torch.distributions.Normal(mu_new, sigma_new).log_prob(mu_new + eps * sigma_new)

# fake transcripts per cell (normalized radii in [0, ~1.05] to also exercise the clamp)
cell_points = [
    torch.tensor([0.05, 0.5, 0.99]),
    torch.tensor([0.2, 0.85, 1.04, 0.33]),
    torch.tensor([]),                       # zero-point cell -> must be dropped
    torch.tensor([0.5]),
    torch.tensor([0.1, 0.1, 0.95, 0.6, 0.7]),
]

# Feed the SAME lam tensor to both gathers so we test ONLY the gather/binning
# logic (vectorized tensor gather vs the original per-molecule python loop),
# not float32-vs-float64 accumulation near the relu/log boundary.
rewards_new, rewards_ref = [], []
for ci, pts in enumerate(cell_points):
    if len(pts) == 0:
        continue
    lam_vec = lam_new[ci]
    # NEW: vectorized gather (matches COX.forward / bottleneck #1)
    bin_idx = torch.clamp(torch.floor(pts * N_BINS).long(), max=N_BINS - 1)
    sum_log_new = torch.log(lam_vec[bin_idx] + 1e-10).sum()
    rewards_new.append((-lam_vec.mean() + sum_log_new).item())
    # REF: original per-molecule python loop over the same lam_vec. Bin in
    # float32 via torch (exactly as the pre-#1 loop did: torch.floor(point*n_bins)),
    # NOT math.floor(float(p)*n_bins) which would promote to float64 and re-bin.
    logs = []
    for p in pts:
        b = int(torch.minimum(torch.floor(p * N_BINS).long(),
                              torch.tensor(N_BINS - 1)))
        logs.append(float(torch.log(lam_vec[b] + 1e-10)))
    rewards_ref.append(-float(lam_vec.mean()) + sum(logs))

dreward = max(abs(a - b) for a, b in zip(rewards_new, rewards_ref))
print(f"per-cell reward: max|new - loop| = {dreward:.3e}  ({len(rewards_new)} cells, 1 dropped)")

# also confirm the batched Normal log_prob matches a per-element Normal log_prob
logp_loop = torch.zeros_like(logp_new)
for c in range(N_CELLS):
    for b in range(N_BINS):
        d = torch.distributions.Normal(mu_new[c, b], sigma_new[b])
        logp_loop[c, b] = d.log_prob(mu_new[c, b] + eps[c, b] * sigma_new[b])
dlogp = (logp_new - logp_loop).abs().max().item()
print(f"log_prob: max|batched - per-elem| = {dlogp:.3e}")

ok = dmu < 1e-5 and dsigma < 1e-6 and dreward < 1e-5 and dlogp < 1e-5
print("\nPASS: vectorized math matches the loop" if ok else "\nFAIL: mismatch")
