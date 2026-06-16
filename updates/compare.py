"""
Overlay baseline vs optimized estimated lam(r) + report timing delta.

#2/#3 batch the rsample() draw, so the RNG stream is consumed differently than the
baseline's per-(cell,bin) scalar draws; curves are therefore expected to TRACK, not
be bit-identical (unlike #1, which was deterministic). Prints max|dlam| per gene.

Run from repo root:  conda run --no-capture-output -n ella python updates/compare.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "output"
base = json.loads((OUT / "baseline" / "lam_curves.json").read_text())
opt = json.loads((OUT / "optimized" / "lam_curves.json").read_text())
tb = json.loads((OUT / "baseline" / "timing.json").read_text())
to = json.loads((OUT / "optimized" / "timing.json").read_text())

genes = sorted(base, key=int)
fig, axes = plt.subplots(1, len(genes), figsize=(4 * len(genes), 3.2), squeeze=False)
print("=== lam(r) baseline vs optimized ===")
for ax, g in zip(axes[0], genes):
    b = np.array(base[g]); o = np.array(opt[g])
    r = (np.arange(len(b)) + 0.5) / len(b)
    ax.plot(r, b, lw=2, label="baseline")
    ax.plot(r, o, lw=2, ls="--", label="optimized")
    dmax = np.abs(b - o).max()
    print(f"  gene {g}: max|dlam| = {dmax:.4e}")
    ax.set_title(f"gene {g}  (max|dlam|={dmax:.2e})")
    ax.set_xlabel("r"); ax.set_ylabel(r"$\lambda(r)$"); ax.grid(alpha=0.3); ax.legend()
fig.suptitle("ELLA lam(r): baseline vs optimized (#1+#2+#3)")
fig.tight_layout()
fig.savefig(OUT / "compare_lam_curves.png", dpi=130)

print("\n=== timing ===")
print(f"  baseline : {tb['total_secs']:.1f}s")
print(f"  optimized: {to['total_secs']:.1f}s")
print(f"  speedup  : {tb['total_secs'] / to['total_secs']:.2f}x "
      f"({100 * (1 - to['total_secs'] / tb['total_secs']):.0f}% faster)")
print(f"\nwrote {OUT / 'compare_lam_curves.png'}")
