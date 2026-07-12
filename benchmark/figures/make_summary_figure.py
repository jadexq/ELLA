"""Nature-style summary figure: ELLA Newton vs Adam on cyto_symmetric (100 genes).

Four sub-panels in a single row, each in its own true units:
  a. Speed        -- wall-clock time (s), grouped bars.
  b. Peak memory  -- peak RSS (GB), grouped bars.
  c. Power        -- BH-FDR < 0.05; single stacked bar (Adam base + Newton increment).
  d. Accuracy     -- per-gene KL(true || est) [nats], box + jittered strip.

All measures come from the cyto_symmetric run, so the figure is internally consistent.
Reads the saved benchmark CSVs; no ELLA re-run. Run in the `ella2d` env.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

BENCH = Path(__file__).resolve().parents[1]
OUT = BENCH / "output"
FIGDIR = BENCH / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

EXPERIMENT = "cyto_symmetric"
METHODS = ["adam", "newton"]                    # Adam left, Newton right (consistent)
FILL = {"adam": "#BFBFBF", "newton": "#2E7D32"}   # light gray / forest green
INK = {"adam": "#5E5E5E", "newton": "#2E7D32"}    # readable text tint
LABEL = {"adam": "ELLA Adam", "newton": "ELLA Newton"}
XPOS = {"adam": 0, "newton": 1}

# ---- Nature-ish rcParams --------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.linewidth": 0.8,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "svg.fonttype": "none",
})


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")


def two_bars(ax, vals, fmt, ylabel, title, headroom=1.18):
    """Grouped two-bar panel (Adam, Newton) with value labels above each bar."""
    top = max(vals.values()) * headroom
    for m in METHODS:
        ax.bar(XPOS[m], vals[m], width=0.62, color=FILL[m], zorder=3)
        ax.text(XPOS[m], vals[m] + top * 0.02, fmt.format(vals[m]),
                ha="center", va="bottom", fontsize=8, color=INK[m])
    ax.set_ylim(0, top)
    ax.set_xlim(-0.7, 1.7)
    ax.set_xticks([])
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=6)
    style(ax)


def main():
    res = pd.read_csv(OUT / "resources.csv")
    res = res[res.experiment == EXPERIMENT].set_index("method")
    power = pd.read_csv(OUT / "cyto_power.csv").set_index("method")
    kld = pd.read_csv(OUT / "cyto_kld.csv")

    fig, axes = plt.subplots(1, 4, figsize=(7.8, 2.7), dpi=300)
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.13, top=0.76, wspace=0.66)
    axA, axB, axC, axD = axes

    # ---- a. Speed -------------------------------------------------------------
    two_bars(axA, {m: res.loc[m, "wall_s"] for m in METHODS},
             "{:.0f}", "Wall-clock time (s)", "Speed")

    # ---- b. Peak memory -------------------------------------------------------
    two_bars(axB, {m: res.loc[m, "peak_mem_gb"] for m in METHODS},
             "{:.1f}", "Peak memory (GB)", "Peak memory")

    # ---- c. Power (stacked: Adam base + Newton increment) ---------------------
    p_adam = power.loc["adam", "power"]
    p_newton = power.loc["newton", "power"]
    inc = p_newton - p_adam
    axC.bar(0, p_adam, width=0.5, color=FILL["adam"], zorder=3)
    axC.bar(0, inc, width=0.5, bottom=p_adam, color=FILL["newton"], zorder=3)
    axC.text(0, p_adam / 2, f"{p_adam:.2f}", ha="center", va="center",
             fontsize=8, color="white", fontweight="bold")
    axC.text(0.30, p_adam + inc / 2, f"+{inc:.2f}", ha="left", va="center",
             fontsize=7.5, color=INK["newton"])
    axC.text(0, p_newton + 0.02, f"{p_newton:.2f}", ha="center", va="bottom",
             fontsize=8, color="black")
    axC.set_ylim(0, 1.08)
    axC.set_xlim(-0.55, 0.75)
    axC.set_xticks([])
    axC.set_ylabel("Power (BH-FDR < 0.05)")
    axC.set_title("Power", pad=6)
    style(axC)

    # ---- d. Accuracy (per-gene KL: box + jittered strip) ----------------------
    rng = np.random.default_rng(0)
    data = {m: kld.loc[kld.method == m, "kl"].to_numpy() for m in METHODS}
    bp = axD.boxplot([data[m] for m in METHODS], positions=[XPOS[m] for m in METHODS],
                     widths=0.5, showfliers=False, patch_artist=True, zorder=2)
    for m, box in zip(METHODS, bp["boxes"]):
        box.set(facecolor=FILL[m], edgecolor="black", linewidth=0.8, alpha=0.85)
    for key in ("whiskers", "caps"):
        for el in bp[key]:
            el.set(color="black", linewidth=0.8)
    for med in bp["medians"]:
        med.set(color="black", linewidth=1.2)
    for m in METHODS:
        d = data[m]
        xj = XPOS[m] + (rng.random(len(d)) - 0.5) * 0.28
        axD.scatter(xj, d, s=5, color=INK[m], alpha=0.35, edgecolor="none", zorder=3)
        axD.text(XPOS[m] + 0.32, np.median(d), f"{np.median(d):.4f}", ha="left",
                 va="center", fontsize=7, color=INK[m])
    axD.set_ylim(0, kld["kl"].max() * 1.12)
    axD.set_xlim(-0.7, 2.05)
    axD.set_xticks([])
    axD.set_ylabel(r"KL(true $\parallel$ est)  (nats)")
    axD.set_title("Accuracy", pad=6)
    style(axD)

    # ---- shared legend + caption ---------------------------------------------
    handles = [Patch(facecolor=FILL[m], edgecolor="none", label=LABEL[m]) for m in METHODS]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.54, 1.005), columnspacing=1.6, handlelength=1.1)
    fig.text(0.075, 0.005, "cyto_symmetric  ·  100 genes  ·  116 cells  ·  32 cores",
             fontsize=6.5, color="0.45", ha="left", va="bottom")

    out = FIGDIR / "benchmark_summary.png"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(FIGDIR / "benchmark_summary.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
