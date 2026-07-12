"""Repo-history diagram (matplotlib port of the former mermaid flowchart).

Styled to match benchmark/figures/make_summary_figure.py: same sans rcParams,
same forest-green / light-gray palette, thin flat strokes. Renders the ELLA
variant lineage:

    ELLA Adam (original, archived)
      ├─ same model, Adam -> bounded Newton  ->  ELLA Newton  (main, flagship)
      └─ different model, Cox-process rewrite ->  ELLA-Cox    (ella-cox, archived)

Outputs repo_history.png + repo_history.pdf beside this script. Run in `ella2d`.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = Path(__file__).resolve().parent

# ---- shared palette / typography (mirrors the benchmark figure) -----------------
GREEN = "#2E7D32"
GREEN_FILL = "#E3F0E3"       # light green flagship fill (mermaid dff0d8, softened)
GRAY_EDGE = "#BFBFBF"
GRAY_FILL = "#F0F0F0"
INK_TITLE = "#333333"
INK_SUB = "#666666"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
})


def node(ax, cx, cy, title, subtitle, tag, flagship=False, w=0.30, h=0.185):
    """Rounded box centered at (cx, cy). Returns (top, bottom) anchor points."""
    edge = GREEN if flagship else GRAY_EDGE
    fill = GREEN_FILL if flagship else GRAY_FILL
    lw = 1.8 if flagship else 1.0
    box = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                         boxstyle="round,pad=0.006,rounding_size=0.022",
                         linewidth=lw, edgecolor=edge, facecolor=fill, zorder=3)
    ax.add_patch(box)
    title_color = GREEN if flagship else INK_TITLE
    ax.text(cx, cy + h * 0.24, title, ha="center", va="center",
            fontsize=10, fontweight="bold", color=title_color, zorder=4)
    ax.text(cx, cy - h * 0.06, subtitle, ha="center", va="center",
            fontsize=7.5, color=INK_SUB, zorder=4)
    ax.text(cx, cy - h * 0.30, tag, ha="center", va="center",
            fontsize=7.5, style="italic", color=(GREEN if flagship else GRAY_EDGE),
            zorder=4)
    return (cx, cy + h / 2), (cx, cy - h / 2)


def edge_label(ax, x, y, text, color=INK_SUB):
    ax.text(x, y, text, ha="center", va="center", fontsize=7.2, color=color,
            zorder=5, bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                                edgecolor="none"))


def main():
    fig, ax = plt.subplots(figsize=(5.6, 2.62), dpi=300)
    # limits tightened to the content bounding box so there is no internal white
    # margin (bbox_inches="tight" only crops to the axes rectangle, not the artists)
    ax.set_xlim(0.09, 0.91)
    ax.set_ylim(0.095, 0.905)
    ax.axis("off")

    # nodes
    _, adam_bot = node(ax, 0.50, 0.80, "ELLA Adam", "original release", "archived")
    newton_top, _ = node(ax, 0.255, 0.20, "ELLA Newton  ★",
                         "branch: main (default)", "recommended", flagship=True)
    cox_top, _ = node(ax, 0.745, 0.20, "ELLA-Cox", "branch: ella-cox", "archived")

    # arrows (Adam -> Newton, Adam -> Cox)
    arrow_kw = dict(arrowstyle="-|>", mutation_scale=12, linewidth=1.1,
                    shrinkA=2, shrinkB=2, zorder=2)
    ax.add_patch(FancyArrowPatch((0.44, adam_bot[1]), newton_top,
                                 color=GREEN, **arrow_kw))
    ax.add_patch(FancyArrowPatch((0.56, adam_bot[1]), cox_top,
                                 color=GRAY_EDGE, **arrow_kw))

    # edge labels
    edge_label(ax, 0.285, 0.52, "same model\nAdam → bounded Newton", color=GREEN)
    edge_label(ax, 0.715, 0.52, "different model\nCox-process rewrite", color=INK_SUB)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"repo_history.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {HERE / 'repo_history.png'} (+ .pdf)")


if __name__ == "__main__":
    main()
