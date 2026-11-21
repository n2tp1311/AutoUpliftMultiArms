"""Draw the report figures from masked_results.json (the only input; no data, no model).

Usage: python make_figures.py   ->  figures/*.png
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
OUT = HERE / "figures"
INK, MUTED, RULE = "#1F2933", "#5B6573", "#CBD5E1"
BLUE, ORANGE = "#1F4E79", "#B45309"
ORDER = ["first_full", "first_reduced", "later_full", "later_reduced"]

plt.rcParams.update({"font.size": 9, "axes.edgecolor": RULE, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": INK, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 200, "savefig.bbox": "tight"})


def forest(ax, rows, title, xlabel, ref=0.0, labels_right=False):
    """rows: list of (label, [estimate, low, high]); drawn top to bottom."""
    for i, (lab, (est, lo, hi)) in enumerate(rows):
        y = len(rows) - 1 - i
        color = BLUE if lo > ref else (ORANGE if hi < ref else MUTED)
        ax.plot([lo, hi], [y, y], color=color, lw=2.2, solid_capstyle="round")
        ax.plot(est, y, "o", color=color, ms=6)
        left = est < ref and not labels_right
        ax.annotate(f"{est:+.1f}", (lo if left else hi, y), xytext=(-5 if left else 5, 0),
                    textcoords="offset points", va="center", ha="right" if left else "left",
                    fontsize=8, color=INK)
    ax.axvline(ref, color=INK, lw=0.8)
    ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_title(title, loc="left", fontsize=9, color=INK, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color=RULE, lw=0.5)


def effects(d):
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.2), gridspec_kw={"height_ratios": [4, 4]})
    for ax, key in zip(axes.flat, ORDER):
        v = d["versions"][key]
        forest(ax, list(v["effects"].items()), f"{v['label']}\nvs {v['control']}",
               "Effect on on-time payment (pp), 95% CI")
        ax.set_xlim(-1.0, 2.0)
    fig.tight_layout(h_pad=2.0)
    fig.savefig(OUT / "fig1_effects.png")
    plt.close(fig)


def calibration(d):
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.2))
    for ax, key in zip(axes.flat, ORDER):
        v = d["versions"][key]
        forest(ax, list(v["calibration_slopes"].items()), f"{v['label']}\nvs {v['control']}",
               "Calibration slope, 95% CI (1 = well ranked, 0 = no signal)",
               labels_right=True)
        ax.axvline(1.0, color=MUTED, lw=0.8, ls="--")
        ax.set_xlim(-1.5, 3.0)
    fig.tight_layout(h_pad=2.0)
    fig.savefig(OUT / "fig3_calibration.png")
    plt.close(fig)


def products(d):
    prods = d["products"]
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.4))
    for ax, key in zip(axes.flat, ORDER):
        v = d["versions"][key]
        arms = list(v["by_product"])
        grid = [v["by_product"][a] for a in arms]
        ax.imshow(grid, cmap="RdBu", vmin=-2.0, vmax=2.0, aspect="auto")
        for i, row in enumerate(grid):
            best = [max(v["by_product"][a][j] for a in arms) for j in range(len(prods))]
            for j, val in enumerate(row):
                strong = abs(val) > 1.2
                ax.text(j, i, f"{val:+.1f}", ha="center", va="center", fontsize=8,
                        color="white" if strong else INK,
                        fontweight="bold" if val == best[j] and len(arms) > 1 else "normal")
        ax.set_xticks(range(len(prods)), prods)
        ax.set_yticks(range(len(arms)), arms)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"{v['label']}\nvs {v['control']} (model estimate, pp)", loc="left",
                     fontsize=9, color=INK, fontweight="bold")
    fig.tight_layout(h_pad=2.0)
    fig.savefig(OUT / "fig2_products.png")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    data = json.loads((HERE / "masked_results.json").read_text())
    effects(data)
    products(data)
    calibration(data)
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
