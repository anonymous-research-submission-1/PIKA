"""Figure 2: model comparison across the IID and OOD evaluation tiers (seed 0).

Values are the seed-0 entries printed by notebooks/pika.ipynb. They are listed
here explicitly so the figure can be redrawn without re-running the notebook;
`python scripts/run_experiments.py` regenerates the same numbers.

    python scripts/plot_model_comparison.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures" / "fig2_model_comparison.png"

MODELS = [
    ("PIKA",           "#4DA3E8"),
    ("LSTM",           "#F4753E"),
    ("XGBoost",        "#5CB85C"),
    ("Random\nForest", "#A54FC4"),
    ("PDD/TI",         "#F5A623"),
    ("PIKA\n(large)",  "#7B94A3"),
    ("Naive\nPersistence", "#8B6D5C"),
]
# Region-balanced RMSE, 65-glacier IID temporal test
IID = [0.6483, 0.5799, 0.6610, 0.7531, 0.6346, 0.6614, 0.9925]
# Pooled RMSE, 15-glacier OOD spatial holdout
OOD = [0.9014, 0.9329, 0.9408, 1.1839, 0.9486, 0.7690, 0.9486]


def panel(ax, values, ylabel, title):
    x = np.arange(len(MODELS))
    ax.bar(x, values, color=[c for _, c in MODELS], width=0.68)
    ax.set_xticks(x)
    ax.set_xticklabels([n for n, _ in MODELS], fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.set_ylim(0, max(values) * 1.13)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    panel(axes[0], IID, "Region-Balanced RMSE (m w.e.)", "IID Temporal Test")
    panel(axes[1], OOD, "Pooled RMSE (m w.e.)", "OOD Holdout Test")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
