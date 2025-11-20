#!/usr/bin/env python3
"""Generate confusion_matrix.png for YOLOv8 80_10_10 split
using the confusion matrix values already stored in metrics.json.

This does NOT change any metric values; it only renders the PNG.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = REPO_ROOT / "runs" / "detect" / "v8_metrics" / "splits" / "80_10_10" / "metrics" / "metrics.json"
OUTPUT_PATH = REPO_ROOT / "runs" / "detect" / "v8_metrics" / "splits" / "80_10_10" / "metrics" / "confusion_matrix.png"


def main() -> None:
    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"metrics.json not found: {METRICS_PATH}")

    with METRICS_PATH.open("r") as f:
        data = json.load(f)

    cm_info = data.get("confusion_matrix")
    if not cm_info:
        raise KeyError("'confusion_matrix' missing from metrics.json")

    # Rebuild matrix in the layout the user requested:
    # rows = Actual [Positive, Negative], cols = Predicted [Positive, Negative]
    # [[TP, FN], [FP, TN]]
    tp = int(cm_info["tp"])
    fn = int(cm_info["fn"])
    fp = int(cm_info["fp"])
    tn = int(cm_info["tn"])
    cm = np.array([[tp, fn], [fp, tn]], dtype=int)

    # Axis labels reflecting this layout
    class_names = ["Positive", "Negative"]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Normalize per row for percent annotations
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_percent = np.divide(cm.astype(float), row_sums, where=row_sums != 0) * 100.0

    sns.heatmap(
        cm,
        annot=False,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Count"},
        ax=ax,
        vmin=0,
    )

    # Add count + percentage text like evaluate.py
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            pct = cm_percent[i, j]
            ax.text(
                j + 0.5,
                i + 0.7,
                f"{count}\n({pct:.1f}%)",
                ha="center",
                va="center",
                color="white" if count > cm.max() / 2 else "black",
                fontsize=10,
                weight="bold",
            )

    ax.set_ylabel("Actual", fontsize=11, weight="bold")
    ax.set_xlabel("Predicted", fontsize=11, weight="bold")
    ax.set_title("Confusion Matrix - YOLOv8 80/10/10", fontsize=12, weight="bold", pad=20)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
