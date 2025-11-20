#!/usr/bin/env python3
"""Generate confusion_matrix.png for YOLOv8 60_20_20 split
using the confusion matrix values stored in metrics.json.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = REPO_ROOT / "runs" / "detect" / "v8_metrics" / "splits" / "60_20_20" / "metrics" / "metrics.json"
OUTPUT_PATH = REPO_ROOT / "runs" / "detect" / "v8_metrics" / "splits" / "60_20_20" / "metrics" / "confusion_matrix.png"


def main() -> None:
    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"metrics.json not found: {METRICS_PATH}")

    with METRICS_PATH.open("r") as f:
        data = json.load(f)

    cm_info = data.get("confusion_matrix")
    if not cm_info or "matrix" not in cm_info:
        raise KeyError("'confusion_matrix.matrix' missing from metrics.json")

    cm = np.array(cm_info["matrix"], dtype=int)

    class_names = ["No Ship", "Ship"]

    fig, ax = plt.subplots(figsize=(8, 6))

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

    ax.set_ylabel("True Label", fontsize=11, weight="bold")
    ax.set_xlabel("Predicted Label", fontsize=11, weight="bold")
    ax.set_title("Confusion Matrix - YOLOv8 60/20/20", fontsize=12, weight="bold", pad=20)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
