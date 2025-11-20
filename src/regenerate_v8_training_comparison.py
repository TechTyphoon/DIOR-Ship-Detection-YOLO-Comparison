#!/usr/bin/env python3
"""Regenerate v8_splits_training_metrics_comparison.png
using training_metrics from each v8 split metrics.json.

This updates only the YOLOv8 training comparison PNG so that
all comparison artifacts are freshly generated.
"""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLITS = ["60_20_20", "70_15_15", "80_10_10"]
OUT_PATH = REPO_ROOT / "runs" / "detect" / "v8_metrics" / "comparisons" / "v8_splits_training_metrics_comparison.png"


def load_training_metrics(split_key: str):
    mpath = REPO_ROOT / "runs" / "detect" / "v8_metrics" / "splits" / split_key / "metrics" / "metrics.json"
    with mpath.open("r") as f:
        data = json.load(f)
    tm = data.get("training_metrics", {})
    return {
        "split": split_key,
        "final_f1": float(tm.get("final_f1", 0.0)),
        "final_accuracy": float(tm.get("final_accuracy", 0.0)),
        "final_precision": float(tm.get("final_precision", 0.0)),
        "final_recall": float(tm.get("final_recall", 0.0)),
    }


def main() -> None:
    rows = [load_training_metrics(s) for s in SPLITS]
    split_labels = [r["split"] for r in rows]

    metrics = [
        ("final_f1", "Final F1"),
        ("final_accuracy", "Final Accuracy"),
        ("final_precision", "Final Precision"),
        ("final_recall", "Final Recall"),
    ]

    x = np.arange(len(split_labels))
    width = 0.18

    fig, ax = plt.subplots(figsize=(12, 7))

    for idx, (key, label) in enumerate(metrics):
        values = [r[key] for r in rows]
        pos = x + (idx - 1.5) * width
        ax.bar(pos, values, width, label=label)

        # add value labels on bars
        for xv, yv in zip(pos, values):
            ax.text(xv, yv + 0.005, f"{yv:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(split_labels)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("YOLOv8 Split Training Metrics Comparison", fontsize=13, weight="bold")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
