#!/usr/bin/env python3
"""Regenerate metrics_summary.png for YOLOv8 60_20_20 split
based on the current metrics.json values.
"""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = REPO_ROOT / "runs" / "detect" / "v8_metrics" / "splits" / "60_20_20" / "metrics"
METRICS_PATH = SPLIT_DIR / "metrics.json"
OUTPUT_PATH = SPLIT_DIR / "metrics_summary.png"


def main() -> None:
    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"metrics.json not found: {METRICS_PATH}")

    with METRICS_PATH.open("r") as f:
        data = json.load(f)

    eval_m = data.get("evaluation_metrics", {})

    labels = ["Accuracy", "Precision", "Recall", "F1"]
    values = [
        float(eval_m.get("accuracy", 0.0)),
        float(eval_m.get("precision", 0.0)),
        float(eval_m.get("recall", 0.0)),
        float(eval_m.get("f1", 0.0)),
    ]

    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.bar(labels, values, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("YOLOv8 Metrics Summary (60/20/20)")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
