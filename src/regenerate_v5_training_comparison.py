import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent
V5_SPLITS_DIR = BASE_DIR / "runs" / "detect" / "v5_metrics" / "splits"


def load_training_metrics(split_name: str) -> dict:
    metrics_path = V5_SPLITS_DIR / split_name / "metrics" / "metrics.json"
    with metrics_path.open("r") as f:
        data = json.load(f)
    return data.get("training_metrics", {})


def plot_training_comparison():
    splits = ["60_20_20", "70_15_15", "80_10_10"]
    display_names = ["60/20/20", "70/15/15", "80/10/10"]

    metrics_per_split = []
    for s in splits:
        m = load_training_metrics(s)
        metrics_per_split.append([
            m.get("final_f1", 0.0),
            m.get("final_accuracy", 0.0),
            m.get("final_precision", 0.0),
            m.get("final_recall", 0.0),
        ])

    metrics_arr = np.array(metrics_per_split)
    labels = ["Final F1", "Final Accuracy", "Final Precision", "Final Recall"]

    x = np.arange(len(display_names))
    width = 0.18

    fig, ax = plt.subplots(figsize=(8, 4))

    for i, metric_name in enumerate(labels):
        ax.bar(x + (i - 1.5) * width, metrics_arr[:, i], width, label=metric_name)

    ax.set_xticks(x)
    ax.set_xticklabels(display_names)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("YOLOv5 Training Metrics Comparison Across Splits")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    out_dir = BASE_DIR / "runs" / "detect" / "v5_metrics" / "comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / "v5_splits_training_metrics_comparison.png", dpi=200)
    plt.close(fig)


def main():
    plot_training_comparison()


if __name__ == "__main__":
    main()
