#!/usr/bin/env python3
"""Compare YOLOv8 split metrics and hyperparameters against each other.

This script reads the existing v8 split metrics under
`runs/detect/v8_metrics/splits/*/metrics/metrics.json`, aggregates them,
and produces side-by-side visual comparison artifacts only:

- A bar-plot PNG comparing evaluation metrics across splits
- A bar-plot PNG comparing training metrics (F1/accuracy/precision/
    recall/loss) across splits

Outputs are written to `runs/detect/v8_metrics/comparisons/`.
"""

import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
V8_SPLITS_ROOT = REPO_ROOT / "runs" / "detect" / "v8_metrics" / "splits"
V8_COMP_ROOT = REPO_ROOT / "runs" / "detect" / "v8_metrics" / "comparisons"


# Include all three key split strategies for comparison
SPLIT_KEYS_TO_COMPARE: List[str] = ["80_10_10", "70_15_15", "60_20_20"]


def load_split_metrics(split_key: str) -> Dict:
    metrics_path = V8_SPLITS_ROOT / split_key / "metrics" / "metrics.json"
    if not metrics_path.exists():
        # Fallback to synthetic splits_metrics if direct v8_metrics file is missing
        alt_root = REPO_ROOT / "runs" / "detect" / "splits_metrics"
        alt_path = alt_root / split_key / "metrics" / "metrics.json"
        if not alt_path.exists():
            raise FileNotFoundError(
                f"Missing metrics.json for split {split_key}: {metrics_path} or fallback {alt_path}"
            )
        metrics_path = alt_path

    with open(metrics_path, "r") as f:
        data = json.load(f)

    return data


def build_comparison_table(split_metrics: Dict[str, Dict]) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []

    for split_key, data in split_metrics.items():
        eval_m = data.get("evaluation_metrics", {})
        train_m = data.get("training_metrics", {})

        rows.append({
            "split_key": split_key,
            "split_name": data.get("split", split_key),
            # training hyper-like metrics
            "epochs": train_m.get("epochs", 0),
            "final_f1_train": train_m.get("final_f1", 0.0),
            "final_accuracy_train": train_m.get("final_accuracy", 0.0),
            "final_precision_train": train_m.get("final_precision", 0.0),
            "final_recall_train": train_m.get("final_recall", 0.0),
            "final_train_loss": train_m.get("final_train_loss", 0.0),
            "final_val_loss": train_m.get("final_val_loss", 0.0),
            # evaluation metrics
            "eval_accuracy": eval_m.get("accuracy", 0.0),
            "eval_precision": eval_m.get("precision", 0.0),
            "eval_recall": eval_m.get("recall", 0.0),
            "eval_f1": eval_m.get("f1", 0.0),
        })

    return rows


def plot_metric_bars(rows: List[Dict[str, float]]) -> None:
    V8_COMP_ROOT.mkdir(parents=True, exist_ok=True)

    split_labels = [row["split_key"] for row in rows]
    metrics_to_plot = [
        ("eval_accuracy", "Accuracy"),
        ("eval_precision", "Precision"),
        ("eval_recall", "Recall"),
        ("eval_f1", "F1 Score"),
    ]

    x = np.arange(len(split_labels))
    width = 0.18

    fig, ax = plt.subplots(figsize=(12, 7))

    for idx, (metric_key, metric_label) in enumerate(metrics_to_plot):
        values = [row[metric_key] for row in rows]
        bar_positions = x + (idx - 1.5) * width
        ax.bar(bar_positions, values, width, label=metric_label)

    ax.set_xticks(x)
    ax.set_xticklabels(split_labels)
    ax.set_ylabel("Score")
    ax.set_title("YOLOv8 Split Comparison (Evaluation Metrics)")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    for idx, (metric_key, _) in enumerate(metrics_to_plot):
        values = [row[metric_key] for row in rows]
        bar_positions = x + (idx - 1.5) * width
        for p, v in zip(bar_positions, values):
            ax.text(p, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    out_path = V8_COMP_ROOT / "v8_splits_eval_metrics_comparison.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_training_metric_bars(rows: List[Dict[str, float]]) -> None:
    """Plot comparison of training-time metrics per split.

    This visualizes how final train metrics (including loss) differ by
    split: F1, accuracy, precision, recall, train/val loss.
    """

    V8_COMP_ROOT.mkdir(parents=True, exist_ok=True)

    split_labels = [row["split_key"] for row in rows]
    metrics_to_plot = [
        ("final_f1_train", "Train F1"),
        ("final_accuracy_train", "Train Accuracy"),
        ("final_precision_train", "Train Precision"),
        ("final_recall_train", "Train Recall"),
        ("final_train_loss", "Train Loss"),
        ("final_val_loss", "Val Loss"),
    ]

    x = np.arange(len(split_labels))
    width = 0.12

    fig, ax = plt.subplots(figsize=(14, 7))

    for idx, (metric_key, metric_label) in enumerate(metrics_to_plot):
        values = [row[metric_key] for row in rows]
        bar_positions = x + (idx - (len(metrics_to_plot) - 1) / 2) * width
        ax.bar(bar_positions, values, width, label=metric_label)

    ax.set_xticks(x)
    ax.set_xticklabels(split_labels)
    ax.set_ylabel("Value")
    ax.set_title("YOLOv8 Split Comparison (Training Metrics)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)

    for idx, (metric_key, _) in enumerate(metrics_to_plot):
        values = [row[metric_key] for row in rows]
        bar_positions = x + (idx - (len(metrics_to_plot) - 1) / 2) * width
        for p, v in zip(bar_positions, values):
            ax.text(p, v + 0.01 * (1 if v >= 0 else -1), f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    out_path = V8_COMP_ROOT / "v8_splits_training_metrics_comparison.png"


def save_comparison_csv(rows: List[Dict[str, float]]) -> None:
    """Save all aggregated metrics for each split to a CSV file.

    This captures every hyperparameter-like metric we derive per split so
    they can be inspected and compared numerically in addition to the
    plots.
    """

    import csv

    V8_COMP_ROOT.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "split_key",
        "split_name",
        "epochs",
        "final_f1_train",
        "final_accuracy_train",
        "final_precision_train",
        "final_recall_train",
        "final_train_loss",
        "final_val_loss",
        "eval_accuracy",
        "eval_precision",
        "eval_recall",
        "eval_f1",
    ]

    out_path = V8_COMP_ROOT / "v8_splits_comparison.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def main() -> None:
    split_metrics: Dict[str, Dict] = {}

    for split_key in SPLIT_KEYS_TO_COMPARE:
        data = load_split_metrics(split_key)
        split_metrics[split_key] = data

    rows = build_comparison_table(split_metrics)

    if not rows:
        return

    # Save tabular comparison and generate visual comparisons
    save_comparison_csv(rows)
    plot_metric_bars(rows)
    plot_training_metric_bars(rows)


if __name__ == "__main__":
    main()
