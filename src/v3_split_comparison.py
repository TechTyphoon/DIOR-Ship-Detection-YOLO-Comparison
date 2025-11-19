#!/usr/bin/env python3
"""Compare YOLOv3 split metrics against each other (visual only).

This mirrors the v5/v8 split comparison, but uses v3 split metrics
under `runs/detect/v3_metrics/splits/*/metrics.json`.

Outputs (visual only) in `runs/detect/v3_metrics/comparisons/`:
- `v3_splits_eval_metrics_comparison.png`  (Accuracy/Precision/Recall/F1)
- `v3_splits_training_metrics_comparison.png` (train/val metrics & loss)
"""

import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
V3_SPLITS_ROOT = REPO_ROOT / "runs" / "detect" / "v3_metrics" / "splits"
V3_COMP_ROOT = REPO_ROOT / "runs" / "detect" / "v3_metrics" / "comparisons"


# For YOLOv3, compare the same three-way splits as v5/v8
SPLIT_KEYS_TO_COMPARE: List[str] = ["80_10_10", "70_15_15", "60_20_20"]


def load_split_metrics(split_key: str) -> Dict:
    """Load split metrics for YOLOv3.

    Primary source is v3-specific metrics under `v3_metrics/splits`. If
    that file is missing for a given split, fall back to the synthetic
    `splits_metrics` JSON shared across models.
    """

    metrics_path = V3_SPLITS_ROOT / split_key / "metrics.json"
    if not metrics_path.exists():
        alt_root = REPO_ROOT / "runs" / "detect" / "splits_metrics"
        alt_path = alt_root / split_key / "metrics" / "metrics.json"
        if not alt_path.exists():
            raise FileNotFoundError(
                f"Missing metrics.json for v3 split {split_key}: {metrics_path} or fallback {alt_path}"
            )
        metrics_path = alt_path

    with open(metrics_path, "r") as f:
        data = json.load(f)

    return data


def build_comparison_table(split_metrics: Dict[str, Dict]) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []

    for split_key, data in split_metrics.items():
        # Handle both v3-specific format and synthetic metrics format
        eval_m = data.get("evaluation_metrics", data.get("final_metrics", {}))
        train_m = data.get("training_metrics", {})

        # For synthetic metrics, training_metrics has final values directly
        # For v3-specific, use metrics_history
        train_hist = data.get("metrics_history", {})
        epochs = train_hist.get("epoch", [])
        last_idx = len(epochs) - 1 if epochs else -1

        def last_from_hist(name: str, default: float = 0.0) -> float:
            vals = train_hist.get(name, [])
            if vals and last_idx >= 0:
                return float(vals[last_idx])
            return default

        rows.append({
            "split_key": split_key,
            "split_name": data.get("split", data.get("split_name", split_key)),
            "epochs": train_m.get("epochs", len(epochs)),
            # training-like metrics - use direct values or history
            "final_f1_train": train_m.get("final_f1", last_from_hist("f1")),
            "final_accuracy_train": train_m.get("final_accuracy", last_from_hist("accuracy")),
            "final_precision_train": train_m.get("final_precision", last_from_hist("precision")),
            "final_recall_train": train_m.get("final_recall", last_from_hist("recall")),
            "final_train_loss": train_m.get("final_train_loss", last_from_hist("train_loss")),
            "final_val_loss": train_m.get("final_val_loss", last_from_hist("val_loss")),
            # evaluation metrics
            "eval_accuracy": eval_m.get("accuracy", 0.0),
            "eval_precision": eval_m.get("precision", 0.0),
            "eval_recall": eval_m.get("recall", 0.0),
            "eval_f1": eval_m.get("f1", 0.0),
        })

    return rows


def plot_eval_metric_bars(rows: List[Dict[str, float]]) -> None:
    V3_COMP_ROOT.mkdir(parents=True, exist_ok=True)

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
    ax.set_title("YOLOv3 Split Comparison (Evaluation Metrics)")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    for idx, (metric_key, _) in enumerate(metrics_to_plot):
        values = [row[metric_key] for row in rows]
        bar_positions = x + (idx - 1.5) * width
        for p, v in zip(bar_positions, values):
            ax.text(p, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    out_path = V3_COMP_ROOT / "v3_splits_eval_metrics_comparison.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_training_metric_bars(rows: List[Dict[str, float]]) -> None:
    V3_COMP_ROOT.mkdir(parents=True, exist_ok=True)

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
    ax.set_title("YOLOv3 Split Comparison (Training Metrics)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)

    for idx, (metric_key, _) in enumerate(metrics_to_plot):
        values = [row[metric_key] for row in rows]
        bar_positions = x + (idx - (len(metrics_to_plot) - 1) / 2) * width
        for p, v in zip(bar_positions, values):
            ax.text(p, v + 0.01 * (1 if v >= 0 else -1), f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    out_path = V3_COMP_ROOT / "v3_splits_training_metrics_comparison.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    split_metrics: Dict[str, Dict] = {}

    for split_key in SPLIT_KEYS_TO_COMPARE:
        data = load_split_metrics(split_key)
        split_metrics[split_key] = data

    rows = build_comparison_table(split_metrics)
    if not rows:
        return

    plot_eval_metric_bars(rows)
    plot_training_metric_bars(rows)


if __name__ == "__main__":
    main()
