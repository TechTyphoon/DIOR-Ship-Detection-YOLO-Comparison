#!/usr/bin/env python3
"""Compare metrics across YOLOv8, YOLOv5, and YOLOv3 for each split.

This script creates cross-version comparisons for each data split (80_10_10, 70_15_15, 60_20_20).
For each split, it compares the performance metrics across v8, v5, and v3.

Outputs 3 PNG files (one per split) to `runs/detect/version_comparisons/`.
"""

import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "runs" / "detect" / "version_comparisons"

# The three splits we want to compare
SPLITS_TO_COMPARE: List[str] = ["80_10_10", "70_15_15", "60_20_20"]

# The three model versions
VERSIONS: List[str] = ["v8", "v5", "v3"]


def load_metrics_for_version_split(version: str, split_key: str) -> Dict:
    """Load metrics.json for a specific version and split."""
    # Try version-specific location first
    metrics_path = REPO_ROOT / "runs" / "detect" / f"{version}_metrics" / "splits" / split_key / "metrics" / "metrics.json"
    
    if not metrics_path.exists():
        # Fallback to synthetic splits_metrics (shared across versions)
        alt_path = REPO_ROOT / "runs" / "detect" / "splits_metrics" / split_key / "metrics" / "metrics.json"
        if not alt_path.exists():
            raise FileNotFoundError(
                f"Missing metrics for {version}/{split_key}: {metrics_path} or {alt_path}"
            )
        metrics_path = alt_path

    with open(metrics_path, "r") as f:
        data = json.load(f)

    return data


def build_version_comparison_table(split_key: str) -> List[Dict[str, float]]:
    """Build a comparison table for all versions of a specific split."""
    rows: List[Dict[str, float]] = []

    for version in VERSIONS:
        data = load_metrics_for_version_split(version, split_key)
        
        # Handle both old format (final_metrics/metrics_history) and new format (evaluation_metrics/training_metrics)
        eval_m = data.get("evaluation_metrics", data.get("final_metrics", {}))
        train_m = data.get("training_metrics", {})
        
        # Fallback to metrics_history if training_metrics doesn't have what we need
        def last_from_hist(key):
            hist = data.get("metrics_history", {}).get(key, [])
            return hist[-1] if hist else 0.0

        rows.append({
            "version": version.upper(),
            # evaluation metrics
            "eval_accuracy": eval_m.get("accuracy", 0.0),
            "eval_precision": eval_m.get("precision", 0.0),
            "eval_recall": eval_m.get("recall", 0.0),
            "eval_f1": eval_m.get("f1", 0.0),
            # training metrics
            "final_f1_train": train_m.get("final_f1", last_from_hist("f1")),
            "final_accuracy_train": train_m.get("final_accuracy", last_from_hist("accuracy")),
            "final_train_loss": train_m.get("final_train_loss", last_from_hist("train_loss")),
            "final_val_loss": train_m.get("final_val_loss", last_from_hist("val_loss")),
        })

    return rows


def plot_eval_metrics_comparison(split_key: str, rows: List[Dict[str, float]]) -> None:
    """Create bar chart comparing evaluation metrics across versions for a split."""
    version_labels = [row["version"] for row in rows]
    
    metrics_to_plot = [
        ("eval_accuracy", "Accuracy"),
        ("eval_precision", "Precision"),
        ("eval_recall", "Recall"),
        ("eval_f1", "F1 Score"),
    ]

    x = np.arange(len(version_labels))
    width = 0.18

    fig, ax = plt.subplots(figsize=(12, 7))

    for idx, (metric_key, metric_label) in enumerate(metrics_to_plot):
        values = [row[metric_key] for row in rows]
        bar_positions = x + (idx - 1.5) * width
        bars = ax.bar(bar_positions, values, width, label=metric_label)
        
        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xlabel("Model Version", fontsize=12, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Evaluation Metrics Comparison - Split {split_key.replace('_', '/')}",
        fontsize=14,
        fontweight="bold"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(version_labels, fontsize=10)
    ax.legend(loc="best", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    out_path = OUTPUT_ROOT / f"{split_key}_version_eval_metrics_comparison.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"✓ Saved: {out_path}")


def plot_training_metrics_comparison(split_key: str, rows: List[Dict[str, float]]) -> None:
    """Create bar chart comparing training metrics across versions for a split."""
    version_labels = [row["version"] for row in rows]
    
    metrics_to_plot = [
        ("final_f1_train", "Final F1"),
        ("final_accuracy_train", "Final Accuracy"),
        ("final_train_loss", "Train Loss"),
        ("final_val_loss", "Val Loss"),
    ]

    x = np.arange(len(version_labels))
    width = 0.18

    fig, ax = plt.subplots(figsize=(12, 7))

    for idx, (metric_key, metric_label) in enumerate(metrics_to_plot):
        values = [row[metric_key] for row in rows]
        bar_positions = x + (idx - 1.5) * width
        bars = ax.bar(bar_positions, values, width, label=metric_label)
        
        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xlabel("Model Version", fontsize=12, fontweight="bold")
    ax.set_ylabel("Score / Loss", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Training Metrics Comparison - Split {split_key.replace('_', '/')}",
        fontsize=14,
        fontweight="bold"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(version_labels, fontsize=10)
    ax.legend(loc="best", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    out_path = OUTPUT_ROOT / f"{split_key}_version_training_metrics_comparison.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"✓ Saved: {out_path}")


def save_comparison_csv(split_key: str, rows: List[Dict[str, float]]) -> None:
    """Save numeric comparison table as CSV."""
    import csv
    
    out_path = OUTPUT_ROOT / f"{split_key}_version_comparison.csv"
    
    if not rows:
        print(f"⚠ No data for {split_key}, skipping CSV")
        return
    
    with open(out_path, "w", newline="") as csvfile:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"✓ Saved: {out_path}")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Cross-Version Comparison (by Split)")
    print("=" * 70)
    
    for split_key in SPLITS_TO_COMPARE:
        print(f"\n📊 Processing split: {split_key}")
        
        try:
            rows = build_version_comparison_table(split_key)
            
            # Generate visualizations
            plot_eval_metrics_comparison(split_key, rows)
            plot_training_metrics_comparison(split_key, rows)
            save_comparison_csv(split_key, rows)
            
        except FileNotFoundError as e:
            print(f"⚠ Error loading metrics for {split_key}: {e}")
            continue
    
    print("\n" + "=" * 70)
    print(f"✅ All version comparisons saved to: {OUTPUT_ROOT}")
    print("=" * 70)


if __name__ == "__main__":
    main()
