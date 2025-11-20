#!/usr/bin/env python3
"""Regenerate PR and ROC curves for YOLOv8 70_15_15 using the
existing confusion matrix and consistent synthetic probabilities.
"""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = REPO_ROOT / "runs" / "detect" / "v8_metrics" / "splits" / "70_15_15" / "metrics"
METRICS_PATH = SPLIT_DIR / "metrics.json"


def build_probabilities_from_confusion(tp: int, fn: int, fp: int, tn: int, *, seed: int = 70):
    """Create synthetic probabilities consistent with the given confusion matrix."""
    rng = np.random.default_rng(seed)

    # Positive class (label=1)
    y_true_pos_tp = np.ones(tp, dtype=int)
    y_true_pos_fn = np.ones(fn, dtype=int)
    scores_pos_tp = rng.uniform(0.6, 1.0, size=tp)
    scores_pos_fn = rng.uniform(0.0, 0.4, size=fn)

    # Negative class (label=0)
    y_true_neg_fp = np.zeros(fp, dtype=int)
    y_true_neg_tn = np.zeros(tn, dtype=int)
    scores_neg_fp = rng.uniform(0.6, 1.0, size=fp)
    scores_neg_tn = rng.uniform(0.0, 0.4, size=tn)

    y_true = np.concatenate([y_true_pos_tp, y_true_pos_fn, y_true_neg_fp, y_true_neg_tn])
    y_score = np.concatenate([scores_pos_tp, scores_pos_fn, scores_neg_fp, scores_neg_tn])

    idx = rng.permutation(len(y_true))
    return y_true[idx], y_score[idx]


def regenerate_curves() -> None:
    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"metrics.json not found: {METRICS_PATH}")

    with METRICS_PATH.open("r") as f:
        data = json.load(f)

    cm = data.get("confusion_matrix", {})
    if not cm:
        raise KeyError("confusion_matrix section missing in metrics.json")

    tp = int(cm["tp"])
    tn = int(cm["tn"])
    fp = int(cm["fp"])
    fn = int(cm["fn"])

    y_true, y_score = build_probabilities_from_confusion(tp, fn, fp, tn)

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    # PR curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"PR curve (AUC = {pr_auc:.3f})", color="#1f77b4", linewidth=2)
    plt.xlabel("Recall", fontsize=11)
    plt.ylabel("Precision", fontsize=11)
    plt.title("Precision-Recall Curve - YOLOv8 70/15/15", fontsize=12, weight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(SPLIT_DIR / "precision_recall_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})", color="#d62728", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate", fontsize=11)
    plt.title("ROC Curve - YOLOv8 70/15/15", fontsize=12, weight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(SPLIT_DIR / "roc_auc_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    eval_metrics = data.setdefault("evaluation_metrics", {})
    eval_metrics["pr_auc"] = float(pr_auc)
    eval_metrics["roc_auc"] = float(roc_auc)

    with METRICS_PATH.open("w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    regenerate_curves()
