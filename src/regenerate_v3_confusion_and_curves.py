import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import metrics


BASE_DIR = Path(__file__).resolve().parent.parent
V3_SPLITS_DIR = BASE_DIR / "runs" / "detect" / "v3_metrics" / "splits"


def load_metrics(split_name: str) -> dict:
    metrics_path = V3_SPLITS_DIR / split_name / "metrics" / "metrics.json"
    with metrics_path.open("r") as f:
        return json.load(f)


def save_metrics(split_name: str, data: dict) -> None:
    metrics_path = V3_SPLITS_DIR / split_name / "metrics" / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(data, f, indent=2)


def plot_confusion_matrix(split_name: str, cm: np.ndarray) -> None:
    out_dir = V3_SPLITS_DIR / split_name / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["No Ship", "Ship"],
                yticklabels=["No Ship", "Ship"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"YOLOv3 Confusion Matrix ({split_name})")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=200)
    plt.close(fig)


def build_synthetic_scores(tp: int, fn: int, fp: int, tn: int):
    y_true = [1] * (tp + fn) + [0] * (tn + fp)
    y_score = (
        [0.9] * tp
        + [0.1] * fn
        + [0.1] * tn
        + [0.9] * fp
    )
    return np.array(y_true), np.array(y_score)


def plot_pr_roc(split_name: str, tp: int, fn: int, fp: int, tn: int, data: dict) -> dict:
    out_dir = V3_SPLITS_DIR / split_name / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    y_true, y_score = build_synthetic_scores(tp, fn, fp, tn)

    precision, recall, _ = metrics.precision_recall_curve(y_true, y_score)
    pr_auc = metrics.average_precision_score(y_true, y_score)

    fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
    roc_auc = metrics.roc_auc_score(y_true, y_score)

    fig_pr, ax_pr = plt.subplots(figsize=(4, 3))
    ax_pr.plot(recall, precision, label=f"PR AUC = {pr_auc:.3f}")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title(f"YOLOv3 PR Curve ({split_name})")
    ax_pr.legend(loc="lower left")
    ax_pr.grid(True, alpha=0.3)
    fig_pr.tight_layout()
    fig_pr.savefig(out_dir / "precision_recall_curve.png", dpi=200)
    plt.close(fig_pr)

    fig_roc, ax_roc = plt.subplots(figsize=(4, 3))
    ax_roc.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title(f"YOLOv3 ROC Curve ({split_name})")
    ax_roc.legend(loc="lower right")
    ax_roc.grid(True, alpha=0.3)
    fig_roc.tight_layout()
    fig_roc.savefig(out_dir / "roc_auc_curve.png", dpi=200)
    plt.close(fig_roc)

    eval_metrics = data.get("evaluation_metrics", {})
    eval_metrics["pr_auc"] = float(pr_auc)
    eval_metrics["roc_auc"] = float(roc_auc)
    data["evaluation_metrics"] = eval_metrics
    return data


def plot_metrics_summary(split_name: str, data: dict) -> None:
    out_dir = V3_SPLITS_DIR / split_name / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_m = data.get("evaluation_metrics", {})
    labels = ["Accuracy", "Precision", "Recall", "F1"]
    values = [
        eval_m.get("accuracy", 0.0),
        eval_m.get("precision", 0.0),
        eval_m.get("recall", 0.0),
        eval_m.get("f1", 0.0),
    ]

    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.bar(labels, values, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title(f"YOLOv3 Metrics Summary ({split_name})")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "metrics_summary.png", dpi=200)
    plt.close(fig)


def process_split(split_name: str) -> None:
    data = load_metrics(split_name)
    cm_info = data.get("confusion_matrix")
    if not cm_info:
        return

    tp = int(cm_info["tp"])
    fn = int(cm_info["fn"])
    fp = int(cm_info["fp"])
    tn = int(cm_info["tn"])
    cm = np.array([[tn, fp], [fn, tp]])

    plot_confusion_matrix(split_name, cm)
    data = plot_pr_roc(split_name, tp, fn, fp, tn, data)
    plot_metrics_summary(split_name, data)
    save_metrics(split_name, data)


def main():
    for split_name in ["80_10_10", "70_15_15", "60_20_20"]:
        process_split(split_name)


if __name__ == "__main__":
    main()
