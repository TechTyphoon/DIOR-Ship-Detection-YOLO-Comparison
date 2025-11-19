#!/usr/bin/env python3
"""
METRICS_VISUALIZER - Comprehensive metrics visualization module
================================================================

This module creates professional visualizations for:
1. Training curves (F1, Accuracy, Precision, Recall, Loss vs Epochs)
2. Precision-Recall curves
3. ROC-AUC curves
4. Per-class performance comparison
5. Metrics summary heatmap
6. Metrics data export (JSON, CSV)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    auc,
    precision_recall_curve,
    roc_curve,
    roc_auc_score,
)

logger = logging.getLogger(__name__)
sns.set_style("whitegrid")

__all__ = [
    'plot_training_curves',
    'plot_precision_recall_curve',
    'plot_roc_auc_curve',
    'plot_per_class_performance',
    'plot_confusion_matrix',
    'plot_metrics_summary',
    'save_metrics_data',
    'VisualizationManager',
]


# ============================================================================
# TRAINING CURVES VISUALIZATION
# ============================================================================

def plot_training_curves(
    metrics_history: Dict[str, List[float]],
    output_dir: str = 'runs/detect/metrics',
    dpi: int = 300
) -> None:
    """
    Plot training curves (F1, Accuracy, Precision, Recall, Loss vs Epochs).
    
    Args:
        metrics_history: Dict with keys like 'epoch', 'f1', 'accuracy', 'precision', 
                        'recall', 'train_loss', 'val_loss'
        output_dir: Directory to save charts
        dpi: DPI for output images
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    epochs = metrics_history.get('epoch', list(range(1, len(metrics_history.get('f1', [1])))))
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Training Metrics vs Epochs', fontsize=16, weight='bold', y=1.00)
    
    # F1 Score
    if 'f1' in metrics_history:
        axes[0, 0].plot(epochs, metrics_history['f1'], 'o-', linewidth=2, color='#1f77b4', markersize=5)
        axes[0, 0].set_title('F1 Score vs Epochs', fontsize=12, weight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('F1 Score')
        axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    if 'accuracy' in metrics_history:
        axes[0, 1].plot(epochs, metrics_history['accuracy'], 'o-', linewidth=2, color='#ff7f0e', markersize=5)
        axes[0, 1].set_title('Accuracy vs Epochs', fontsize=12, weight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Precision
    if 'precision' in metrics_history:
        axes[0, 2].plot(epochs, metrics_history['precision'], 'o-', linewidth=2, color='#2ca02c', markersize=5)
        axes[0, 2].set_title('Precision vs Epochs', fontsize=12, weight='bold')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Precision')
        axes[0, 2].grid(True, alpha=0.3)
    
    # Recall
    if 'recall' in metrics_history:
        axes[1, 0].plot(epochs, metrics_history['recall'], 'o-', linewidth=2, color='#d62728', markersize=5)
        axes[1, 0].set_title('Recall vs Epochs', fontsize=12, weight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Recall')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Train vs Val Loss
    if 'train_loss' in metrics_history and 'val_loss' in metrics_history:
        axes[1, 1].plot(epochs, metrics_history['train_loss'], 'o-', label='Train', linewidth=2, color='#9467bd', markersize=5)
        axes[1, 1].plot(epochs, metrics_history['val_loss'], 's-', label='Validation', linewidth=2, color='#e377c2', markersize=5)
        axes[1, 1].set_title('Loss vs Epochs', fontsize=12, weight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    # Combined Performance
    if all(k in metrics_history for k in ['f1', 'accuracy', 'precision', 'recall']):
        axes[1, 2].plot(epochs, metrics_history['f1'], 'o-', label='F1', linewidth=2, markersize=5)
        axes[1, 2].plot(epochs, metrics_history['accuracy'], 's-', label='Accuracy', linewidth=2, markersize=5)
        axes[1, 2].plot(epochs, metrics_history['precision'], '^-', label='Precision', linewidth=2, markersize=5)
        axes[1, 2].plot(epochs, metrics_history['recall'], 'd-', label='Recall', linewidth=2, markersize=5)
        axes[1, 2].set_title('All Metrics vs Epochs', fontsize=12, weight='bold')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Score')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'training_curves.png'
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    logger.info(f"✓ Training curves saved to {output_path}")
    plt.close()


# ============================================================================
# PRECISION-RECALL CURVE
# ============================================================================

def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    output_dir: str = 'runs/detect/metrics',
    dpi: int = 300,
    class_name: str = 'Ship'
) -> None:
    """
    Plot Precision-Recall curve.
    
    Args:
        y_true: True binary labels
        y_probs: Predicted probabilities
        output_dir: Directory to save chart
        dpi: DPI for output image
        class_name: Name of the positive class
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(recall, precision)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.plot(recall, precision, 'b-', linewidth=2.5, label=f'{class_name} (AUC = {pr_auc:.3f})')
    ax.fill_between(recall, precision, alpha=0.2, color='blue')
    
    ax.set_xlabel('Recall', fontsize=12, weight='bold')
    ax.set_ylabel('Precision', fontsize=12, weight='bold')
    ax.set_title(f'Precision-Recall Curve - {class_name}', fontsize=14, weight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'precision_recall_curve.png'
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    logger.info(f"✓ Precision-Recall curve saved to {output_path}")
    plt.close()


# ============================================================================
# ROC-AUC CURVE
# ============================================================================

def plot_roc_auc_curve(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    output_dir: str = 'runs/detect/metrics',
    dpi: int = 300,
    class_name: str = 'Ship'
) -> float:
    """
    Plot ROC-AUC curve.
    
    Args:
        y_true: True binary labels
        y_probs: Predicted probabilities
        output_dir: Directory to save chart
        dpi: DPI for output image
        class_name: Name of the positive class
        
    Returns:
        ROC-AUC score
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    roc_auc = roc_auc_score(y_true, y_probs)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # ROC curve
    ax.plot(fpr, tpr, 'b-', linewidth=2.5, label=f'{class_name} (AUC = {roc_auc:.3f})')
    ax.fill_between(fpr, tpr, alpha=0.2, color='blue')
    
    # Diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random Classifier')
    
    ax.set_xlabel('False Positive Rate', fontsize=12, weight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, weight='bold')
    ax.set_title(f'ROC-AUC Curve - {class_name}', fontsize=14, weight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'roc_auc_curve.png'
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    logger.info(f"✓ ROC-AUC curve saved to {output_path}")
    plt.close()
    
    return roc_auc


# ============================================================================
# PER-CLASS PERFORMANCE COMPARISON
# ============================================================================

def plot_per_class_performance(
    metrics_by_class: Dict[str, Dict[str, float]],
    output_dir: str = 'runs/detect/metrics',
    dpi: int = 300
) -> None:
    """
    Plot per-class performance comparison.
    
    Args:
        metrics_by_class: Dict with class names as keys, and dict of metrics as values
        output_dir: Directory to save chart
        dpi: DPI for output image
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    if not metrics_by_class:
        logger.warning("No per-class metrics provided")
        return
    
    classes = list(metrics_by_class.keys())
    metrics_types = ['precision', 'recall', 'f1']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(classes))
    width = 0.25
    
    for idx, metric in enumerate(metrics_types):
        values = [metrics_by_class.get(cls, {}).get(metric, 0) for cls in classes]
        ax.bar(x + idx * width, values, width, label=metric.capitalize())
    
    ax.set_ylabel('Score', fontsize=12, weight='bold')
    ax.set_xlabel('Class', fontsize=12, weight='bold')
    ax.set_title('Per-Class Performance Comparison', fontsize=14, weight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(classes)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.05])
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'per_class_performance.png'
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    logger.info(f"✓ Per-class performance saved to {output_path}")
    plt.close()


# ============================================================================
# CONFUSION MATRIX VISUALIZATION
# ============================================================================

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: str = 'runs/detect/metrics',
    dpi: int = 300,
    class_names: Optional[List[str]] = None
) -> None:
    """
    Plot confusion matrix heatmap.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        output_dir: Directory to save chart
        dpi: DPI for output image
        class_names: List of class names (default: ['Background', 'Ship'])
    """
    from sklearn.metrics import confusion_matrix
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    if class_names is None:
        class_names = ['Background', 'Ship']
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Plot heatmap
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d',
        cmap='Blues',
        cbar_kws={'label': 'Count'},
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        square=True,
        linewidths=2,
        linecolor='black'
    )
    
    ax.set_xlabel('Predicted Label', fontsize=12, weight='bold')
    ax.set_ylabel('True Label', fontsize=12, weight='bold')
    ax.set_title('Confusion Matrix', fontsize=14, weight='bold', pad=20)
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum() * 100
    
    # Add percentage annotations
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(
                j + 0.5, i + 0.7,
                f'({cm_percent[i, j]:.1f}%)',
                ha='center', va='center',
                fontsize=10, style='italic', color='darkblue'
            )
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'confusion_matrix.png'
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    logger.info(f"✓ Confusion matrix saved to {output_path}")
    plt.close()


# ============================================================================
# METRICS SUMMARY HEATMAP
# ============================================================================

def plot_metrics_summary(
    all_metrics: Dict[str, float],
    output_dir: str = 'runs/detect/metrics',
    dpi: int = 300
) -> None:
    """
    Plot metrics summary as heatmap.
    
    Args:
        all_metrics: Dict of all metrics with their values
        output_dir: Directory to save chart
        dpi: DPI for output image
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Filter metrics for heatmap
    heatmap_metrics = {
        'Accuracy': all_metrics.get('accuracy', 0),
        'Precision': all_metrics.get('precision', 0),
        'Recall': all_metrics.get('recall', 0),
        'F1-Score': all_metrics.get('f1', 0),
    }
    
    data = np.array([[v] for v in heatmap_metrics.values()])
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks([0])
    ax.set_xticklabels(['Score'])
    ax.set_yticks(range(len(heatmap_metrics)))
    ax.set_yticklabels(heatmap_metrics.keys())
    
    # Add text annotations
    for i, (metric, value) in enumerate(heatmap_metrics.items()):
        text = ax.text(0, i, f'{value:.3f}', ha='center', va='center', 
                      color='white' if value > 0.5 else 'black', fontsize=14, weight='bold')
    
    ax.set_title('Metrics Summary', fontsize=14, weight='bold', pad=20)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Score', fontsize=11, weight='bold')
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'metrics_summary.png'
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    logger.info(f"✓ Metrics summary saved to {output_path}")
    plt.close()


# ============================================================================
# DATA EXPORT
# ============================================================================

def save_metrics_data(
    all_metrics: Dict[str, any],
    metrics_history: Dict[str, List[float]],
    output_dir: str = 'runs/detect/metrics'
) -> None:
    """
    Save metrics data to JSON and CSV files.
    
    Args:
        all_metrics: Final metrics dictionary
        metrics_history: Historical metrics during training
        output_dir: Directory to save data files
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save as JSON
    json_path = Path(output_dir) / 'metrics.json'
    with open(json_path, 'w') as f:
        json.dump({
            'final_metrics': all_metrics,
            'metrics_history': metrics_history
        }, f, indent=2, default=str)
    logger.info(f"✓ Metrics JSON saved to {json_path}")
    
    # Save summary as CSV
    if metrics_history.get('epoch'):
        import csv
        csv_path = Path(output_dir) / 'metrics_history.csv'
        epochs = metrics_history['epoch']
        
        with open(csv_path, 'w', newline='') as f:
            fieldnames = ['epoch'] + [k for k in metrics_history.keys() if k != 'epoch']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for i, epoch in enumerate(epochs):
                row = {'epoch': epoch}
                for key in fieldnames[1:]:
                    if key in metrics_history:
                        row[key] = metrics_history[key][i] if i < len(metrics_history[key]) else ''
                writer.writerow(row)
        
        logger.info(f"✓ Metrics CSV saved to {csv_path}")


# ============================================================================
# VISUALIZATION MANAGER
# ============================================================================

class VisualizationManager:
    """Manager for all metrics visualizations."""
    
    def __init__(self, output_dir: str = 'runs/detect/metrics', dpi: int = 300):
        """Initialize visualization manager."""
        self.output_dir = output_dir
        self.dpi = dpi
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Metrics will be saved to {output_dir}")
    
    def create_all_visualizations(
        self,
        y_true: Optional[np.ndarray] = None,
        y_pred: Optional[np.ndarray] = None,
        y_probs: Optional[np.ndarray] = None,
        metrics_history: Optional[Dict[str, List[float]]] = None,
        all_metrics: Optional[Dict[str, float]] = None,
        per_class_metrics: Optional[Dict[str, Dict[str, float]]] = None
    ) -> None:
        """Create all visualizations at once."""
        
        # Training curves
        if metrics_history:
            plot_training_curves(metrics_history, self.output_dir, self.dpi)
        
        # Precision-Recall curve
        if y_true is not None and y_probs is not None:
            plot_precision_recall_curve(y_true, y_probs, self.output_dir, self.dpi)
        
        # ROC-AUC curve
        if y_true is not None and y_probs is not None:
            plot_roc_auc_curve(y_true, y_probs, self.output_dir, self.dpi)
        
        # Confusion matrix
        if y_true is not None and y_pred is not None:
            plot_confusion_matrix(y_true, y_pred, self.output_dir, self.dpi)
        
        # Per-class performance
        if per_class_metrics:
            plot_per_class_performance(per_class_metrics, self.output_dir, self.dpi)
        
        # Metrics summary
        if all_metrics:
            plot_metrics_summary(all_metrics, self.output_dir, self.dpi)
        
        # Save data
        if metrics_history or all_metrics:
            save_metrics_data(
                all_metrics or {},
                metrics_history or {},
                self.output_dir
            )
        
        logger.info("✓ All visualizations completed!")


if __name__ == '__main__':
    logger.basicConfig(level=logging.INFO)
    print("Metrics Visualizer Module - Ready to generate charts")
