#!/usr/bin/env python3
"""
Chart Generation Script - Generates all metrics visualizations
This script creates sample data and generates all charts for demonstration.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import numpy as np
import logging
from metrics_visualizer import VisualizationManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_realistic_data():
    """Generate realistic sample data for visualization."""
    
    np.random.seed(42)
    n_samples = 1000
    
    # Generate realistic binary predictions (ship detection)
    y_true = np.random.binomial(1, 0.3, n_samples)  # 30% ships, 70% background
    y_probs = np.random.beta(2, 2, n_samples)  # Beta distribution for probabilities
    
    # Make predictions correlate with ground truth
    for i in range(n_samples):
        if y_true[i] == 1:
            y_probs[i] = np.random.beta(5, 2, 1)[0]  # Higher prob for ships
        else:
            y_probs[i] = np.random.beta(2, 5, 1)[0]  # Lower prob for background
    
    y_pred = (y_probs > 0.5).astype(int)
    
    logger.info(f"Generated {n_samples} samples")
    logger.info(f"Positive class (ships): {y_true.sum()} samples")
    logger.info(f"Negative class (background): {(1 - y_true).sum()} samples")
    
    return y_true, y_pred, y_probs


def generate_training_history():
    """Generate realistic training history."""
    epochs = list(range(1, 51))  # 50 epochs
    
    # Create realistic curves with some noise
    metrics_history = {
        'epoch': epochs,
        'f1': [0.45 + i * 0.008 + np.random.normal(0, 0.01) for i in epochs],
        'accuracy': [0.50 + i * 0.007 + np.random.normal(0, 0.01) for i in epochs],
        'precision': [0.55 + i * 0.006 + np.random.normal(0, 0.01) for i in epochs],
        'recall': [0.40 + i * 0.009 + np.random.normal(0, 0.01) for i in epochs],
        'train_loss': [2.5 - i * 0.035 + np.random.normal(0, 0.05) for i in epochs],
        'val_loss': [2.6 - i * 0.032 + np.random.normal(0, 0.06) for i in epochs],
    }
    
    # Ensure values are in valid ranges
    for key in ['f1', 'accuracy', 'precision', 'recall']:
        metrics_history[key] = np.clip(metrics_history[key], 0, 1).tolist()
    for key in ['train_loss', 'val_loss']:
        metrics_history[key] = np.clip(metrics_history[key], 0, None).tolist()
    
    return metrics_history


def main():
    """Main execution function."""
    
    logger.info("=" * 70)
    logger.info("METRICS VISUALIZATION GENERATOR")
    logger.info("=" * 70)
    
    # Create output directory
    output_dir = 'runs/detect/metrics'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\nOutput directory: {output_dir}")
    
    # Generate data
    logger.info("\n1. Generating sample data...")
    y_true, y_pred, y_probs = generate_realistic_data()
    
    logger.info("2. Generating training history...")
    metrics_history = generate_training_history()
    
    # Calculate final metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    all_metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_probs),
    }
    
    logger.info(f"\nFinal Metrics:")
    for metric, value in all_metrics.items():
        logger.info(f"  {metric.upper()}: {value:.4f}")
    
    # Per-class metrics
    per_class_metrics = {
        'Ship': {
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
        },
        'Background': {
            'precision': precision_score(1 - y_true, 1 - y_pred),
            'recall': recall_score(1 - y_true, 1 - y_pred),
            'f1': f1_score(1 - y_true, 1 - y_pred),
        }
    }
    
    # Create visualizations
    logger.info("\n3. Creating visualizations...")
    manager = VisualizationManager(output_dir=output_dir, dpi=300)
    
    manager.create_all_visualizations(
        y_true=y_true,
        y_pred=y_pred,
        y_probs=y_probs,
        metrics_history=metrics_history,
        all_metrics=all_metrics,
        per_class_metrics=per_class_metrics
    )
    
    # List generated files
    logger.info("\n4. Generated charts:")
    charts_dir = Path(output_dir)
    for chart_file in sorted(charts_dir.glob('*.png')):
        file_size = chart_file.stat().st_size / 1024  # Size in KB
        logger.info(f"  ✓ {chart_file.name} ({file_size:.1f} KB)")
    
    logger.info("\n5. Generated data files:")
    for data_file in sorted(charts_dir.glob('*.json')) + sorted(charts_dir.glob('*.csv')):
        file_size = data_file.stat().st_size / 1024  # Size in KB
        logger.info(f"  ✓ {data_file.name} ({file_size:.1f} KB)")
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
    logger.info("=" * 70)
    logger.info(f"\nAll files saved to: {output_dir}")
    logger.info("\nGenerated files:")
    logger.info("  1. training_curves.png - F1, Accuracy, Precision, Recall, Loss vs Epochs")
    logger.info("  2. precision_recall_curve.png - Precision-Recall curve")
    logger.info("  3. roc_auc_curve.png - ROC-AUC curve")
    logger.info("  4. confusion_matrix.png - Confusion Matrix heatmap")
    logger.info("  5. per_class_performance.png - Per-class metrics comparison")
    logger.info("  6. metrics_summary.png - Heatmap of final metrics")
    logger.info("  7. metrics.json - All metrics data in JSON format")
    logger.info("  8. metrics_history.csv - Training history in CSV format")


if __name__ == '__main__':
    main()
