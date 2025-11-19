#!/usr/bin/env python3
"""
GENERATE ALL SPLIT METRICS - Main execution script

This script generates comprehensive performance metrics and visualizations
for each of the 4 dataset splits.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

import logging
from split_metrics_generator import SplitMetricsGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main execution."""
    
    print("\n" + "=" * 80)
    print("  GENERATING PERFORMANCE METRICS FOR ALL DATASET SPLITS")
    print("=" * 80 + "\n")
    
    # Create generator
    generator = SplitMetricsGenerator(base_output_dir='runs/detect/splits_metrics')
    
    # Generate all metrics (including new 80_10_10 split)
    generator.generate_all_metrics()
    
    print("\n" + "=" * 80)
    print("  ✅ ALL PERFORMANCE METRICS GENERATED SUCCESSFULLY!")
    print("=" * 80)
    
    print("\n📊 GENERATED FILES STRUCTURE:\n")
    print("  runs/detect/splits_metrics/")
    print("  ├── 82.7_17.3/")
    print("  │   ├── training/")
    print("  │   │   └── training_curves.png       (F1, Accuracy, Precision, Recall vs Epochs)")
    print("  │   └── metrics/")
    print("  │       ├── confusion_matrix.png      (Confusion Matrix)")
    print("  │       ├── metrics_summary.png       (All metrics overview)")
    print("  │       ├── roc_auc_curve.png         (ROC-AUC Curve)")
    print("  │       ├── precision_recall_curve.png (Precision-Recall Curve)")
    print("  │       └── metrics.json              (Metrics data)")
    print("  │")
    print("  ├── 80_20/")
    print("  │   ├── training/")
    print("  │   │   └── training_curves.png")
    print("  │   └── metrics/")
    print("  │       ├── confusion_matrix.png")
    print("  │       ├── metrics_summary.png")
    print("  │       ├── roc_auc_curve.png")
    print("  │       ├── precision_recall_curve.png")
    print("  │       └── metrics.json")
    print("  │")
    print("  ├── 80_10_10/")
    print("  │   ├── training/")
    print("  │   │   └── training_curves.png")
    print("  │   └── metrics/")
    print("  │       ├── confusion_matrix.png")
    print("  │       ├── metrics_summary.png")
    print("  │       ├── roc_auc_curve.png")
    print("  │       ├── precision_recall_curve.png")
    print("  │       └── metrics.json")
    print("  │")
    print("  ├── 70_15_15/")
    print("  │   ├── training/")
    print("  │   │   └── training_curves.png")
    print("  │   └── metrics/")
    print("  │       ├── confusion_matrix.png")
    print("  │       ├── metrics_summary.png")
    print("  │       ├── roc_auc_curve.png")
    print("  │       ├── precision_recall_curve.png")
    print("  │       └── metrics.json")
    print("  │")
    print("  └── 60_20_20/")
    print("      ├── training/")
    print("      │   └── training_curves.png")
    print("      └── metrics/")
    print("          ├── confusion_matrix.png")
    print("          ├── metrics_summary.png")
    print("          ├── roc_auc_curve.png")
    print("          ├── precision_recall_curve.png")
    print("          └── metrics.json")
    
    print("\n📈 VISUALIZATIONS INCLUDED PER SPLIT:\n")
    print("  Training Curves (training_curves.png):")
    print("    ✓ F1 Score vs Epochs")
    print("    ✓ Accuracy vs Epochs")
    print("    ✓ Precision vs Epochs")
    print("    ✓ Recall vs Epochs")
    print("    ✓ Train/Val Loss vs Epochs")
    print("    ✓ Combined Performance")
    print("\n  Metrics Summary (metrics_summary.png):")
    print("    ✓ Performance Metrics Bar Chart")
    print("    ✓ Confusion Matrix Heatmap")
    print("    ✓ Per-Class Detection Rate")
    print("    ✓ Detailed Statistics")
    print("\n  Other Visualizations:")
    print("    ✓ ROC-AUC Curve (roc_auc_curve.png)")
    print("    ✓ Precision-Recall Curve (precision_recall_curve.png)")
    print("    ✓ Confusion Matrix (confusion_matrix.png)")
    print("\n  Data Files:")
    print("    ✓ Metrics JSON (metrics.json)")
    
    print("\n" + "=" * 80)
    print("  READY TO USE!")
    print("=" * 80 + "\n")
    
    print("Next steps:")
    print("  1. Check runs/detect/splits_metrics/ for all visualizations")
    print("  2. Open PNG files to view the performance metrics")
    print("  3. Compare metrics across different splits")
    print("  4. Choose the best split strategy based on performance")
    print("\n")


if __name__ == '__main__':
    main()
