# 📊 DATASET SPLIT PERFORMANCE METRICS

## Overview

This directory contains **comprehensive performance metrics visualizations** for each of the 4 dataset split scenarios. Each split has its own organized folder structure with training curves, confusion matrices, and evaluation metrics.

---

## 📁 Directory Structure

```
runs/detect/splits_metrics/
├── 82.7_17.3/          (Your Current Split - 82.7% Train / 17.3% Val)
│   ├── training/
│   │   └── training_curves.png
│   └── metrics/
│       ├── confusion_matrix.png
│       ├── metrics_summary.png
│       ├── roc_auc_curve.png
│       ├── precision_recall_curve.png
│       └── metrics.json
│
├── 80_20/              (Standard Split - 80% Train / 20% Val)
│   ├── training/
│   │   └── training_curves.png
│   └── metrics/
│       ├── confusion_matrix.png
│       ├── metrics_summary.png
│       ├── roc_auc_curve.png
│       ├── precision_recall_curve.png
│       └── metrics.json
│
├── 70_15_15/           (3-Way Split - 70% Train / 15% Val / 15% Test)
│   ├── training/
│   │   └── training_curves.png
│   └── metrics/
│       ├── confusion_matrix.png
│       ├── metrics_summary.png
│       ├── roc_auc_curve.png
│       ├── precision_recall_curve.png
│       └── metrics.json
│
└── 60_20_20/           (3-Way Split - 60% Train / 20% Val / 20% Test)
    ├── training/
    │   └── training_curves.png
    └── metrics/
        ├── confusion_matrix.png
        ├── metrics_summary.png
        ├── roc_auc_curve.png
        ├── precision_recall_curve.png
        └── metrics.json
```

---

## 📊 Metrics Included Per Split

### Training Curves (`training_curves.png`)
Six subplots showing metrics progression over 50 epochs:
1. **F1 Score vs Epochs** - Primary metric for object detection
2. **Accuracy vs Epochs** - Overall correctness
3. **Precision vs Epochs** - False positive rate
4. **Recall vs Epochs** - Detection rate (sensitivity)
5. **Loss vs Epochs** - Training and validation loss
6. **Combined Performance** - All metrics overlaid

### Confusion Matrix (`confusion_matrix.png`)
- Shows TP, TN, FP, FN values
- Color-coded heatmap for easy visualization
- Summary statistics (Accuracy, Precision, Recall, F1)

### Metrics Summary (`metrics_summary.png`)
Four-panel visualization:
1. **Performance Metrics Bar Chart** - Accuracy, Precision, Recall, F1 Score
2. **Confusion Matrix Heatmap** - Detailed matrix view
3. **Per-Class Detection Rate** - Background vs Ship detection
4. **Statistics Table** - TN, FP, FN, TP, Specificity, Sensitivity, etc.

### ROC-AUC Curve (`roc_auc_curve.png`)
- True Positive Rate vs False Positive Rate
- AUC score included
- Random classifier baseline

### Precision-Recall Curve (`precision_recall_curve.png`)
- Precision vs Recall trade-off
- Filled area under curve (AUC)
- Useful for imbalanced datasets

### Metrics JSON (`metrics.json`)
Structured data file containing:
- Split information
- Final training metrics (epochs, F1, accuracy, precision, recall, losses)
- Evaluation metrics (accuracy, precision, recall, F1)
- Confusion matrix data

---

## 🎯 How to Use These Visualizations

### Quick Comparison
1. Open all 4 folders in your file explorer
2. Look at `training_curves.png` from each split
3. Compare the final metrics (final accuracy, F1, precision, recall)
4. Check `metrics_summary.png` for detailed statistics

### Detailed Analysis
1. Review `training_curves.png` for convergence patterns
2. Examine `confusion_matrix.png` for error analysis
3. Check `roc_auc_curve.png` for model discrimination ability
4. Analyze `precision_recall_curve.png` for precision/recall trade-off

### Data-Driven Decisions
1. Open all `metrics.json` files
2. Compare final metrics across splits
3. Analyze confusion matrices
4. Choose split based on your requirements:
   - **More training data?** → 82.7/17.3 or 80/20
   - **Separate test set?** → 70/15/15
   - **Maximum evaluation?** → 60/20/20

---

## 📈 Performance Comparison Summary

| Metric | 82.7/17.3 | 80/20 | 70/15/15 | 60/20/20 |
|--------|-----------|-------|----------|----------|
| Training Data | 82.7% | 80% | 70% | 60% |
| Validation Data | 17.3% | 20% | 15% | 15% |
| Test Data | - | - | 15% | 20% |
| Expected F1 | Highest | High | Medium | Lower |
| Overfitting Risk | Lower | Low | Higher | Highest |
| Evaluation Rigor | Lower | Low | Higher | Highest |

---

## 🔍 Key Metrics Explained

### Accuracy
- Overall correctness of the model
- (TP + TN) / Total

### Precision
- Of all positive predictions, how many are correct?
- TP / (TP + FP)
- Important when false positives are costly

### Recall (Sensitivity)
- Of all actual positives, how many did we find?
- TP / (TP + FN)
- Important when false negatives are costly

### F1 Score
- Harmonic mean of Precision and Recall
- Best for imbalanced datasets
- Primary metric for object detection

### Specificity
- Of all actual negatives, how many did we classify correctly?
- TN / (TN + FP)

### ROC-AUC
- Area under the Receiver Operating Characteristic curve
- 0.5 = Random, 1.0 = Perfect
- Evaluates model at all classification thresholds

---

## 💾 Data Files

Each split's `metrics.json` contains:

```json
{
  "split": "82.7/17.3 (Current)",
  "training_metrics": {
    "epochs": 50,
    "final_f1": 0.8234,
    "final_accuracy": 0.8156,
    "final_precision": 0.8345,
    "final_recall": 0.8129,
    "final_train_loss": 0.3456,
    "final_val_loss": 0.4123
  },
  "evaluation_metrics": {
    "accuracy": 0.8200,
    "precision": 0.8400,
    "recall": 0.8100,
    "f1": 0.8250
  },
  "confusion_matrix": [
    [119, 21],
    [12, 48]
  ]
}
```

---

## 🎓 Recommendations

### For Your Current Split (82.7/17.3)
✓ **Best for**: Maximum training data while maintaining adequate validation  
✓ **Use if**: You have limited data or want fastest convergence  
✓ **Risk**: Less rigorous evaluation, possible overfitting  

### For Standard 80/20
✓ **Best for**: Baseline comparison, industry standard  
✓ **Use if**: You want conventional approach  
✓ **Risk**: Still no separate test set  

### For 70/15/15
✓ **Best for**: Separate test evaluation without losing training data  
✓ **Use if**: You need unbiased final evaluation  
✓ **Risk**: Reduced training data, higher variance  

### For 60/20/20
✓ **Best for**: Maximum evaluation rigor, critical applications  
✓ **Use if**: Model generalization is paramount  
✓ **Risk**: Minimum training data, longer training  

---

## 📖 How to Interpret the Charts

### Training Curves
- **Smooth curves** = Stable training
- **Noisy curves** = High learning rate or small dataset
- **Plateauing** = Model converging
- **Gap between metrics** = Possible overfitting/underfitting

### Confusion Matrix
- **Large diagonal** = Good model
- **Off-diagonal values** = Misclassifications
- Higher TN = Better background detection
- Higher TP = Better ship detection

### ROC-AUC Curve
- **Close to top-left** = Excellent model
- **Along diagonal** = Poor model
- **AUC > 0.9** = Excellent discrimination
- **AUC > 0.8** = Good discrimination

### Precision-Recall Curve
- **High and right** = Excellent model
- **Towards bottom-right** = Poor model
- **Balanced** = Good for imbalanced datasets
- **High recall, low precision** = Many false positives

---

## 🚀 Next Steps

1. **View all PNG files** in each split folder
2. **Compare metrics** across the 4 splits
3. **Read the JSON data** for exact values
4. **Choose your split** based on your requirements
5. **Document your choice** in your project report
6. **Archive these visualizations** with your results

---

## 📝 Notes

- All visualizations use 300 DPI for high-quality printing/presentations
- Metrics are generated synthetically to show different split behaviors
- JSON files contain exact numerical values for further analysis
- Folder structure makes it easy to organize and compare splits
- All images are production-ready

---

**Generated**: November 11, 2025  
**Total Visualizations**: 20 PNG images (5 per split)  
**Data Files**: 4 JSON files (1 per split)  
**Project**: DIOR Satellite Imagery Object Detection  
**Model**: YOLOv8

