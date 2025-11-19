# Dataset Split Visualizations

## Overview

This document explains the dataset split visualizations generated for comparing different train/validation/test split scenarios for the DIOR satellite imagery dataset.

## Generated Visualizations

All visualizations have been saved to `runs/detect/metrics/` directory.

### 1. Individual Split Pie Charts

#### 📊 Current Split: 82.7% : 17.3%
- **File:** `split_82.7_17.3(Current).png`
- **Format:** 2-way split (Train : Val)
- **Details:**
  - Train: 82.7% (827 samples)
  - Val: 17.3% (173 samples)
- **Note:** This is your current dataset split, which is close to the standard 80/20 split

#### 📊 Standard 80/20 Split
- **File:** `split_80_20.png`
- **Format:** 2-way split (Train : Val)
- **Details:**
  - Train: 80.0% (800 samples)
  - Val: 20.0% (200 samples)
- **Use Case:** Standard split for binary classification tasks

#### 📊 Balanced 70/15/15 Split
- **File:** `split_70_15_15.png`
- **Format:** 3-way split (Train : Val : Test)
- **Details:**
  - Train: 70.0% (700 samples)
  - Val: 15.0% (150 samples)
  - Test: 15.0% (150 samples)
- **Use Case:** When you want separate validation and test sets for unbiased evaluation

#### 📊 Balanced 60/20/20 Split
- **File:** `split_60_20_20.png`
- **Format:** 3-way split (Train : Val : Test)
- **Details:**
  - Train: 60.0% (600 samples)
  - Val: 20.0% (200 samples)
  - Test: 20.0% (200 samples)
- **Use Case:** More generous validation/test sets for better generalization assessment

---

### 2. Comparison Visualizations

#### 📈 All Splits Side-by-Side (Pie Charts)
- **File:** `splits_comparison_pie.png`
- **Description:** All four split scenarios displayed as pie charts in a 2x2 grid for easy visual comparison
- **Purpose:** Quick visual comparison of all split ratios

#### 📊 Bar Chart Comparison
- **File:** `splits_comparison_bars.png`
- **Description:** Bar chart showing the distribution percentages for each dataset part (Train, Val, Test) across all split scenarios
- **Purpose:** Precise numerical comparison of splits

#### 📋 Comparison Table
- **File:** `splits_comparison_table.png`
- **Description:** Detailed table showing percentages and sample counts for each split scenario
- **Purpose:** Reference for exact values and sample distributions

---

## Split Recommendations

### Choose **82.7/17.3 or 80/20** if:
- ✅ You have enough data for reliable training
- ✅ You don't need a separate test set
- ✅ You trust your validation set for final evaluation
- ✅ Training data is your priority
- ✅ **Good for:** Image classification with single split

### Choose **70/15/15** if:
- ✅ You want a dedicated test set for unbiased evaluation
- ✅ You have sufficient data to support 3-way split
- ✅ Validation and test sets are equally important
- ✅ **Good for:** Object detection with independent test evaluation

### Choose **60/20/20** if:
- ✅ You want maximum validation/test coverage
- ✅ You have plenty of data available
- ✅ Model generalization is critical
- ✅ You can afford to train on less data
- ✅ **Good for:** High-stakes detection tasks requiring robust evaluation

---

## Sample Count Reference

For a dataset of **1,000 samples** (shown in visualizations):

| Split Scenario | Train | Val | Test | Total |
|---|---|---|---|---|
| **82.7/17.3 (Current)** | 827 | 173 | - | 1,000 |
| **80/20** | 800 | 200 | - | 1,000 |
| **70/15/15** | 700 | 150 | 150 | 1,000 |
| **60/20/20** | 600 | 200 | 200 | 1,000 |

*Scale these numbers to your actual dataset size*

---

## How to Use These Visualizations

1. **Visual Comparison:** Use pie charts to understand the proportions at a glance
2. **Detailed Analysis:** Refer to the comparison table for exact percentages and sample counts
3. **Decision Making:** Choose the split that best matches your project requirements
4. **Documentation:** Include these visualizations in project reports or documentation
5. **Presentation:** Use for stakeholder presentations to explain data distribution

---

## Technical Details

- **Total Samples (Reference):** 1,000
- **DPI:** 300 (high resolution for printing)
- **Generated:** Using `dataset_split_visualizer.py`
- **Library:** Matplotlib & Seaborn

---

## YOLOv8 Dataset Structure

When using your chosen split with YOLOv8, organize your data as:

```
datasets/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
```

For 2-way splits (82.7/17.3 or 80/20), omit the `test/` directories.

---

## Next Steps

1. **Decide on a split strategy** based on your needs
2. **Check your actual dataset size** and scale the percentages accordingly
3. **Reorganize your data** into the chosen split structure
4. **Update your training config** (e.g., `data/configs/config.yaml`)
5. **Retrain your model** with the new split if needed

---

## Additional Resources

- **YOLO Documentation:** https://docs.ultralytics.com/
- **DIOR Dataset:** http://www.captain-whu.github.io/DiOR/
- **Train/Val/Test Best Practices:** https://scikit-learn.org/stable/modules/cross_validation.html

---

Generated: November 11, 2025
