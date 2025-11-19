"""
UTILS_METRICS - Metric computation and cross-validation utilities
==================================================================

This module provides:
1. Comprehensive metric computation with cross-checks
2. Manual metric calculation (without sklearn) for validation
3. Confusion matrix utilities
4. Per-class and macro/micro averaging

All metrics are validated against sklearn implementations.
"""

import logging
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_fscore_support,
)

__all__ = [
    'compute_metrics',
    'cross_check_metrics',
    'verify_confusion_matrix',
    'validate_predictions',
    'get_class_weights',
    'compute_roc_auc',
    'manual_confusion_matrix',
    'manual_accuracy',
    'manual_precision',
    'manual_recall',
    'manual_f1_score',
]

logger = logging.getLogger(__name__)

# Floating point tolerance for cross-checks
TOLERANCE = 1e-6


# ============================================================================
# MANUAL METRIC CALCULATIONS (for cross-validation)
# ============================================================================

def manual_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 2
) -> np.ndarray:
    """
    Manually compute confusion matrix without sklearn.
    For validation/cross-checking purposes.
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred):
        cm[int(true_label), int(pred_label)] += 1
    return cm


def manual_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Manually compute accuracy without sklearn."""
    return np.sum(y_true == y_pred) / len(y_true)


def manual_precision(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_idx: int = 1
) -> float:
    """
    Manually compute precision for a specific class.
    precision = TP / (TP + FP)
    """
    cm = manual_confusion_matrix(y_true, y_pred, num_classes=2)
    tp = cm[class_idx, class_idx]
    fp = cm[1 - class_idx, class_idx]  # False positives (predicted as class but aren't)
    
    if tp + fp == 0:
        return 0.0
    
    return tp / (tp + fp)


def manual_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_idx: int = 1
) -> float:
    """
    Manually compute recall for a specific class.
    recall = TP / (TP + FN)
    """
    cm = manual_confusion_matrix(y_true, y_pred, num_classes=2)
    tp = cm[class_idx, class_idx]
    fn = cm[class_idx, 1 - class_idx]  # False negatives
    
    if tp + fn == 0:
        return 0.0
    
    return tp / (tp + fn)


def manual_f1_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_idx: int = 1
) -> float:
    """Manually compute F1-score."""
    precision = manual_precision(y_true, y_pred, class_idx=class_idx)
    recall = manual_recall(y_true, y_pred, class_idx=class_idx)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)


# ============================================================================
# PRIMARY METRIC COMPUTATION
# ============================================================================

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 2
) -> Dict[str, float]:
    """
    Compute comprehensive metrics for binary classification.
    
    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted labels (0 or 1)
        num_classes: Number of classes (default 2 for binary)
    
    Returns:
        Dictionary with accuracy, precision, recall, f1, etc.
    """
    # Validate input shapes
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")
    
    if len(y_true) == 0:
        raise ValueError("Empty predictions")
    
    # Ensure arrays are numpy
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)
    
    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Per-class metrics using sklearn
    precision_arr, recall_arr, f1_arr, support_arr = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    
    # Macro and micro averages
    precision_macro = np.mean(precision_arr)
    recall_macro = np.mean(recall_arr)
    f1_macro = np.mean(f1_arr)
    
    # Micro averages (for binary, equals overall metrics)
    precision_micro = precision_score(y_true, y_pred, average='micro')
    recall_micro = recall_score(y_true, y_pred, average='micro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    
    # Weighted averages (weighted by support/class frequency)
    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Class 1 specific metrics (for binary classification, often the positive class)
    precision_class1 = precision_arr[1]
    recall_class1 = recall_arr[1]
    f1_class1 = f1_arr[1]
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision_macro),  # Default to macro for binary
        'recall': float(recall_macro),
        'f1_score': float(f1_macro),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'f1_macro': float(f1_macro),
        'precision_micro': float(precision_micro),
        'recall_micro': float(recall_micro),
        'f1_micro': float(f1_micro),
        'precision_weighted': float(precision_weighted),
        'recall_weighted': float(recall_weighted),
        'f1_weighted': float(f1_weighted),
        'precision_class0': float(precision_arr[0]),
        'recall_class0': float(recall_arr[0]),
        'f1_class0': float(f1_arr[0]),
        'precision_class1': float(precision_class1),
        'recall_class1': float(recall_class1),
        'f1_class1': float(f1_class1),
        'support_class0': int(support_arr[0]),
        'support_class1': int(support_arr[1]),
    }
    
    return metrics


# ============================================================================
# CROSS-VALIDATION & VERIFICATION
# ============================================================================

def cross_check_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Cross-check metrics computed by sklearn against manual calculations.
    Logs any mismatches. Raises assertion error if tolerance exceeded.
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)
    
    # Check accuracy
    acc_sklearn = accuracy_score(y_true, y_pred)
    acc_manual = manual_accuracy(y_true, y_pred)
    assert abs(acc_sklearn - acc_manual) < TOLERANCE, \
        f"Accuracy mismatch: sklearn={acc_sklearn}, manual={acc_manual}"
    
    # Check precision (class 1)
    prec_sklearn = precision_score(y_true, y_pred, average=None, zero_division=0)[1]
    prec_manual = manual_precision(y_true, y_pred, class_idx=1)
    assert abs(prec_sklearn - prec_manual) < TOLERANCE, \
        f"Precision mismatch: sklearn={prec_sklearn}, manual={prec_manual}"
    
    # Check recall (class 1)
    rec_sklearn = recall_score(y_true, y_pred, average=None, zero_division=0)[1]
    rec_manual = manual_recall(y_true, y_pred, class_idx=1)
    assert abs(rec_sklearn - rec_manual) < TOLERANCE, \
        f"Recall mismatch: sklearn={rec_sklearn}, manual={rec_manual}"
    
    # Check F1 (class 1)
    f1_sklearn = f1_score(y_true, y_pred, average=None, zero_division=0)[1]
    f1_manual = manual_f1_score(y_true, y_pred, class_idx=1)
    assert abs(f1_sklearn - f1_manual) < TOLERANCE, \
        f"F1 mismatch: sklearn={f1_sklearn}, manual={f1_manual}"
    
    # Check confusion matrix
    cm_sklearn = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_manual = manual_confusion_matrix(y_true, y_pred, num_classes=2)
    assert np.array_equal(cm_sklearn, cm_manual), \
        f"Confusion matrix mismatch:\nsklearn:\n{cm_sklearn}\nmanual:\n{cm_manual}"
    
    logger.info("✓ All metric cross-checks passed")


def verify_confusion_matrix(
    cm: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> None:
    """
    Verify confusion matrix is consistent with predictions.
    Checks that sum(cm) == len(y_true).
    """
    expected_total = len(y_true)
    actual_total = cm.sum()
    
    assert actual_total == expected_total, \
        f"Confusion matrix sum mismatch: expected {expected_total}, got {actual_total}"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_class_weights(y_true: np.ndarray) -> Tuple[float, float]:
    """
    Compute class weights to handle imbalance.
    Returns: (weight_class0, weight_class1)
    """
    unique, counts = np.unique(y_true, return_counts=True)
    total = len(y_true)
    
    weights = {}
    for label, count in zip(unique, counts):
        # Inverse frequency weighting
        weights[int(label)] = total / (2 * count)
    
    # Ensure both classes have weights
    if 0 not in weights:
        weights[0] = 1.0
    if 1 not in weights:
        weights[1] = 1.0
    
    return weights[0], weights[1]


def compute_roc_auc(y_true: np.ndarray, y_probs: np.ndarray) -> float:
    """
    Compute ROC-AUC score.
    
    Args:
        y_true: Ground truth binary labels
        y_probs: Predicted probabilities for class 1
    
    Returns:
        ROC-AUC score
    """
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, y_probs))
    except Exception as e:
        logger.warning(f"Could not compute ROC-AUC: {e}")
        return float('nan')


# ============================================================================
# SANITY CHECKS
# ============================================================================

def validate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> None:
    """
    Validate that predictions are well-formed.
    Checks:
    - All labels are 0 or 1
    - No NaN or inf values
    - Shapes match
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Shape check
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    
    # Check for NaN/inf
    if np.any(np.isnan(y_true)) or np.any(np.isinf(y_true)):
        raise ValueError("y_true contains NaN or inf")
    if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
        raise ValueError("y_pred contains NaN or inf")
    
    # Check values are 0 or 1
    unique_true = np.unique(y_true)
    unique_pred = np.unique(y_pred)
    
    for val in unique_true:
        if val not in (0, 1):
            raise ValueError(f"y_true contains invalid label: {val}")
    
    for val in unique_pred:
        if val not in (0, 1):
            raise ValueError(f"y_pred contains invalid label: {val}")
    
    logger.debug("✓ Predictions validated successfully")
