"""
TEST_METRICS - Unit tests for metric computation and evaluation pipeline
=========================================================================

Tests include:
1. Metric computation on known arrays with expected results
2. Cross-checking metrics with manual calculations
3. Confusion matrix creation and validation
4. Inference pipeline with dummy models
5. Input validation and edge cases

Run with: pytest test_metrics.py -v
"""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

import utils_metrics


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def simple_predictions():
    """Simple known predictions for validation."""
    y_true = np.array([1, 0, 1, 0, 1, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 1, 0, 0, 1, 1, 0, 0, 1, 0])
    return y_true, y_pred


@pytest.fixture
def perfect_predictions():
    """Perfect predictions (all correct)."""
    y_true = np.array([1, 0, 1, 0, 1, 1])
    y_pred = np.array([1, 0, 1, 0, 1, 1])
    return y_true, y_pred


@pytest.fixture
def worst_predictions():
    """Worst predictions (all wrong)."""
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    y_pred = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    return y_true, y_pred


@pytest.fixture
def imbalanced_predictions():
    """Imbalanced class distribution."""
    y_true = np.array([1] * 90 + [0] * 10)
    y_pred = np.array([1] * 85 + [0] * 5 + [0] * 5 + [1] * 5)
    return y_true, y_pred


# ============================================================================
# TESTS: METRIC COMPUTATION
# ============================================================================

class TestMetricComputation:
    """Tests for basic metric computation."""
    
    def test_compute_metrics_perfect_predictions(self, perfect_predictions):
        """With perfect predictions, all metrics should be 1.0."""
        y_true, y_pred = perfect_predictions
        metrics = utils_metrics.compute_metrics(y_true, y_pred)
        
        assert metrics['accuracy'] == 1.0
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1_score'] == 1.0
        assert metrics['precision_class0'] == 1.0
        assert metrics['precision_class1'] == 1.0
    
    def test_compute_metrics_worst_predictions(self, worst_predictions):
        """With all wrong predictions, accuracy should be 0.0."""
        y_true, y_pred = worst_predictions
        metrics = utils_metrics.compute_metrics(y_true, y_pred)
        
        assert metrics['accuracy'] == 0.0
    
    def test_compute_metrics_simple(self, simple_predictions):
        """Test metric computation on known simple array."""
        y_true, y_pred = simple_predictions
        metrics = utils_metrics.compute_metrics(y_true, y_pred)
        
        # Manually computed expected values
        # y_true = [1, 0, 1, 0, 1, 1, 0, 0, 1, 0]
        # y_pred = [1, 1, 0, 0, 1, 1, 0, 0, 1, 0]
        # Correct: positions 0, 1, 3, 4, 5, 7, 8, 9 → 8/10 = 0.8
        assert metrics['accuracy'] == 0.8
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
    
    def test_compute_metrics_imbalanced(self, imbalanced_predictions):
        """Test metrics with imbalanced classes."""
        y_true, y_pred = imbalanced_predictions
        metrics = utils_metrics.compute_metrics(y_true, y_pred)
        
        assert 0.0 <= metrics['accuracy'] <= 1.0
        assert 0.0 <= metrics['precision'] <= 1.0
        assert 0.0 <= metrics['recall'] <= 1.0
        assert 0.0 <= metrics['f1_score'] <= 1.0
    
    def test_compute_metrics_returns_dict(self, simple_predictions):
        """Check that all expected keys are in output."""
        y_true, y_pred = simple_predictions
        metrics = utils_metrics.compute_metrics(y_true, y_pred)
        
        expected_keys = {
            'accuracy', 'precision', 'recall', 'f1_score',
            'precision_macro', 'recall_macro', 'f1_macro',
            'precision_micro', 'recall_micro', 'f1_micro',
            'precision_weighted', 'recall_weighted', 'f1_weighted',
            'precision_class0', 'recall_class0', 'f1_class0',
            'precision_class1', 'recall_class1', 'f1_class1',
            'support_class0', 'support_class1'
        }
        
        assert expected_keys.issubset(set(metrics.keys()))


# ============================================================================
# TESTS: MANUAL METRIC CALCULATIONS
# ============================================================================

class TestManualMetrics:
    """Tests for manual metric calculations (cross-validation)."""
    
    def test_manual_accuracy_perfect(self, perfect_predictions):
        """Manual accuracy on perfect predictions."""
        y_true, y_pred = perfect_predictions
        acc = utils_metrics.manual_accuracy(y_true, y_pred)
        assert acc == 1.0
    
    def test_manual_accuracy_worst(self, worst_predictions):
        """Manual accuracy on worst predictions."""
        y_true, y_pred = worst_predictions
        acc = utils_metrics.manual_accuracy(y_true, y_pred)
        assert acc == 0.0
    
    def test_manual_accuracy_simple(self, simple_predictions):
        """Manual accuracy on simple array."""
        y_true, y_pred = simple_predictions
        acc = utils_metrics.manual_accuracy(y_true, y_pred)
        # 8 correct out of 10
        assert acc == 0.8
    
    def test_manual_confusion_matrix(self, simple_predictions):
        """Test manual confusion matrix."""
        y_true, y_pred = simple_predictions
        cm = utils_metrics.manual_confusion_matrix(y_true, y_pred, num_classes=2)
        
        # Check shape
        assert cm.shape == (2, 2)
        
        # Check sum equals total samples
        assert cm.sum() == len(y_true)
        
        # All values non-negative
        assert np.all(cm >= 0)
    
    def test_manual_precision_recall_f1(self, simple_predictions):
        """Test manual precision, recall, F1."""
        y_true, y_pred = simple_predictions
        
        prec = utils_metrics.manual_precision(y_true, y_pred, class_idx=1)
        rec = utils_metrics.manual_recall(y_true, y_pred, class_idx=1)
        f1 = utils_metrics.manual_f1_score(y_true, y_pred, class_idx=1)
        
        # All should be in [0, 1]
        assert 0.0 <= prec <= 1.0
        assert 0.0 <= rec <= 1.0
        assert 0.0 <= f1 <= 1.0
        
        # F1 is harmonic mean of precision and recall
        if prec + rec > 0:
            expected_f1 = 2 * (prec * rec) / (prec + rec)
            assert abs(f1 - expected_f1) < 1e-10


# ============================================================================
# TESTS: CROSS-VALIDATION
# ============================================================================

class TestCrossValidation:
    """Tests for cross-checking metrics."""
    
    def test_cross_check_metrics_simple(self, simple_predictions):
        """Cross-check metrics don't raise errors."""
        y_true, y_pred = simple_predictions
        # Should not raise
        utils_metrics.cross_check_metrics(y_true, y_pred)
    
    def test_cross_check_metrics_perfect(self, perfect_predictions):
        """Cross-check on perfect predictions."""
        y_true, y_pred = perfect_predictions
        utils_metrics.cross_check_metrics(y_true, y_pred)
    
    def test_cross_check_metrics_imbalanced(self, imbalanced_predictions):
        """Cross-check on imbalanced data."""
        y_true, y_pred = imbalanced_predictions
        utils_metrics.cross_check_metrics(y_true, y_pred)
    
    def test_verify_confusion_matrix(self, simple_predictions):
        """Test confusion matrix verification."""
        y_true, y_pred = simple_predictions
        cm = utils_metrics.manual_confusion_matrix(y_true, y_pred, num_classes=2)
        
        # Should not raise
        utils_metrics.verify_confusion_matrix(cm, y_true, y_pred)


# ============================================================================
# TESTS: INPUT VALIDATION
# ============================================================================

class TestValidation:
    """Tests for input validation."""
    
    def test_validate_predictions_valid(self, simple_predictions):
        """Valid predictions should pass."""
        y_true, y_pred = simple_predictions
        # Should not raise
        utils_metrics.validate_predictions(y_true, y_pred)
    
    def test_validate_predictions_shape_mismatch(self):
        """Shape mismatch should raise."""
        y_true = np.array([1, 0, 1])
        y_pred = np.array([1, 0])
        
        with pytest.raises(ValueError, match="Shape mismatch"):
            utils_metrics.validate_predictions(y_true, y_pred)
    
    def test_validate_predictions_invalid_label(self):
        """Invalid label values should raise."""
        y_true = np.array([1, 0, 2])  # 2 is invalid
        y_pred = np.array([1, 0, 1])
        
        with pytest.raises(ValueError, match="invalid label"):
            utils_metrics.validate_predictions(y_true, y_pred)
    
    def test_validate_predictions_nan(self):
        """NaN values should raise."""
        y_true = np.array([1, 0, np.nan])
        y_pred = np.array([1, 0, 1])
        
        with pytest.raises(ValueError, match="NaN"):
            utils_metrics.validate_predictions(y_true, y_pred)
    
    def test_compute_metrics_empty_input(self):
        """Empty input should raise."""
        with pytest.raises(ValueError, match="Empty predictions"):
            utils_metrics.compute_metrics(np.array([]), np.array([]))
    
    def test_compute_metrics_shape_mismatch(self):
        """Shape mismatch should raise."""
        with pytest.raises(ValueError, match="same length"):
            utils_metrics.compute_metrics(
                np.array([1, 0, 1]),
                np.array([1, 0])
            )


# ============================================================================
# TESTS: EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_single_sample(self):
        """Single sample prediction."""
        y_true = np.array([1])
        y_pred = np.array([1])
        metrics = utils_metrics.compute_metrics(y_true, y_pred)
        
        assert metrics['accuracy'] == 1.0
    
    def test_all_same_class_true(self):
        """All true labels are same class."""
        y_true = np.array([1, 1, 1, 1, 1])
        y_pred = np.array([1, 1, 0, 1, 1])
        metrics = utils_metrics.compute_metrics(y_true, y_pred)
        
        # Should still compute metrics without error
        assert 'accuracy' in metrics
        assert metrics['accuracy'] == 0.8
    
    def test_all_same_class_pred(self):
        """All predicted labels are same class."""
        y_true = np.array([1, 1, 0, 0, 1])
        y_pred = np.array([1, 1, 1, 1, 1])
        metrics = utils_metrics.compute_metrics(y_true, y_pred)
        
        assert 'accuracy' in metrics
        assert metrics['accuracy'] == 0.6
    
    def test_large_batch(self):
        """Large batch of samples."""
        y_true = np.random.randint(0, 2, size=10000)
        y_pred = np.random.randint(0, 2, size=10000)
        
        metrics = utils_metrics.compute_metrics(y_true, y_pred)
        
        assert 'accuracy' in metrics
        assert 0.0 <= metrics['accuracy'] <= 1.0


# ============================================================================
# TESTS: CLASS WEIGHTS AND HELPER FUNCTIONS
# ============================================================================

class TestHelperFunctions:
    """Tests for helper functions."""
    
    def test_get_class_weights_balanced(self):
        """Balanced classes should have equal weights."""
        y_true = np.array([0] * 50 + [1] * 50)
        w0, w1 = utils_metrics.get_class_weights(y_true)
        
        # Weights should be equal for balanced classes
        assert abs(w0 - w1) < 1e-6
    
    def test_get_class_weights_imbalanced(self):
        """Imbalanced classes should have different weights."""
        y_true = np.array([1] * 90 + [0] * 10)
        w0, w1 = utils_metrics.get_class_weights(y_true)
        
        # Class 0 (minority) should have higher weight
        assert w0 > w1
    
    def test_compute_roc_auc(self):
        """Test ROC-AUC computation."""
        y_true = np.array([1, 0, 1, 0, 1, 1, 0, 0])
        y_probs = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.6, 0.3, 0.2])
        
        auc = utils_metrics.compute_roc_auc(y_true, y_probs)
        
        # AUC should be in [0, 1] or NaN
        assert (0.0 <= auc <= 1.0) or np.isnan(auc)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests for full evaluation pipeline components."""
    
    def test_metrics_consistency_across_calls(self, simple_predictions):
        """Same input should produce same metrics."""
        y_true, y_pred = simple_predictions
        
        metrics1 = utils_metrics.compute_metrics(y_true, y_pred)
        metrics2 = utils_metrics.compute_metrics(y_true, y_pred)
        
        # Dictionaries should be identical
        for key in metrics1:
            assert abs(metrics1[key] - metrics2[key]) < 1e-10
    
    def test_metrics_serializable(self, simple_predictions):
        """Metrics should be JSON serializable."""
        y_true, y_pred = simple_predictions
        metrics = utils_metrics.compute_metrics(y_true, y_pred)
        
        # Should not raise
        json_str = json.dumps(metrics)
        loaded = json.loads(json_str)
        
        assert loaded['accuracy'] == metrics['accuracy']
    
    def test_full_evaluation_workflow(self, simple_predictions):
        """Test full workflow: compute, cross-check, validate."""
        y_true, y_pred = simple_predictions
        
        # Validate inputs
        utils_metrics.validate_predictions(y_true, y_pred)
        
        # Compute metrics
        metrics = utils_metrics.compute_metrics(y_true, y_pred)
        
        # Cross-check
        utils_metrics.cross_check_metrics(y_true, y_pred)
        
        # Verify confusion matrix
        cm = utils_metrics.manual_confusion_matrix(y_true, y_pred)
        utils_metrics.verify_confusion_matrix(cm, y_true, y_pred)
        
        # All should complete without error
        assert metrics['accuracy'] > 0


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestPerformance:
    """Performance tests to ensure efficiency."""
    
    def test_large_dataset_performance(self):
        """Metric computation should be fast on large datasets."""
        import time
        
        y_true = np.random.randint(0, 2, size=100000)
        y_pred = np.random.randint(0, 2, size=100000)
        
        start = time.time()
        metrics = utils_metrics.compute_metrics(y_true, y_pred)
        elapsed = time.time() - start
        
        # Should complete in < 1 second
        assert elapsed < 1.0
        assert 'accuracy' in metrics


# ============================================================================
# DOCTESTS & EXAMPLES
# ============================================================================

class TestExamples:
    """Example usage patterns."""
    
    def test_binary_classification_workflow(self):
        """Example of binary classification workflow."""
        # Create sample data
        y_true = np.array([1, 0, 1, 0, 1, 1, 0, 0, 1, 0], dtype=np.int32)
        y_pred = np.array([1, 1, 0, 0, 1, 1, 0, 0, 1, 0], dtype=np.int32)
        
        # Compute metrics
        metrics = utils_metrics.compute_metrics(y_true, y_pred)
        
        # Check results are reasonable
        assert 'accuracy' in metrics
        assert isinstance(metrics['accuracy'], (float, int))
        assert 0.0 <= metrics['accuracy'] <= 1.0


if __name__ == '__main__':
    # Run tests: pytest test_metrics.py -v
    pytest.main([__file__, '-v'])
