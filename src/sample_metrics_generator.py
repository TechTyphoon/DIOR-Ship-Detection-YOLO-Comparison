#!/usr/bin/env python3
"""
Sample metrics data generator for demonstration.
Creates realistic sample metrics data.
"""

import json
from pathlib import Path


def generate_sample_metrics_data():
    """Generate sample metrics data for visualization testing."""
    
    output_dir = Path('runs/detect/metrics')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample training history
    epochs = list(range(1, 51))  # 50 epochs
    metrics_history = {
        'epoch': epochs,
        'f1': [0.45 + i * 0.008 + (i % 5) * 0.01 for i in epochs],
        'accuracy': [0.50 + i * 0.007 + (i % 5) * 0.008 for i in epochs],
        'precision': [0.55 + i * 0.006 + (i % 5) * 0.007 for i in epochs],
        'recall': [0.40 + i * 0.009 + (i % 5) * 0.009 for i in epochs],
        'train_loss': [2.5 - i * 0.035 - (i % 5) * 0.02 for i in epochs],
        'val_loss': [2.6 - i * 0.032 - (i % 5) * 0.015 for i in epochs],
    }
    
    # Final metrics
    all_metrics = {
        'accuracy': 0.92,
        'precision': 0.94,
        'recall': 0.90,
        'f1': 0.92,
        'roc_auc': 0.96,
    }
    
    # Per-class metrics
    per_class_metrics = {
        'Ship': {
            'precision': 0.94,
            'recall': 0.90,
            'f1': 0.92,
        },
        'Background': {
            'precision': 0.91,
            'recall': 0.93,
            'f1': 0.92,
        }
    }
    
    # Save to JSON
    data = {
        'final_metrics': all_metrics,
        'metrics_history': metrics_history,
        'per_class_metrics': per_class_metrics,
    }
    
    json_path = output_dir / 'sample_metrics.json'
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Sample metrics data saved to {json_path}")
    return data


if __name__ == '__main__':
    generate_sample_metrics_data()
