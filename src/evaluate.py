#!/usr/bin/env python3
"""
EVALUATE - Production-grade Ship Detection Model Evaluator
===========================================================================

ASSUMPTIONS & CONFIGURABLE PARAMETERS:
1. Binary classification: ship (label 1) vs. no-ship (label 0)
2. Input images are resized to --img-size (default 224x224)
3. Test labels provided in CSV format: columns [filename, label] or [filename, ship_present]
4. Model outputs: 
   - PyTorch: tensor shape [batch, 1] for binary (sigmoid applied) or [batch, 2] (softmax+argmax)
   - TensorFlow: same shapes
5. Generated/predicted images are optional; if provided, must be in --generated-dir with same filename format
6. GPU available but not required; falls back to CPU gracefully

MAIN METRICS COMPUTED:
- Accuracy, Precision, Recall, F1-score (binary and macro/micro if multi-class)
- Confusion matrix (plotted + numeric)
- Per-class metrics and overall statistics
- Cross-check: each metric is validated against multiple implementations (sklearn + manual)

OUTPUTS SAVED:
- metrics.json         → machine-readable metrics + metadata (timestamps, git hash)
- metrics.csv         → human-readable metrics
- confusion_matrix.png → annotated confusion matrix visualization
- classification_report.txt → full scikit-learn classification report

USAGE EXAMPLE:
    python evaluate.py \\
        --model-path runs/detect/yolov8n_epochs50_batch16/weights/best.pt \\
        --test-dir datasets/images/val \\
        --labels-csv datasets/labels/val_labels.csv \\
        --output-dir results/ \\
        --framework pytorch \\
        --batch-size 32 \\
        --device cuda \\
        --seed 42 \\
        --img-size 224 \\
        --threshold 0.5

For dry-run (test on 10 images without saving):
    python evaluate.py ... --dry-run

Unit tests:
    pytest test_metrics.py -v

===========================================================================
"""

import argparse
import csv
import json
import logging
import os
import random
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

__all__ = [
    'evaluate_model',
    'plot_confusion_matrix',
    'set_seed',
    'get_git_hash',
    'select_device',
    'load_labels_from_csv',
    'load_image',
    'get_available_test_images',
]

# Try importing machine learning frameworks
try:
    import torch
    import torchvision.transforms as transforms
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False
    tf = None

# Metrics and visualization
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_fscore_support,
    classification_report,
)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

# Import custom metrics utilities
import utils_metrics


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across all libraries."""
    np.random.seed(seed)
    random.seed(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if HAS_TF:
        tf.random.set_seed(seed)
        tf.config.run_functions_eagerly(True)
    logger.info(f"Random seeds set to {seed}")


def get_git_hash() -> str:
    """Retrieve current git commit hash if available."""
    try:
        import subprocess
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=os.path.dirname(__file__),
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()[:8]
    except Exception as e:
        logger.debug(f"Could not retrieve git hash: {e}")
    return "unknown"


def select_device(device_str: str) -> Union[str, torch.device]:
    """Select and validate device (cpu or cuda)."""
    if device_str.lower() == 'auto':
        if HAS_TORCH and torch.cuda.is_available():
            device_str = 'cuda'
        else:
            device_str = 'cpu'
    
    if device_str.lower() == 'cuda':
        if not HAS_TORCH or not torch.cuda.is_available():
            logger.warning("CUDA requested but not available; falling back to CPU")
            device_str = 'cpu'
    
    logger.info(f"Using device: {device_str}")
    
    if HAS_TORCH:
        return torch.device(device_str)
    return device_str


def load_labels_from_csv(csv_path: str) -> Dict[str, int]:
    """
    Load ground-truth labels from CSV.
    Expected columns: 'filename' and one of ['label', 'ship_present', 'is_ship'].
    Returns: dict mapping filename -> binary label (0 or 1)
    """
    labels = {}
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Labels CSV not found: {csv_path}")
    
    label_column = None
    try:
        df = pd.read_csv(csv_path)
        
        # Find label column
        for col in ['label', 'ship_present', 'is_ship', 'target']:
            if col in df.columns:
                label_column = col
                break
        
        if label_column is None:
            raise ValueError(
                f"No recognized label column found. Expected one of: "
                f"'label', 'ship_present', 'is_ship', 'target'. "
                f"Found columns: {list(df.columns)}"
            )
        
        # Parse labels
        for _, row in df.iterrows():
            filename = str(row['filename']).strip()
            label = int(row[label_column])
            
            # Validate: label should be 0 or 1
            if label not in (0, 1):
                logger.warning(
                    f"Unexpected label value {label} for {filename}; "
                    f"expected 0 or 1. Skipping this sample."
                )
                continue
            
            labels[filename] = label
        
        logger.info(f"Loaded {len(labels)} labels from {csv_path}")
        return labels
    
    except Exception as e:
        logger.error(f"Error loading labels CSV: {e}")
        raise


def load_image(image_path: str, img_size: int = 224) -> np.ndarray:
    """
    Load and preprocess image for model input.
    Returns: np array of shape (img_size, img_size, 3) with values in [0, 1].
    """
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((img_size, img_size), Image.Resampling.BILINEAR)
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        if img_array.shape != (img_size, img_size, 3):
            raise ValueError(f"Unexpected image shape: {img_array.shape}")
        
        return img_array
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        raise


def get_available_test_images(
    test_dir: str,
    labels: Dict[str, int],
    limit: Optional[int] = None
) -> List[Tuple[str, int]]:
    """
    Collect all test images from directory that have corresponding labels.
    Returns: list of (image_path, label) tuples
    """
    test_images = []
    
    if not os.path.isdir(test_dir):
        raise NotADirectoryError(f"Test directory not found: {test_dir}")
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    for filename in os.listdir(test_dir):
        if not any(filename.lower().endswith(ext) for ext in image_extensions):
            continue
        
        # Try to find label (strip extension and look for base name)
        base_name = os.path.splitext(filename)[0]
        
        label = None
        # Try exact match first
        if filename in labels:
            label = labels[filename]
        # Try base name match
        elif base_name in labels:
            label = labels[base_name]
        # Try with various extensions
        else:
            for ext in image_extensions:
                test_key = base_name + ext
                if test_key in labels:
                    label = labels[test_key]
                    break
        
        if label is None:
            logger.debug(f"No label found for image: {filename}")
            continue
        
        image_path = os.path.join(test_dir, filename)
        if os.path.isfile(image_path):
            test_images.append((image_path, label))
    
    if not test_images:
        raise ValueError(
            f"No valid images with labels found in {test_dir}. "
            f"Check that --test-dir points to image folder and "
            f"--labels-csv contains matching filenames."
        )
    
    if limit is not None and len(test_images) > limit:
        test_images = random.sample(test_images, limit)
    
    logger.info(f"Found {len(test_images)} test images with labels")
    return test_images


# ============================================================================
# MODEL LOADING & INFERENCE (PyTorch)
# ============================================================================

def load_pytorch_model(
    model_path: str,
    device: torch.device
) -> torch.nn.Module:
    """Load PyTorch model for inference."""
    if not HAS_TORCH:
        raise ImportError("PyTorch not installed. Install via: pip install torch")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        # Try loading as .pt file (YOLO format)
        model = torch.load(model_path, map_location=device)
        logger.info(f"Loaded PyTorch model from {model_path}")
    except Exception as e:
        logger.error(f"Could not load model as .pt file: {e}")
        try:
            # Fallback: try as state dict
            model = torch.nn.Module()  # Placeholder
            model.load_state_dict(torch.load(model_path, map_location=device))
            logger.info(f"Loaded PyTorch model state dict from {model_path}")
        except Exception as e2:
            raise RuntimeError(f"Failed to load PyTorch model: {e2}")
    
    model.to(device)
    model.eval()
    return model


def pytorch_inference(
    model: torch.nn.Module,
    images: np.ndarray,
    device: torch.device,
    num_classes: int = 2
) -> np.ndarray:
    """
    Run inference on batch of images using PyTorch model.
    Args:
        model: PyTorch model
        images: np array of shape (batch_size, H, W, 3) with values in [0, 1]
        device: torch device
        num_classes: number of output classes (default 2 for binary)
    
    Returns: np array of shape (batch_size, num_classes) with class probabilities
    """
    with torch.no_grad():
        # Convert to tensor and move to device
        # Rearrange from (B, H, W, 3) to (B, 3, H, W) for PyTorch
        images_tensor = torch.from_numpy(images).permute(0, 3, 1, 2).to(device)
        
        # Forward pass
        outputs = model(images_tensor)
        
        # Handle different output formats
        if isinstance(outputs, dict):
            # YOLO-style output
            if 'box_outputs' in outputs:
                logits = outputs['box_outputs']
            elif 'logits' in outputs:
                logits = outputs['logits']
            else:
                raise ValueError(f"Unexpected model output dict keys: {outputs.keys()}")
        else:
            logits = outputs
        
        # If output is single-channel (binary sigmoid output)
        if logits.shape[1] == 1:
            # Apply sigmoid for binary classification
            probs = torch.sigmoid(logits).squeeze(-1)
            # Create 2-class probability tensor: [prob_ship, prob_no_ship]
            probs = torch.stack([probs, 1.0 - probs], dim=1)
        else:
            # Assume logits shape is (batch, num_classes)
            probs = torch.softmax(logits, dim=1)
        
        return probs.cpu().numpy()


# ============================================================================
# MODEL LOADING & INFERENCE (TensorFlow - stub)
# ============================================================================

def load_tf_model(model_path: str) -> "tf.keras.Model":
    """Load TensorFlow/Keras model for inference."""
    if not HAS_TF:
        raise ImportError("TensorFlow not installed. Install via: pip install tensorflow")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        model = tf.keras.models.load_model(model_path)
        logger.info(f"Loaded TensorFlow model from {model_path}")
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load TensorFlow model: {e}")


def tf_inference(
    model: "tf.keras.Model",
    images: np.ndarray,
    batch_size: int = 32,
    num_classes: int = 2
) -> np.ndarray:
    """
    Run inference on batch of images using TensorFlow model.
    Similar interface to pytorch_inference for consistency.
    """
    # Images assumed to be (B, H, W, 3) with values in [0, 1] - good for Keras
    try:
        logits = model.predict(images, batch_size=batch_size, verbose=0)
        
        # If output is single-channel (binary)
        if logits.shape[1] == 1:
            probs = 1.0 / (1.0 + np.exp(-logits.squeeze(-1)))
            probs = np.stack([probs, 1.0 - probs], axis=1)
        else:
            # Apply softmax
            probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        
        return probs
    except Exception as e:
        logger.error(f"TensorFlow inference failed: {e}")
        raise


# ============================================================================
# MAIN EVALUATION PIPELINE
# ============================================================================

def evaluate_model(
    model_path: str,
    test_dir: str,
    labels_csv: str,
    output_dir: str,
    framework: str = 'pytorch',
    batch_size: int = 32,
    device_str: str = 'auto',
    seed: int = 42,
    img_size: int = 224,
    threshold: float = 0.5,
    dry_run: bool = False,
    generated_dir: Optional[str] = None,
) -> Dict[str, Union[str, float, dict]]:
    """
    Main evaluation pipeline.
    
    Returns:
        Dictionary containing all metrics and metadata.
    """
    logger.info("=" * 70)
    logger.info("STARTING MODEL EVALUATION")
    logger.info("=" * 70)
    
    # Set seed for reproducibility
    set_seed(seed)
    
    # Validate inputs
    if framework not in ('pytorch', 'tensorflow', 'tf'):
        raise ValueError(f"Unknown framework: {framework}")
    
    if not os.path.isdir(test_dir):
        raise NotADirectoryError(f"Test directory not found: {test_dir}")
    
    if not os.path.isfile(labels_csv):
        raise FileNotFoundError(f"Labels CSV not found: {labels_csv}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load labels
    logger.info("Loading labels...")
    labels = load_labels_from_csv(labels_csv)
    
    # Get test images
    logger.info("Collecting test images...")
    test_images = get_available_test_images(test_dir, labels)
    
    # For dry-run, limit to 10 images
    if dry_run:
        test_images = test_images[:10]
        logger.info(f"DRY-RUN mode: using only {len(test_images)} images")
    
    logger.info(f"Will evaluate on {len(test_images)} test images")
    
    # Load model based on framework
    logger.info(f"Loading {framework} model...")
    device = select_device(device_str) if framework == 'pytorch' else 'cpu'
    
    if framework == 'pytorch':
        model = load_pytorch_model(model_path, device)
    else:
        model = load_tf_model(model_path)
    
    # Prepare batch inference
    logger.info("Running inference on test set...")
    y_true_test = []
    y_pred_probs_test = []
    test_image_names = []
    
    num_batches = (len(test_images) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Inference batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(test_images))
        batch_images_data = test_images[start_idx:end_idx]
        
        # Load images
        batch_images = []
        batch_labels = []
        batch_names = []
        
        for image_path, label in batch_images_data:
            try:
                img = load_image(image_path, img_size=img_size)
                batch_images.append(img)
                batch_labels.append(label)
                batch_names.append(os.path.basename(image_path))
            except Exception as e:
                logger.warning(f"Skipping corrupted image {image_path}: {e}")
                continue
        
        if not batch_images:
            continue
        
        batch_images = np.array(batch_images)
        
        # Run inference
        if framework == 'pytorch':
            probs = pytorch_inference(model, batch_images, device, num_classes=2)
        else:
            probs = tf_inference(model, batch_images, batch_size=len(batch_images), num_classes=2)
        
        y_true_test.extend(batch_labels)
        y_pred_probs_test.extend(probs)
        test_image_names.extend(batch_names)
    
    if not y_true_test:
        raise RuntimeError("No valid predictions generated. Check image paths and model.")
    
    y_true_test = np.array(y_true_test)
    y_pred_probs_test = np.array(y_pred_probs_test)
    
    logger.info(f"Inference complete on {len(y_true_test)} images")
    
    # Convert probabilities to class predictions using threshold
    # y_pred_probs_test shape: (N, 2) with columns [prob_ship, prob_no_ship]
    y_pred_test = (y_pred_probs_test[:, 0] >= threshold).astype(int)
    
    logger.info(f"Applied threshold {threshold} for binary predictions")
    
    # Compute metrics
    logger.info("Computing metrics...")
    metrics = {
        'test_set': utils_metrics.compute_metrics(y_true_test, y_pred_test),
        'metadata': {
            'timestamp': datetime.utcnow().isoformat(),
            'git_hash': get_git_hash(),
            'framework': framework,
            'model_path': model_path,
            'num_test_samples': len(y_true_test),
            'img_size': img_size,
            'batch_size': batch_size,
            'threshold': threshold,
            'seed': seed,
            'device': str(device),
        }
    }
    
    # Cross-check metrics
    logger.info("Cross-checking metrics...")
    utils_metrics.cross_check_metrics(y_true_test, y_pred_test)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true_test, y_pred_test)
    metrics['confusion_matrix'] = cm.tolist()
    
    # Check for generated/predicted images if provided
    if generated_dir and os.path.isdir(generated_dir):
        logger.info(f"Evaluating generated images from {generated_dir}...")
        y_true_gen = []
        y_pred_gen = []
        
        gen_images = []
        for filename in os.listdir(generated_dir):
            if not any(filename.lower().endswith(ext) for ext in {'.jpg', '.jpeg', '.png', '.bmp'}):
                continue
            
            # Try to find true label
            label = None
            if filename in labels:
                label = labels[filename]
            else:
                base_name = os.path.splitext(filename)[0]
                if base_name in labels:
                    label = labels[base_name]
            
            if label is None:
                continue
            
            gen_images.append((os.path.join(generated_dir, filename), label))
        
        if gen_images:
            logger.info(f"Found {len(gen_images)} generated images with labels")
            
            for gen_image_path, label in gen_images:
                try:
                    img = load_image(gen_image_path, img_size=img_size)
                    img_batch = np.expand_dims(img, axis=0)
                    
                    if framework == 'pytorch':
                        probs = pytorch_inference(model, img_batch, device, num_classes=2)
                    else:
                        probs = tf_inference(model, img_batch, batch_size=1, num_classes=2)
                    
                    pred = int((probs[0, 0] >= threshold))
                    y_true_gen.append(label)
                    y_pred_gen.append(pred)
                except Exception as e:
                    logger.warning(f"Error processing generated image {gen_image_path}: {e}")
            
            if y_true_gen:
                y_true_gen = np.array(y_true_gen)
                y_pred_gen = np.array(y_pred_gen)
                metrics['generated_set'] = utils_metrics.compute_metrics(y_true_gen, y_pred_gen)
                logger.info(f"Generated set metrics computed on {len(y_true_gen)} images")
    
    # Visualization
    logger.info("Generating visualizations...")
    plot_confusion_matrix(
        cm,
        class_names=['No Ship', 'Ship'],
        output_path=os.path.join(output_dir, 'confusion_matrix.png')
    )
    
    # Generate classification report
    report = classification_report(y_true_test, y_pred_test, target_names=['No Ship', 'Ship'])
    
    # Save outputs
    if not dry_run:
        logger.info("Saving outputs...")
        
        # Save metrics as JSON
        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save metrics as CSV
        metrics_df = pd.DataFrame([metrics['test_set']])
        metrics_df.to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)
        
        # Save classification report
        with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
            f.write(report)
        
        logger.info(f"Outputs saved to {output_dir}")
        logger.info(f"  - metrics.json")
        logger.info(f"  - metrics.csv")
        logger.info(f"  - confusion_matrix.png")
        logger.info(f"  - classification_report.txt")
    else:
        logger.info("DRY-RUN: outputs not saved")
    
    # Print summary
    logger.info("=" * 70)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Accuracy:  {metrics['test_set']['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['test_set']['precision']:.4f}")
    logger.info(f"Recall:    {metrics['test_set']['recall']:.4f}")
    logger.info(f"F1-score:  {metrics['test_set']['f1_score']:.4f}")
    logger.info("=" * 70)
    
    return metrics


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    output_path: str = 'confusion_matrix.png'
) -> None:
    """
    Plot and save confusion matrix with annotations.
    
    Args:
        cm: Confusion matrix from sklearn
        class_names: List of class names (default: ['Class 0', 'Class 1'])
        output_path: Path to save PNG
    """
    if class_names is None:
        class_names = [f'Class {i}' for i in range(cm.shape[0])]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Normalize for percentage display
    cm_percent = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100
    
    # Plot heatmap
    sns.heatmap(
        cm,
        annot=False,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'},
        ax=ax,
        vmin=0
    )
    
    # Add annotations: count and percentage
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            pct = cm_percent[i, j]
            text = ax.text(
                j + 0.5, i + 0.7,
                f'{count}\n({pct:.1f}%)',
                ha='center', va='center',
                color='white' if count > cm.max() / 2 else 'black',
                fontsize=10, weight='bold'
            )
    
    ax.set_ylabel('True Label', fontsize=11, weight='bold')
    ax.set_xlabel('Predicted Label', fontsize=11, weight='bold')
    ax.set_title('Confusion Matrix - Ship Detection', fontsize=12, weight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Confusion matrix saved to {output_path}")
    plt.close()


# ============================================================================
# MAIN CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ship detection model on test set",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:

  Basic evaluation:
    python evaluate.py --model-path best.pt --test-dir data/test \\
      --labels-csv data/test_labels.csv --output-dir results/

  With TensorFlow model:
    python evaluate.py --model-path model.h5 --framework tensorflow \\
      --test-dir data/test --labels-csv data/labels.csv

  Dry-run (10 images, no file output):
    python evaluate.py ... --dry-run

  Custom threshold and image size:
    python evaluate.py ... --threshold 0.6 --img-size 512

  With GPU and specific seed:
    python evaluate.py ... --device cuda --seed 123

LABEL CSV FORMAT:
  filename,label
  image001.jpg,1
  image002.jpg,0
  image003.jpg,1
        """
    )
    
    parser.add_argument('--model-path', required=True,
                        help='Path to trained model (.pt for PyTorch, .h5 for Keras)')
    parser.add_argument('--test-dir', required=True,
                        help='Directory containing test images')
    parser.add_argument('--labels-csv', required=True,
                        help='CSV with columns: filename, label (0 or 1)')
    parser.add_argument('--output-dir', default='results',
                        help='Directory to save metrics and plots (default: results)')
    parser.add_argument('--framework', choices=['pytorch', 'tensorflow', 'tf'],
                        default='pytorch',
                        help='ML framework (default: pytorch)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for inference (default: 32)')
    parser.add_argument('--device', default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help='Device: auto/cpu/cuda (default: auto)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--img-size', type=int, default=224,
                        help='Input image size (default: 224)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold for binary predictions (default: 0.5)')
    parser.add_argument('--generated-dir', default=None,
                        help='Optional directory with generated/predicted images')
    parser.add_argument('--dry-run', action='store_true',
                        help='Dry-run: test on 10 images without saving files')
    
    args = parser.parse_args()
    
    try:
        metrics = evaluate_model(
            model_path=args.model_path,
            test_dir=args.test_dir,
            labels_csv=args.labels_csv,
            output_dir=args.output_dir,
            framework=args.framework,
            batch_size=args.batch_size,
            device_str=args.device,
            seed=args.seed,
            img_size=args.img_size,
            threshold=args.threshold,
            dry_run=args.dry_run,
            generated_dir=args.generated_dir,
        )
        
        logger.info("✓ Evaluation completed successfully")
        return 0
    
    except Exception as e:
        logger.error(f"✗ Evaluation failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
