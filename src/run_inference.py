#!/usr/bin/env python3
"""
Run Ship Detection Inference on Sample Images

This script runs ship detection inference on satellite images from the dataset
and generates visualizations with detection results.

SHIPS ONLY - All other object classes are ignored.

Usage:
    python run_inference.py
"""

import os
import random
from typing import List

import cv2
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

import constants
import helpers


# ============================================================================
# IMAGE MANAGEMENT
# ============================================================================

def collect_all_images() -> List[tuple]:
    """
    Collect all available images from test and train directories.
    
    Returns:
        List of tuples: (filename, directory_path)
    """
    return helpers.find_images_in_directories(
        constants.TEST_IMAGES_DIR,
        constants.TRAINVAL_IMAGES_DIR
    )


# ============================================================================
# DETECTION & RESULTS
# ============================================================================

def run_detection_batch(
    model,
    images_data: List[tuple],
    is_trained_model: bool,
    max_images: int = 6
) -> List[dict]:
    """
    Run detection on batch of images.
    
    Args:
        model: Loaded YOLO model.
        images_data: List of (filename, directory) tuples.
        is_trained_model: Whether using trained ship detection model.
        max_images: Maximum number of images to process.
    
    Returns:
        List of detection result dictionaries.
    """
    results = []
    images_to_process = min(max_images, len(images_data))
    selected = random.sample(images_data, images_to_process)
    
    print(f"\n[3/4] Running SHIP detection inference...")
    print("-" * 70)
    
    for i, (filename, directory) in enumerate(selected, 1):
        img_path = os.path.join(directory, filename)
        print(f"\n[{i}/{images_to_process}] {filename}")
        
        # Run prediction
        annotated, result = helpers.run_detection_on_image(
            model,
            img_path,
            image_size=constants.IMAGE_SIZE,
            confidence=constants.DEFAULT_CONFIDENCE_THRESHOLD,
            verbose=False
        )
        
        # Count ships detected
        ships_detected = helpers.count_ships_in_detections(result, is_trained_model)
        print(f"   ✓ Ships detected: {ships_detected}")
        
        # Save annotated image
        output_path = os.path.join(
            constants.INFERENCE_OUTPUT_DIR,
            f"ships_detected_{i}_{filename}"
        )
        cv2.imwrite(output_path, annotated)
        
        results.append({
            "filename": filename,
            "path": img_path,
            "ships": ships_detected,
            "output": output_path,
        })
    
    return results


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_results_grid(results: List[dict]) -> str:
    """
    Create grid visualization of detection results.
    
    Args:
        results: List of detection result dictionaries.
    
    Returns:
        Path to saved visualization.
    """
    num_results = len(results)
    cols = 3
    rows = (num_results + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    
    # Flatten axes for easier indexing
    if num_results == 1:
        axes_flat = [axes]
    elif rows == 1:
        axes_flat = list(axes) if isinstance(axes, tuple) else [axes]
    else:
        axes_flat = axes.flatten()
    
    for idx, result_info in enumerate(results):
        ax = axes_flat[idx]
        
        img = cv2.imread(result_info["output"])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
        ax.axis("off")
        
        title = f"{result_info['filename']}\nShips: {result_info['ships']}"
        ax.set_title(title, fontsize=10, pad=5, fontweight="bold")
    
    # Hide empty subplots
    for idx in range(num_results, len(axes_flat)):
        axes_flat[idx].axis("off")
    
    plt.suptitle("Ship Detection Results", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    
    summary_path = os.path.join(
        constants.INFERENCE_OUTPUT_DIR,
        "ships_detection_results.png"
    )
    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    return summary_path


# ============================================================================
# REPORTING
# ============================================================================

def print_results_summary(results: List[dict]) -> None:
    """Print summary of inference results."""
    num_images = len(results)
    total_ships = sum(r["ships"] for r in results)
    avg_ships = total_ships / num_images if num_images > 0 else 0.0
    
    print("\n" + "=" * 70)
    print("📊 SHIP DETECTION RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nImages processed: {num_images}")
    print(f"Total ships detected: {total_ships}")
    print(f"Average ships per image: {avg_ships:.2f}")
    
    print("\n📁 Output files:")
    print(f"   • Individual results: {constants.INFERENCE_OUTPUT_DIR}/ships_detected_*.jpg")
    print(f"   • Summary visualization: {os.path.join(constants.INFERENCE_OUTPUT_DIR, 'ships_detection_results.png')}")
    
    print("\n" + "=" * 70)
    print("✅ Ship detection completed! Check the results in:", constants.INFERENCE_OUTPUT_DIR)
    print("=" * 70)


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    """Main inference pipeline."""
    print("=" * 70)
    print("🛰️  SHIP DETECTION - Satellite Imagery (YOLOv8)")
    print("=" * 70)
    
    # Load model
    print("\n[1/4] Loading model...")
    model, is_trained_model = helpers.load_model()
    
    if is_trained_model:
        print("   ✓ Using TRAINED SHIP DETECTION model")
        print("   ✓ Single class: SHIP ONLY")
    else:
        print("   ⚠️  Using pre-trained COCO model (general detection)")
        print("   Note: For ship-specific detection, train the model first")
    
    print("   ✓ Model loaded!")
    
    # Find images
    print("\n[2/4] Searching for images in dataset...")
    all_images = collect_all_images()
    
    if not all_images:
        print("\n❌ No images found! Run: python download_dataset.py")
        return
    
    num_test = min(6, len(all_images))
    print(f"\n   Found {len(all_images)} images in dataset")
    print(f"   ✓ Selected {num_test} random images for testing")
    
    # Create output directory
    constants.ensure_directory(constants.INFERENCE_OUTPUT_DIR)
    
    # Run detection
    results = run_detection_batch(model, all_images, is_trained_model, max_images=num_test)
    
    # Create visualization
    print(f"\n[4/4] Creating visualization...")
    summary_path = create_results_grid(results)
    print(f"   ✓ Saved: {summary_path}")
    
    # Print summary
    print_results_summary(results)


if __name__ == "__main__":
    main()

