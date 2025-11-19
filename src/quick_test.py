#!/usr/bin/env python3
"""
Quick Ship Detection Test on Sample Images

This script runs ship detection inference on a small random sample of 
satellite images and generates a summary visualization.

SHIPS ONLY - All other object classes are ignored.

Usage:
    python quick_test.py
"""

import os
import random
from typing import List, Tuple

import cv2
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

import constants
import helpers


# ============================================================================
# IMAGE COLLECTION
# ============================================================================

def collect_available_images() -> List[str]:
    """
    Collect all available test and train images from dataset.
    
    Returns:
        List of all image filenames found.
    """
    images = []
    
    for directory in [constants.TEST_IMAGES_DIR, constants.TRAINVAL_IMAGES_DIR]:
        if os.path.exists(directory):
            for filename in os.listdir(directory):
                if any(filename.lower().endswith(ext) for ext in constants.IMAGE_EXTENSIONS):
                    images.append(filename)
    
    return images


def get_image_full_path(filename: str) -> str:
    """Get full path to image file."""
    return helpers.get_image_path(
        filename,
        constants.TEST_IMAGES_DIR,
        constants.TRAINVAL_IMAGES_DIR
    )


# ============================================================================
# DETECTION & RESULTS
# ============================================================================

def run_detection_batch(
    model,
    image_filenames: List[str],
    is_trained_model: bool
) -> List[dict]:
    """
    Run detection on batch of images.
    
    Args:
        model: Loaded YOLO model.
        image_filenames: List of image filenames to process.
        is_trained_model: Whether using trained ship detection model.
    
    Returns:
        List of detection result dictionaries.
    """
    results = []
    
    for i, filename in enumerate(image_filenames, 1):
        full_path = get_image_full_path(filename)
        if not full_path:
            continue
        
        print(f"\n[{i}/{len(image_filenames)}] Processing: {filename}")
        
        # Run prediction
        annotated, result = helpers.run_detection_on_image(
            model,
            full_path,
            image_size=constants.IMAGE_SIZE,
            confidence=constants.DEFAULT_CONFIDENCE_THRESHOLD,
            verbose=False
        )
        
        # Count ships detected
        ships_detected = helpers.count_ships_in_detections(result, is_trained_model)
        print(f"   ✓ Ships detected: {ships_detected}")
        
        # Save annotated image
        output_path = os.path.join(
            constants.QUICK_TEST_OUTPUT_DIR,
            f"ships_result_{i}_{filename}"
        )
        cv2.imwrite(output_path, annotated)
        print(f"   ✓ Saved to: {output_path}")
        
        results.append({
            "filename": filename,
            "ships": ships_detected,
            "output_path": output_path,
        })
    
    return results


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_summary_visualization(results: List[dict]) -> str:
    """
    Create summary plot of detection results.
    
    Args:
        results: List of detection result dictionaries.
    
    Returns:
        Path to saved summary plot.
    """
    num_images = len(results)
    
    fig, axes = plt.subplots(num_images, 1, figsize=(12, 4 * num_images))
    if num_images == 1:
        axes = [axes]
    
    for ax, result_info in zip(axes, results):
        img = cv2.imread(result_info["output_path"])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
        ax.axis("off")
        
        title = f"{result_info['filename']}\nShips Detected: {result_info['ships']}"
        ax.set_title(title, fontsize=11, pad=10, fontweight="bold")
    
    plt.suptitle(
        "Ship Detection Quick Test Results",
        fontsize=14,
        fontweight="bold",
        y=0.995
    )
    plt.tight_layout()
    
    summary_path = os.path.join(
        constants.QUICK_TEST_OUTPUT_DIR,
        "ships_detection_summary.png"
    )
    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    return summary_path


# ============================================================================
# SUMMARY REPORTING
# ============================================================================

def print_summary_report(results: List[dict]) -> None:
    """Print summary of detection results."""
    total_images = len(results)
    total_ships = sum(r["ships"] for r in results)
    avg_ships = total_ships / total_images if total_images > 0 else 0.0
    
    print("\n" + "=" * 70)
    print("SHIP DETECTION RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nTotal images processed: {total_images}")
    print(f"Total ships detected: {total_ships}")
    print(f"Average ships per image: {avg_ships:.2f}")
    
    print("\nDetailed Results:")
    for i, result_info in enumerate(results, 1):
        print(f"\n  Image {i}: {result_info['filename']}")
        print(f"    - Ships detected: {result_info['ships']}")
    
    print(f"\n✓ All results saved in: {constants.QUICK_TEST_OUTPUT_DIR}/")
    print("=" * 70)
    print("\n✅ Ship detection test completed successfully!")


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    """Main quick test pipeline."""
    print("=" * 70)
    print("🛰️  SHIP DETECTION - Quick Test (YOLOv8)")
    print("=" * 70)
    
    # Collect images
    all_images = collect_available_images()
    
    if not all_images:
        print("\n❌ No images found! Please ensure the dataset is downloaded.")
        print("   Run: python download_dataset.py")
        return
    
    print(f"\n✓ Found {len(all_images)} images in dataset")
    
    # Load model
    print("\n[1/4] Loading YOLOv8 model for SHIP DETECTION...")
    model, is_trained_model = helpers.load_model()
    
    if is_trained_model:
        print("✓ Loaded TRAINED SHIP DETECTION model")
    else:
        print("⚠️  Using pre-trained YOLOv8 model (general detection)")
    
    # Select images for testing
    num_test = min(5, len(all_images))
    print(f"\n[2/4] Selecting {num_test} random images for testing...")
    selected_images = random.sample(all_images, num_test)
    
    # Create output directory
    constants.ensure_output_directories()
    
    # Run detection
    print(f"\n[3/4] Running SHIP DETECTION on {num_test} images...")
    print("-" * 70)
    results = run_detection_batch(model, selected_images, is_trained_model)
    
    # Create visualization
    print("\n[4/4] Creating summary visualization...")
    summary_path = create_summary_visualization(results)
    print(f"✓ Summary saved to: {summary_path}")
    
    # Print report
    print_summary_report(results)


if __name__ == "__main__":
    main()

