#!/usr/bin/env python3
"""
HELPERS - Shared utility functions for ship detection project
==============================================================

Common functions for model loading, image processing, visualization,
and other utilities shared across multiple scripts.
"""

import os
from typing import List, Optional, Tuple
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from . import constants


# ============================================================================
# MODEL LOADING UTILITIES
# ============================================================================

def find_model_path() -> str:
    """
    Find the best available model.
    
    Returns available models in order of preference:
    1. Trained ship detection model (best.pt)
    2. Pre-trained YOLOv8 weights
    3. YOLOv8 model identifier (will download if needed)
    """
    model_candidates = [
        constants.BEST_MODEL_PATH,
        constants.PRETRAINED_MODEL_PATH,
        "yolov8n",  # Will download if needed
    ]
    
    for path in model_candidates:
        if os.path.exists(path) or path == "yolov8n":
            return path
    
    # Fallback to yolov8n
    return "yolov8n"


def load_model(model_path: Optional[str] = None) -> Tuple[YOLO, bool]:
    """
    Load YOLO model for ship detection.
    
    Args:
        model_path: Path to model weights. If None, searches for best available.
    
    Returns:
        Tuple of (model, is_ship_detection_model)
        - model: Loaded YOLO model
        - is_ship_detection_model: True if using trained ship model, False if pre-trained
    """
    if model_path is None:
        model_path = find_model_path()
    
    model = YOLO(model_path)
    is_trained_ship_model = "best.pt" in str(model_path)
    
    return model, is_trained_ship_model


def get_model_info(model_path: str, is_trained: bool) -> dict:
    """
    Get information about the model being used.
    
    Returns:
        Dictionary with model info.
    """
    return {
        "path": model_path,
        "is_trained_ship_model": is_trained,
        "status": "TRAINED SHIP DETECTION" if is_trained else "Pre-trained COCO",
        "classes": ["ship"] if is_trained else "multiple (general detection)",
    }


# ============================================================================
# IMAGE UTILITIES
# ============================================================================

def find_images_in_directories(*directory_paths: str) -> List[Tuple[str, str]]:
    """
    Find all images in given directories.
    
    Args:
        *directory_paths: Variable number of directory paths to search.
    
    Returns:
        List of tuples: (filename, directory_path)
    """
    all_images = []
    
    for directory in directory_paths:
        if not os.path.exists(directory):
            continue
        
        for filename in os.listdir(directory):
            if any(filename.lower().endswith(ext) for ext in constants.IMAGE_EXTENSIONS):
                all_images.append((filename, directory))
    
    return all_images


def get_image_path(filename: str, *search_directories: str) -> Optional[str]:
    """
    Get full path to image by searching in multiple directories.
    
    Args:
        filename: Name of image file.
        *search_directories: Directories to search in order.
    
    Returns:
        Full path to image if found, None otherwise.
    """
    for directory in search_directories:
        full_path = os.path.join(directory, filename)
        if os.path.isfile(full_path):
            return full_path
    
    return None


# ============================================================================
# DETECTION UTILITIES
# ============================================================================

def count_ships_in_detections(
    result,
    is_trained_model: bool,
    confidence_threshold: float = constants.DEFAULT_CONFIDENCE_THRESHOLD
) -> int:
    """
    Count number of ships detected in YOLO result.
    
    For trained ship detection model: class_id == 0
    For COCO model: class_id == 8 (boat class)
    
    Args:
        result: YOLO prediction result object.
        is_trained_model: Whether using trained ship detection model.
        confidence_threshold: Minimum confidence to count detection.
    
    Returns:
        Number of ships detected above confidence threshold.
    """
    if not hasattr(result, 'boxes') or len(result.boxes) == 0:
        return 0
    
    ship_count = 0
    target_class_id = constants.SHIP_CLASS_ID if is_trained_model else constants.COCO_BOAT_CLASS_ID
    
    for box in result.boxes:
        cls_id = int(box.cls[0])
        confidence = float(box.conf[0])
        
        if cls_id == target_class_id and confidence >= confidence_threshold:
            ship_count += 1
    
    return ship_count


def run_detection_on_image(
    model: YOLO,
    image_path: str,
    image_size: int = constants.IMAGE_SIZE,
    confidence: float = constants.DEFAULT_CONFIDENCE_THRESHOLD,
    verbose: bool = False
) -> Tuple[np.ndarray, object]:
    """
    Run ship detection on single image.
    
    Args:
        model: Loaded YOLO model.
        image_path: Path to image file.
        image_size: Input image size for model.
        confidence: Confidence threshold.
        verbose: Print verbose output.
    
    Returns:
        Tuple of (annotated_image, result_object)
    """
    results = model.predict(
        source=image_path,
        imgsz=image_size,
        conf=confidence,
        save=False,
        verbose=verbose
    )
    
    result = results[0]
    annotated = result.plot()
    
    return annotated, result


# ============================================================================
# DIRECTORY UTILITIES
# ============================================================================

def ensure_output_directories() -> None:
    """Ensure all required output directories exist."""
    constants.ensure_directories(
        constants.INFERENCE_OUTPUT_DIR,
        constants.QUICK_TEST_OUTPUT_DIR,
    )


def get_latest_predict_dir() -> Optional[str]:
    """
    Get path to latest prediction directory created by YOLO.
    
    Returns:
        Path to latest 'predict*' directory or None if not found.
    """
    predict_base = "runs/detect"
    
    if not os.path.exists(predict_base):
        return None
    
    predict_dirs = [
        d for d in os.listdir(predict_base) 
        if d.startswith("predict") and os.path.isdir(os.path.join(predict_base, d))
    ]
    
    if not predict_dirs:
        return None
    
    # Get most recently modified directory
    latest = max(
        predict_dirs,
        key=lambda d: os.path.getmtime(os.path.join(predict_base, d))
    )
    
    return os.path.join(predict_base, latest)


# ============================================================================
# VISUALIZATION UTILITIES
# ============================================================================

def get_bbox_coordinates_from_yolo_annotation(
    annotation_text: str,
    image_width: int,
    image_height: int
) -> List[Tuple[int, int, int, int]]:
    """
    Convert YOLO format annotations to bounding box coordinates.
    
    YOLO format: class_id center_x center_y width height (all normalized)
    
    Args:
        annotation_text: YOLO annotation line.
        image_width: Image width in pixels.
        image_height: Image height in pixels.
    
    Returns:
        List of tuples: (xmin, ymin, xmax, ymax)
    """
    parts = annotation_text.strip().split()
    if len(parts) < 5:
        return []
    
    _, center_x, center_y, width, height = parts[:5]
    
    center_x = float(center_x)
    center_y = float(center_y)
    width = float(width)
    height = float(height)
    
    xmin = int((center_x - width / 2) * image_width)
    ymin = int((center_y - height / 2) * image_height)
    xmax = int((center_x + width / 2) * image_width)
    ymax = int((center_y + height / 2) * image_height)
    
    # Clip to image boundaries
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(image_width - 1, xmax)
    ymax = min(image_height - 1, ymax)
    
    return [(xmin, ymin, xmax, ymax)]


def draw_bbox_on_image(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    label: str = constants.SHIP_CLASS,
    color: Tuple[int, int, int] = (255, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Draw bounding box on image.
    
    Args:
        image: Image array (BGR format).
        bbox: Bounding box as (xmin, ymin, xmax, ymax).
        label: Label text.
        color: Box color in BGR format.
        thickness: Line thickness.
    
    Returns:
        Modified image array.
    """
    image = image.copy()
    xmin, ymin, xmax, ymax = bbox
    
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness)
    cv2.putText(
        image,
        label,
        (xmin, max(0, ymin - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2
    )
    
    return image


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_detection_summary(
    detection_results: List[dict],
) -> dict:
    """
    Generate summary statistics from detection results.
    
    Args:
        detection_results: List of detection result dicts with 'ships' key.
    
    Returns:
        Summary dictionary with statistics.
    """
    total_ships = sum(r.get("ships", 0) for r in detection_results)
    total_images = len(detection_results)
    avg_ships = total_ships / total_images if total_images > 0 else 0
    
    return {
        "total_images_processed": total_images,
        "total_ships_detected": total_ships,
        "average_ships_per_image": avg_ships,
        "images": detection_results,
    }
