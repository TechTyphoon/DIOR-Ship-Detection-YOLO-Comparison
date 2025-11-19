#!/usr/bin/env python3
"""
CONSTANTS - Configuration and constants for ship detection project
===================================================================

Centralized constants for paths, model configuration, class definitions,
and other configuration values used across the project.
"""

import os
from pathlib import Path
from typing import Dict

# ============================================================================
# DATASET & PATH CONSTANTS
# ============================================================================

# Raw dataset directories
RAW_DATA_DIR = "raw_data"
DIOR_DATA_DIR = "dior_data"

# Dataset split paths
ANNOTATIONS_DIR = os.path.join(DIOR_DATA_DIR, "Annotations", "Horizontal Bounding Boxes")
TRAINVAL_IMAGES_DIR = os.path.join(DIOR_DATA_DIR, "JPEGImages-trainval")
TEST_IMAGES_DIR = os.path.join(DIOR_DATA_DIR, "JPEGImages-test")

# Processed dataset paths
YOLO_ANNOTATIONS_DIR = os.path.join(DIOR_DATA_DIR, "yolo_annotations")

# Training dataset structure
DATASETS_ROOT = "datasets"
IMAGES_DIR = os.path.join(DATASETS_ROOT, "images")
LABELS_DIR = os.path.join(DATASETS_ROOT, "labels")

TRAIN_IMAGES_DIR = os.path.join(IMAGES_DIR, "train")
VAL_IMAGES_DIR = os.path.join(IMAGES_DIR, "val")
TEST_IMAGES_DIR_PROCESSED = os.path.join(IMAGES_DIR, "test")

TRAIN_LABELS_DIR = os.path.join(LABELS_DIR, "train")
VAL_LABELS_DIR = os.path.join(LABELS_DIR, "val")
TEST_LABELS_DIR = os.path.join(LABELS_DIR, "test")

# Model paths
MODEL_WEIGHTS_DIR = "runs/detect/yolov8n_epochs50_batch16/weights"
BEST_MODEL_PATH = os.path.join(MODEL_WEIGHTS_DIR, "best.pt")
PRETRAINED_MODEL_PATH = "yolov8n.pt"

# Output directories
INFERENCE_OUTPUT_DIR = "inference_results"
QUICK_TEST_OUTPUT_DIR = "quick_test_results"

# ============================================================================
# OBJECT DETECTION CONSTANTS
# ============================================================================

# Ship class definition
SHIP_CLASS = "ship"
SHIP_CLASS_ID = 0
CLASS_DICT = {SHIP_CLASS: SHIP_CLASS_ID}
CLASS_DICT_IDX = {SHIP_CLASS_ID: SHIP_CLASS}

# Image configuration
IMAGE_SIZE = 800
DEFAULT_IMAGE_HEIGHT = 800
DEFAULT_IMAGE_WIDTH = 800

# Detection thresholds and confidence
DEFAULT_CONFIDENCE_THRESHOLD = 0.25
DEFAULT_NMS_THRESHOLD = 0.5

# COCO class mapping for reference
COCO_BOAT_CLASS_ID = 8  # Boats in COCO dataset

# ============================================================================
# DATASET DOWNLOAD URLS (Google Drive)
# ============================================================================

DATASET_URLS = {
    "Annotations.zip": "https://drive.google.com/uc?id=1KoQzqR20qvIXDf1qsXCHGxD003IPmXMw",
    "JPEGImages-test.zip": "https://drive.google.com/uc?id=11SXPqcESez9qTn4Z5Q3v35K9hRwO_epr",
    "JPEGImages-trainval.zip": "https://drive.google.com/uc?id=1ZHbHDM6hYAEGDC_K5eiW0yF_lzVgpuir",
}

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

# Training hyperparameters
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 16
DEFAULT_IMG_SIZE = 800
DEFAULT_SEED = 42

# Validation split ratio
VALIDATION_SPLIT_RATIO = 0.2  # 20% of test set for validation

# ============================================================================
# VISUALIZATION CONSTANTS
# ============================================================================

# Plot output filenames
DATASET_DIST_PLOT = "ship_dataset_distribution.png"
DATASET_SPLIT_PLOT = "ship_dataset_split.png"
SHIP_INSTANCES_PLOT = "ship_instances_count.png"
SAMPLE_IMAGES_PLOT = "sample_ships.png"

# ============================================================================
# CONFIG FILE TEMPLATES
# ============================================================================

TRAINING_CONFIG_TEMPLATE = """# Path to ship detection dataset
path: 
train: 'images/train'
val: 'images/val'

# Ship detection - single class
names:
  0: ship
"""

TEST_CONFIG_TEMPLATE = """# Path to ship detection dataset
path:
train:
val: 'images/test'

# Ship detection - single class
names:
  0: ship
"""

# ============================================================================
# FILE PATTERNS
# ============================================================================

# File extensions
XML_EXTENSION = ".xml"
JPG_EXTENSION = ".jpg"
TXT_EXTENSION = ".txt"
PNG_EXTENSION = ".png"

# Image extensions for validation
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

# ============================================================================
# LOGGING & MESSAGE TEMPLATES
# ============================================================================

# Info messages
INFO_DATASET_EXISTS = "[INFO] {} directory exists, skipping {}."
INFO_CREATED = "[INFO] {} directory created."
INFO_DOWNLOADING = "[INFO] Downloading data...\n"
INFO_DOWNLOADED = "[INFO] Data has been downloaded."
INFO_EXTRACTING = '[INFO] File "{}" is being extracted to "{}".'
INFO_EXTRACTION_COMPLETE = "[INFO] All annotations converted to YOLO format."

# Success messages
SUCCESS_MODEL_LOADED = "   ✓ Model loaded!"
SUCCESS_SHIP_DETECTION_MODEL = "   ✓ Using TRAINED SHIP DETECTION model"
SUCCESS_SHIPS_ONLY = "   ✓ Single class: SHIP ONLY"

# Warning messages
WARNING_PRETRAINED_MODEL = "   ⚠️  Using pre-trained COCO model (general detection)"

# Error messages
ERROR_NO_IMAGES = "\n❌ No images found! Run: python3 download_dataset.py"
ERROR_VIDEO_NOT_FOUND = "\n❌ Video not found at: {}"

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def ensure_directory(directory_path: str) -> None:
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def ensure_directories(*directory_paths: str) -> None:
    """Create multiple directories if they don't exist."""
    for directory_path in directory_paths:
        ensure_directory(directory_path)


def get_all_dataset_dirs() -> Dict[str, str]:
    """Get dictionary of all dataset-related directories."""
    return {
        "raw_data": RAW_DATA_DIR,
        "dior_data": DIOR_DATA_DIR,
        "annotations": ANNOTATIONS_DIR,
        "trainval_images": TRAINVAL_IMAGES_DIR,
        "test_images": TEST_IMAGES_DIR,
        "yolo_annotations": YOLO_ANNOTATIONS_DIR,
        "train_images": TRAIN_IMAGES_DIR,
        "val_images": VAL_IMAGES_DIR,
        "test_images_processed": TEST_IMAGES_DIR_PROCESSED,
        "train_labels": TRAIN_LABELS_DIR,
        "val_labels": VAL_LABELS_DIR,
        "test_labels": TEST_LABELS_DIR,
    }
