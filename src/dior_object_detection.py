#!/usr/bin/env python3
"""
Ship Detection on Satellite Imagery using YOLOv8 - Complete Pipeline

Object detection model trained specifically for detecting ships in satellite/remote
sensing images using the DIOR dataset. This script performs the complete pipeline:
1. Download and extraction of dataset
2. Data preprocessing and exploratory analysis
3. Annotation format conversion (XML to YOLO format)
4. Model training
5. Model validation and testing

SHIP DETECTION ONLY: All non-ship object classes are filtered out.
"""

import os
import shutil
import random
from typing import Dict, List, Tuple
from collections import Counter
from xml.etree import ElementTree

import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from ultralytics import YOLO

import constants


# ============================================================================
# DATASET DOWNLOAD & EXTRACTION
# ============================================================================

def download_and_extract_dataset() -> None:
    """Download DIOR dataset from Google Drive and extract locally."""
    print("[INFO] Downloading dataset for SHIP DETECTION project...\n")
    print("[INFO] Note: Only ship annotations will be used.\n")
    
    # Ensure raw data directory exists
    if os.path.exists(constants.RAW_DATA_DIR):
        print("[INFO] Raw data directory exists, skipping download.")
    else:
        os.makedirs(constants.RAW_DATA_DIR)
        print("[INFO] Raw data directory created.")
        print("[INFO] Downloading data...\n")
        
        import gdown
        for filename, url in constants.DATASET_URLS.items():
            output_path = os.path.join(constants.RAW_DATA_DIR, filename)
            gdown.download(url, output=output_path)
        
        print("[INFO] Data download completed.")
    
    # Extract files
    if os.path.exists(constants.DIOR_DATA_DIR):
        print("[INFO] DIOR data directory exists, skipping extraction.")
    else:
        os.makedirs(constants.DIOR_DATA_DIR)
        print("[INFO] Extracting dataset files...\n")
        
        for filename in os.listdir(constants.RAW_DATA_DIR):
            if filename.endswith(".zip"):
                filepath = os.path.join(constants.RAW_DATA_DIR, filename)
                shutil.unpack_archive(filename=filepath, extract_dir=constants.DIOR_DATA_DIR)
                print(f'[INFO] Extracted "{filename}" to "{constants.DIOR_DATA_DIR}".')


# ============================================================================
# DATA COLLECTION
# ============================================================================

def collect_dataset_files() -> Tuple[List[str], List[str], List[str], np.ndarray]:
    """
    Collect all annotation and image files from dataset directories.
    
    Returns:
        Tuple of (annotation_files, trainval_images, test_images, all_images)
    """
    annot_files = sorted([
        os.path.join(constants.ANNOTATIONS_DIR, f)
        for f in os.listdir(constants.ANNOTATIONS_DIR)
        if f.endswith(constants.XML_EXTENSION)
    ])
    
    trainval_files = sorted([
        os.path.join(constants.TRAINVAL_IMAGES_DIR, f)
        for f in os.listdir(constants.TRAINVAL_IMAGES_DIR)
        if f.endswith(constants.JPG_EXTENSION)
    ])
    
    test_files = sorted([
        os.path.join(constants.TEST_IMAGES_DIR, f)
        for f in os.listdir(constants.TEST_IMAGES_DIR)
        if f.endswith(constants.JPG_EXTENSION)
    ])
    
    all_images = np.concatenate((trainval_files, test_files))
    
    return annot_files, trainval_files, test_files, all_images


# ============================================================================
# METADATA EXTRACTION & EDA
# ============================================================================

def extract_metadata_from_annotations(
    annotation_files: List[str],
    image_files: np.ndarray
) -> Tuple[List[dict], Counter, Counter, Dict[str, int]]:
    """
    Extract metadata and object information from annotation XML files.
    Only SHIP annotations are retained.
    
    Returns:
        Tuple of (metadata_list, train_counter, test_counter, class_dict)
    """
    metadata_list = []
    object_list_train = []
    object_list_test = []
    
    for xml_file in tqdm(annotation_files, desc="Processing annotations"):
        meta_dict = {}
        root = ElementTree.parse(xml_file).getroot()
        
        # Extract filename and split type
        filename_text = root.find('filename').text
        image_path = None
        for img_path in image_files:
            if filename_text in img_path:
                image_path = img_path
                break
        
        if image_path is None:
            continue
        
        meta_dict['filename'] = image_path
        meta_dict['split_type'] = image_path.split(os.sep)[1]
        
        # Extract image dimensions
        size_elem = root.find('size')
        meta_dict['width'] = int(size_elem.find('width').text)
        meta_dict['height'] = int(size_elem.find('height').text)
        
        # Extract objects (only ships)
        ship_objects = [
            obj.find('name').text
            for obj in root.findall('object')
            if obj.find('name').text.lower() == constants.SHIP_CLASS.lower()
        ]
        
        meta_dict['objects'] = ', '.join(np.unique(ship_objects)) if ship_objects else ""
        
        if ship_objects:
            metadata_list.append(meta_dict)
            
            trainval_split = constants.TRAINVAL_IMAGES_DIR.split(os.sep)[1]
            if meta_dict['split_type'] == trainval_split:
                object_list_train.extend(ship_objects)
            else:
                object_list_test.extend(ship_objects)
    
    # Count instances and create class dictionary
    instance_train = Counter(object_list_train)
    instance_test = Counter(object_list_test)
    class_dict = {k: v for v, k in enumerate(sorted(set(object_list_train)))}
    
    print(f"[INFO] Found {len(metadata_list)} images with ships")
    print(f"[INFO] Class dictionary: {class_dict}")
    
    return metadata_list, instance_train, instance_test, class_dict


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_eda_visualizations(
    metadata_df: pd.DataFrame,
    instance_train: Counter,
    instance_test: Counter
) -> None:
    """Create exploratory data analysis visualizations."""
    print("\n[INFO] Generating EDA visualizations...\n")
    
    # Dataset distribution
    plt.figure(figsize=(10, 6))
    plt.barh([constants.SHIP_CLASS], [len(metadata_df)])
    plt.xlabel("Images (Count)")
    plt.title("Ship Detection Dataset")
    plt.savefig(constants.DATASET_DIST_PLOT, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Dataset distribution saved to {constants.DATASET_DIST_PLOT}")
    
    # Dataset split
    plt.figure(figsize=(12, 6))
    trainval_split = constants.TRAINVAL_IMAGES_DIR.split(os.sep)[1]
    test_split = constants.TEST_IMAGES_DIR.split(os.sep)[1]
    train_count = len(metadata_df[metadata_df["split_type"] == trainval_split])
    test_count = len(metadata_df[metadata_df["split_type"] == test_split])
    
    plt.bar(["Train/Val", "Test"], [train_count, test_count], color=["#1f77b4", "#ff7f0e"])
    plt.ylabel("Number of Images")
    plt.title("Ship Detection Dataset Split", fontsize=16)
    plt.savefig(constants.DATASET_SPLIT_PLOT, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Dataset split saved to {constants.DATASET_SPLIT_PLOT}")
    
    # Ship instances count
    plt.figure(figsize=(12, 6))
    plt.bar(
        ["Train/Val Ships", "Test Ships"],
        [instance_train.get(constants.SHIP_CLASS, 0), instance_test.get(constants.SHIP_CLASS, 0)],
        color=["#2ca02c", "#d62728"]
    )
    plt.ylabel("Number of Ship Instances")
    plt.title("Total Ship Instances Per Dataset Split", fontsize=16)
    plt.savefig(constants.SHIP_INSTANCES_PLOT, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Ship instances saved to {constants.SHIP_INSTANCES_PLOT}")
    
    # Sample images with ships
    if len(metadata_df) > 0:
        viz_list = metadata_df["filename"].tolist()
        plt.figure(figsize=(20, 5))
        rand_samples = random.sample(viz_list, min(4, len(viz_list)))
        
        for i, img_path in enumerate(rand_samples, 1):
            plt.subplot(1, 4, i)
            plt.imshow(plt.imread(img_path))
            plt.axis(False)
        
        plt.suptitle("Sample Images with Ships", fontsize=20, fontweight="bold")
        plt.savefig(constants.SAMPLE_IMAGES_PLOT, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Sample images saved to {constants.SAMPLE_IMAGES_PLOT}")


# ============================================================================
# ANNOTATION CONVERSION (XML to YOLO)
# ============================================================================

def extract_data_from_xml(xml_file: str) -> Dict:
    """
    Extract annotation data from XML file.
    SHIP DETECTION ONLY - Only extracts ship annotations.
    
    Args:
        xml_file: Path to XML annotation file.
    
    Returns:
        Dictionary containing filename, image_size, and bounding boxes.
    """
    root = ElementTree.parse(xml_file).getroot()
    
    data_dict = {'filename': None, 'image_size': None, 'bboxes': []}
    
    for element in root:
        # Get filename
        if element.tag == 'filename':
            data_dict['filename'] = element.text
        
        # Get image size
        elif element.tag == 'size':
            image_size = []
            for size_element in element:
                image_size.append(int(size_element.text))
            data_dict['image_size'] = image_size
        
        # Get bounding boxes (only ships)
        elif element.tag == 'object':
            # Check if object is a ship
            obj_name = element.find('name')
            if obj_name is None or obj_name.text.lower() != constants.SHIP_CLASS.lower():
                continue
            
            bbox = {'class': constants.SHIP_CLASS}
            bbox_elem = element.find('bndbox')
            
            if bbox_elem is not None:
                for coord in ['xmin', 'ymin', 'xmax', 'ymax']:
                    coord_elem = bbox_elem.find(coord)
                    if coord_elem is not None:
                        bbox[coord] = int(coord_elem.text)
            
            data_dict['bboxes'].append(bbox)
    
    return data_dict


def convert_annotation_to_yolo(data_dict: Dict) -> None:
    """
    Convert annotation from dictionary format to YOLO format.
    SHIP DETECTION ONLY - Only processes ship annotations.
    
    Args:
        data_dict: Dictionary with annotation data.
    """
    if not data_dict['filename'] or not data_dict['image_size']:
        return
    
    data_lines = []
    img_w, img_h = data_dict['image_size'][0], data_dict['image_size'][1]
    
    for bbox in data_dict['bboxes']:
        if bbox['class'].lower() != constants.SHIP_CLASS.lower():
            continue
        
        # Convert bbox to YOLO format (normalized center coordinates)
        class_id = constants.SHIP_CLASS_ID
        x_center = ((bbox['xmin'] + bbox['xmax']) / 2) / img_w
        y_center = ((bbox['ymin'] + bbox['ymax']) / 2) / img_h
        width = (bbox['xmax'] - bbox['xmin']) / img_w
        height = (bbox['ymax'] - bbox['ymin']) / img_h
        
        data_lines.append(f'{class_id} {x_center:.3f} {y_center:.3f} {width:.3f} {height:.3f}')
    
    if not data_lines:
        return
    
    # Save YOLO format annotation
    constants.ensure_directory(constants.YOLO_ANNOTATIONS_DIR)
    output_filename = data_dict['filename'].replace(constants.JPG_EXTENSION, constants.TXT_EXTENSION)
    output_path = os.path.join(constants.YOLO_ANNOTATIONS_DIR, os.path.basename(output_filename))
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(data_lines))


def convert_all_annotations_to_yolo(annotation_files: List[str]) -> None:
    """Convert all XML annotations to YOLO format."""
    print('[INFO] Converting annotations to YOLO format...')
    
    for annot_file in tqdm(annotation_files):
        data_dict = extract_data_from_xml(annot_file)
        convert_annotation_to_yolo(data_dict)
    
    print('[INFO] All annotations converted to YOLO format.')


# ============================================================================
# VISUALIZATION WITH BOUNDING BOXES
# ============================================================================

def plot_bounding_boxes(
    img_path: str,
    annot_path: str,
    class_dict: Dict[int, str]
) -> None:
    """
    Plot image with bounding boxes.
    
    Args:
        img_path: Path to image file.
        annot_path: Path to YOLO annotation file.
        class_dict: Dictionary mapping class IDs to class names.
    """
    image = cv2.imread(img_path)
    if image is None:
        return
    
    img_h, img_w = image.shape[:2]
    
    try:
        with open(annot_path, 'r') as f:
            annotations = f.read().strip().split('\n')
    except FileNotFoundError:
        return
    
    for annotation in annotations:
        if not annotation.strip():
            continue
        
        parts = annotation.split()
        if len(parts) < 5:
            continue
        
        class_id = int(float(parts[0]))
        x_center = float(parts[1]) * img_w
        y_center = float(parts[2]) * img_h
        width = float(parts[3]) * img_w
        height = float(parts[4]) * img_h
        
        xmin = max(0, int(x_center - width / 2))
        ymin = max(0, int(y_center - height / 2))
        xmax = min(img_w - 1, int(x_center + width / 2))
        ymax = min(img_h - 1, int(y_center + height / 2))
        
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 255, 0), 2)
        label = class_dict.get(class_id, f'Class {class_id}')
        cv2.putText(
            image,
            label,
            (xmin, max(0, ymin - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 0),
            2
        )
    
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis(False)


# ============================================================================
# DATASET ORGANIZATION FOR TRAINING
# ============================================================================

def organize_dataset_for_training(
    trainval_files: List[str],
    test_files: List[str],
    val_split_ratio: float = 0.2
) -> None:
    """
    Organize dataset into train/val/test directories for YOLO training.
    
    Args:
        trainval_files: List of training/validation image files.
        test_files: List of test image files.
        val_split_ratio: Proportion of test set to use for validation.
    """
    print("[INFO] Organizing dataset for training...")
    
    # Create directories
    constants.ensure_directories(
        constants.TRAIN_IMAGES_DIR,
        constants.VAL_IMAGES_DIR,
        constants.TEST_IMAGES_DIR_PROCESSED,
        constants.TRAIN_LABELS_DIR,
        constants.VAL_LABELS_DIR,
        constants.TEST_LABELS_DIR,
    )
    
    # Calculate split
    val_count = int(len(test_files) * val_split_ratio)
    val_files = test_files[:val_count]
    test_remaining = test_files[val_count:]
    
    # Move training images and labels
    print("[INFO] Moving training images and labels...")
    for filepath in tqdm(trainval_files):
        if os.path.isfile(filepath):
            shutil.move(filepath, constants.TRAIN_IMAGES_DIR)
        
        label_file = os.path.join(
            constants.YOLO_ANNOTATIONS_DIR,
            os.path.basename(filepath).replace(constants.JPG_EXTENSION, constants.TXT_EXTENSION)
        )
        if os.path.isfile(label_file):
            shutil.move(label_file, constants.TRAIN_LABELS_DIR)
    
    # Move validation images and labels
    print("[INFO] Moving validation images and labels...")
    for filepath in tqdm(val_files):
        if os.path.isfile(filepath):
            shutil.move(filepath, constants.VAL_IMAGES_DIR)
        
        label_file = os.path.join(
            constants.YOLO_ANNOTATIONS_DIR,
            os.path.basename(filepath).replace(constants.JPG_EXTENSION, constants.TXT_EXTENSION)
        )
        if os.path.isfile(label_file):
            shutil.move(label_file, constants.VAL_LABELS_DIR)
    
    # Move test images and labels
    print("[INFO] Moving test images and labels...")
    for filepath in tqdm(test_remaining):
        if os.path.isfile(filepath):
            shutil.move(filepath, constants.TEST_IMAGES_DIR_PROCESSED)
        
        label_file = os.path.join(
            constants.YOLO_ANNOTATIONS_DIR,
            os.path.basename(filepath).replace(constants.JPG_EXTENSION, constants.TXT_EXTENSION)
        )
        if os.path.isfile(label_file):
            shutil.move(label_file, constants.TEST_LABELS_DIR)


# ============================================================================
# TRAINING & VALIDATION
# ============================================================================

def create_training_config() -> None:
    """Create YOLO training configuration file."""
    with open('config.yaml', 'w') as f:
        f.write(constants.TRAINING_CONFIG_TEMPLATE)
    print('[INFO] config.yaml created.')


def create_test_config() -> None:
    """Create YOLO test configuration file."""
    with open('test_config.yaml', 'w') as f:
        f.write(constants.TEST_CONFIG_TEMPLATE)
    print('[INFO] test_config.yaml created.')


def train_model() -> None:
    """Train YOLOv8 model on ship detection dataset."""
    print("[INFO] Training YOLOv8 model...")
    create_training_config()
    
    model = YOLO('yolov8n.yaml')
    results = model.train(
        data='config.yaml',
        imgsz=constants.IMAGE_SIZE,
        epochs=constants.DEFAULT_EPOCHS,
        batch=constants.DEFAULT_BATCH_SIZE,
        name='yolov8n_epochs50_batch16'
    )
    
    print("[INFO] Training completed.")
    return results


def validate_model() -> None:
    """Validate trained model on test set."""
    print("[INFO] Validating model on test set...")
    create_test_config()
    
    model = YOLO(constants.BEST_MODEL_PATH)
    results = model.val(
        data='test_config.yaml',
        imgsz=constants.IMAGE_SIZE,
        name='yolov8n_val_on_test'
    )
    
    print("[INFO] Validation completed.")
    return results


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main() -> None:
    """Execute complete ship detection pipeline."""
    print("\n" + "=" * 70)
    print("SHIP DETECTION FROM SATELLITE IMAGERY - YOLOv8")
    print("=" * 70 + "\n")
    
    # Step 1: Download and extract dataset
    print("\n[STEP 1] DOWNLOAD & EXTRACT DATASET")
    print("-" * 70)
    download_and_extract_dataset()
    
    # Step 2: Collect dataset files
    print("\n[STEP 2] COLLECT DATASET FILES")
    print("-" * 70)
    annot_files, trainval_files, test_files, all_images = collect_dataset_files()
    print(f"[INFO] Found {len(annot_files)} annotations")
    print(f"[INFO] Found {len(trainval_files)} training images")
    print(f"[INFO] Found {len(test_files)} test images")
    
    # Step 3: Extract metadata and perform EDA
    print("\n[STEP 3] EXTRACT METADATA & PERFORM EDA")
    print("-" * 70)
    metadata_list, instance_train, instance_test, class_dict = extract_metadata_from_annotations(annot_files, all_images)
    metadata_df = pd.DataFrame(metadata_list)
    create_eda_visualizations(metadata_df, instance_train, instance_test)
    
    # Step 4: Convert annotations to YOLO format
    print("\n[STEP 4] CONVERT ANNOTATIONS TO YOLO FORMAT")
    print("-" * 70)
    convert_all_annotations_to_yolo(annot_files)
    
    # Step 5: Organize dataset for training
    print("\n[STEP 5] ORGANIZE DATASET FOR TRAINING")
    print("-" * 70)
    organize_dataset_for_training(trainval_files, test_files, val_split_ratio=0.2)
    
    # Step 6: Train model
    print("\n[STEP 6] TRAIN MODEL")
    print("-" * 70)
    train_model()
    
    # Step 7: Validate model
    print("\n[STEP 7] VALIDATE MODEL")
    print("-" * 70)
    validate_model()
    
    print("\n" + "=" * 70)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
