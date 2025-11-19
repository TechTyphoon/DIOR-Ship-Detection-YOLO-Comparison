#!/usr/bin/env python3
"""
Download and extract DIOR dataset for ship detection.

This script downloads the DIOR (Object Detection In Optical Remote sensing images)
dataset components from Google Drive and extracts them locally.

SHIP DETECTION ONLY: Dataset will be filtered for ship annotations only.

Usage:
    python download_dataset.py
"""

import os
import shutil
from typing import Dict

import gdown

import constants


def log_info(message: str) -> None:
    """Print info message with consistent formatting."""
    print(f"[INFO] {message}")


def ensure_download_directory(directory: str) -> None:
    """Create directory for downloads if it doesn't exist."""
    if os.path.exists(directory):
        log_info(f"{directory} directory exists, checking for existing files...")
    else:
        os.makedirs(directory)
        log_info(f"{directory} directory created.")


def download_dataset_files(urls: Dict[str, str], output_dir: str) -> None:
    """
    Download dataset files from Google Drive.
    
    Args:
        urls: Dictionary mapping filename to download URL.
        output_dir: Directory to save downloaded files.
    """
    log_info("Downloading DIOR dataset files...\n")
    
    for filename, url in urls.items():
        filepath = os.path.join(output_dir, filename)
        
        if os.path.exists(filepath):
            log_info(f"{filename} already exists, skipping download.")
        else:
            log_info(f"Downloading {filename}...")
            gdown.download(url, output=filepath, quiet=False)
            log_info(f"{filename} downloaded successfully.\n")
    
    log_info("All files downloaded.\n")


def extract_dataset_files(
    filenames: list,
    source_dir: str,
    extract_dir: str
) -> None:
    """
    Extract dataset zip files.
    
    Args:
        filenames: List of zip filenames to extract.
        source_dir: Directory containing zip files.
        extract_dir: Directory to extract to.
    """
    log_info("Extracting dataset files...\n")
    
    if os.path.exists(extract_dir):
        log_info("DIOR data directory exists, checking contents...")
    else:
        os.makedirs(extract_dir)
        log_info("DIOR data directory created.")
    
    for filename in filenames:
        filepath = os.path.join(source_dir, filename)
        if os.path.exists(filepath):
            log_info(f"Extracting {filename}...")
            shutil.unpack_archive(filename=filepath, extract_dir=extract_dir)
            log_info(f"{filename} extracted successfully.\n")


def print_summary(dior_data_dir: str) -> None:
    """Print summary of downloaded dataset."""
    log_info("Dataset download and extraction complete!")
    log_info("Images are located in:")
    print(f"  - {os.path.join(dior_data_dir, 'JPEGImages-trainval')}")
    print(f"  - {os.path.join(dior_data_dir, 'JPEGImages-test')}")
    log_info("Annotations are located in:")
    print(f"  - {os.path.join(dior_data_dir, 'Annotations')}")


def main() -> None:
    """Download and extract DIOR dataset."""
    log_info("Starting dataset download for SHIP DETECTION project...\n")
    log_info("Note: Only ship annotations will be used\n")
    
    # Ensure directories exist
    ensure_download_directory(constants.RAW_DATA_DIR)
    
    # Download files
    download_dataset_files(
        constants.DATASET_URLS,
        constants.RAW_DATA_DIR
    )
    
    # Extract files
    extract_dataset_files(
        list(constants.DATASET_URLS.keys()),
        constants.RAW_DATA_DIR,
        constants.DIOR_DATA_DIR
    )
    
    # Print summary
    print_summary(constants.DIOR_DATA_DIR)


if __name__ == "__main__":
    main()

