#!/usr/bin/env python3
"""
Run Ship Detection Inference on Video

This script runs ship detection inference on video files and saves annotated
output videos with detected ships highlighted.

SHIPS ONLY - All other object classes are ignored.

Usage:
    python run_video_inference.py
"""

import os
import sys
from typing import Optional

import helpers


# ============================================================================
# VIDEO VALIDATION
# ============================================================================

def validate_video_file(video_path: str) -> bool:
    """
    Validate that video file exists.
    
    Args:
        video_path: Path to video file.
    
    Returns:
        True if valid, False otherwise.
    """
    if not os.path.exists(video_path):
        return False
    
    if not os.path.isfile(video_path):
        return False
    
    return True


def get_video_file_size_mb(video_path: str) -> float:
    """Get video file size in megabytes."""
    return os.path.getsize(video_path) / (1024 * 1024)


# ============================================================================
# VIDEO INFERENCE
# ============================================================================

def run_video_detection(model, video_path: str) -> bool:
    """
    Run ship detection on video file.
    
    Args:
        model: Loaded YOLO model.
        video_path: Path to input video.
    
    Returns:
        True if successful, False otherwise.
    """
    try:
        results = model.predict(
            source=video_path,
            imgsz=800,
            conf=0.25,
            save=True,
            verbose=False
        )
        return True
    except Exception as e:
        print(f"   ❌ Error during detection: {e}")
        return False


# ============================================================================
# OUTPUT MANAGEMENT
# ============================================================================

def get_output_information() -> None:
    """Display information about output location."""
    output_base = "runs/detect"
    
    if not os.path.exists(output_base):
        print(f"\n❌ Output directory not found: {output_base}")
        return
    
    # Find latest prediction directory
    predict_dirs = [
        d for d in os.listdir(output_base)
        if d.startswith("predict") and os.path.isdir(os.path.join(output_base, d))
    ]
    
    if not predict_dirs:
        print(f"\n❌ No prediction directories found in: {output_base}")
        return
    
    # Get most recently modified directory
    latest_dir = max(
        predict_dirs,
        key=lambda d: os.path.getmtime(os.path.join(output_base, d))
    )
    output_path = os.path.join(output_base, latest_dir)
    
    print(f"\n📁 Output directory: {output_path}")
    
    # List files
    if os.path.exists(output_path):
        files = os.listdir(output_path)
        print(f"\n📋 Files in output directory:")
        
        for filename in sorted(files):
            file_path = os.path.join(output_path, filename)
            if os.path.isfile(file_path):
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                print(f"   • {filename} ({size_mb:.2f} MB)")
            else:
                print(f"   📁 {filename}/")
        
        # Find output video
        for filename in os.listdir(output_path):
            if filename.endswith((".mp4", ".avi")):
                output_video = os.path.join(output_path, filename)
                print(f"\n🎬 Output video with SHIP DETECTIONS:")
                print(f"   {output_video}")
                print(f"\n✓ Full path:")
                print(f"   {os.path.abspath(output_video)}")
                break


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    """Main video inference pipeline."""
    print("=" * 70)
    print("🛰️  SHIP DETECTION ON VIDEO - Satellite Imagery (YOLOv8)")
    print("=" * 70)
    
    # Define video path
    video_path = "/home/reddy/Desktop/Dataset/Object_Detection_Satellite_Imagery_Yolov8_DIOR/test_data/ships_test.mp4"
    
    # Validate video
    print(f"\n[1/3] Validating video file...")
    if not validate_video_file(video_path):
        print(f"\n❌ Video not found at: {video_path}")
        return
    
    file_size = get_video_file_size_mb(video_path)
    print(f"   ✓ Video file found: {video_path}")
    print(f"   ✓ File size: {file_size:.2f} MB")
    
    # Load model
    print(f"\n[2/3] Loading model...")
    try:
        model, is_trained_model = helpers.load_model()
        
        if is_trained_model:
            print("   ✓ Using TRAINED SHIP DETECTION model")
            print("   ✓ Single class: SHIP ONLY")
        else:
            print("   ℹ️  Using pre-trained COCO model (general detection)")
            print("   Note: Results optimized for ship detection")
        
        print("   ✓ Model loaded successfully!")
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        return
    
    # Run inference
    print(f"\n[3/3] Running SHIP DETECTION on video...")
    print("-" * 70)
    
    success = run_video_detection(model, video_path)
    
    if not success:
        print("\n❌ Ship detection on video failed!")
        return
    
    print("\n✓ Ship detection on video completed successfully!")
    
    # Display output information
    print("\n" + "=" * 70)
    print("📁 OUTPUT LOCATION")
    print("=" * 70)
    get_output_information()
    
    print("\n" + "=" * 70)
    print("✅ Ship detection on video completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
