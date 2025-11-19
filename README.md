# Ship Detection from Satellite Imagery using YOLOv8

A YOLOv8 nano model for detecting ships in satellite imagery. **Ships only - all other object classes removed.**

## Overview

- **Model**: YOLOv8 Nano (lightweight, real-time)
- **Dataset**: DIOR (800x800 px satellite images, ship-only filtered)
- **Task**: Maritime vessel detection and tracking
- **Training**: 50 epochs, batch size 16

## Key Files

- `dior_object_detection.py` - Full pipeline (download, preprocess, train, test)
- `run_inference.py` - Run detection on images
- `run_video_inference.py` - Run detection on videos
- `quick_test.py` - Quick test on 5 random images
- `config.yaml` - Training configuration

## Usage

```bash
# Quick test
python3 quick_test.py

# Full pipeline
python3 dior_object_detection.py

# Detect on images
python3 run_inference.py

# Detect on video
python3 run_video_inference.py
```

## References

- YOLOv8 Docs: https://docs.ultralytics.com/
- DIOR Dataset: https://drive.google.com/open?id=1UdlgHk49iu6WpcJ5467iT-UqNPpx__CC
