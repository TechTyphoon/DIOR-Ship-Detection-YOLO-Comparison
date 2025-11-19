"""
Ship Detection on Satellite Imagery using YOLOv8
================================================

Core module for DIOR dataset ship detection pipeline.
Includes configuration, utilities, and main execution scripts.
"""

from .constants import *
from .helpers import *
from .utils_metrics import *

__all__ = [
    'constants',
    'helpers',
    'utils_metrics',
    'download_dataset',
    'dior_object_detection',
    'quick_test',
    'run_inference',
    'run_video_inference',
    'evaluate',
]
