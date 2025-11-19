#!/usr/bin/env python3
"""
Entry point for image inference
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from run_inference import main

if __name__ == '__main__':
    main()
