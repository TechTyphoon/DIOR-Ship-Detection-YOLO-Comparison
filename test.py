#!/usr/bin/env python3
"""
Entry point for quick test with sample images
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from quick_test import main

if __name__ == '__main__':
    main()
