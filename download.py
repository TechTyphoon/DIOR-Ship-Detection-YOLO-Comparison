#!/usr/bin/env python3
"""
Entry point for downloading the DIOR dataset
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from download_dataset import main

if __name__ == '__main__':
    main()
