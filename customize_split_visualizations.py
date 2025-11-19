#!/usr/bin/env python3
"""
CUSTOMIZE_SPLIT_VISUALIZATIONS - Generate split visualizations for your actual dataset

This script allows you to generate dataset split visualizations customized for your 
actual dataset size. Update the total_samples variable to match your dataset size.

Usage:
    python customize_split_visualizations.py

Then adjust total_samples parameter based on your actual dataset size.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import logging
from dataset_split_visualizer import DatasetSplitVisualizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main execution function."""
    
    # ============================================================
    # ⚙️  CONFIGURATION - ADJUST THESE VALUES
    # ============================================================
    
    # Total number of samples in your dataset
    # Count the number of images in your dataset and update this value
    total_samples = 1000  # ← UPDATE THIS VALUE
    
    # Output directory for visualizations
    output_dir = 'runs/detect/metrics'
    
    # ============================================================
    # END CONFIGURATION
    # ============================================================
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info("\n" + "=" * 70)
    logger.info("CUSTOM DATASET SPLIT VISUALIZER")
    logger.info("=" * 70)
    logger.info(f"\n📊 Generating visualizations with:")
    logger.info(f"   Total Samples: {total_samples:,}")
    logger.info(f"   Output Directory: {output_dir}\n")
    
    # Create visualizer with custom total samples
    visualizer = DatasetSplitVisualizer(
        output_dir=output_dir,
        total_samples=total_samples
    )
    
    # Generate all visualizations
    visualizer.generate_all_visualizations()
    
    logger.info("\n✅ All visualizations have been generated successfully!")
    logger.info(f"\n📁 Generated files in '{output_dir}':")
    logger.info("   ✓ split_82.7_17.3(Current).png")
    logger.info("   ✓ split_80_20.png")
    logger.info("   ✓ split_70_15_15.png")
    logger.info("   ✓ split_60_20_20.png")
    logger.info("   ✓ splits_comparison_pie.png")
    logger.info("   ✓ splits_comparison_bars.png")
    logger.info("   ✓ splits_comparison_table.png")
    
    logger.info("\n📖 For detailed information, read: DATASET_SPLIT_GUIDE.md")
    logger.info("📄 For quick summary, read: DATASET_SPLIT_SUMMARY.md\n")


if __name__ == '__main__':
    main()
