#!/usr/bin/env python3
"""
Generate Dataset Split Visualizations
======================================

This script generates comprehensive visualizations for different dataset split scenarios.
It creates pie charts, comparison tables, and bar charts for:
- Current Split: 82.7% : 17.3% (Train : Val)
- Standard 80/20: 80% : 20% (Train : Val)
- Balanced 70/15/15: 70% : 15% : 15% (Train : Val : Test)
- Balanced 60/20/20: 60% : 20% : 20% (Train : Val : Test)
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
    
    # Create output directory
    output_dir = 'runs/detect/metrics'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # You can adjust total_samples based on your actual dataset size
    # This is just for reference visualization
    total_samples = 1000
    
    # Create visualizer
    visualizer = DatasetSplitVisualizer(
        output_dir=output_dir,
        total_samples=total_samples
    )
    
    # Generate all visualizations
    visualizer.generate_all_visualizations()
    
    logger.info("\n✅ All visualizations have been generated successfully!")
    logger.info(f"📁 Check the '{output_dir}' directory for the generated images:\n")
    logger.info("   - split_82.7_17.3.png          (Current split)")
    logger.info("   - split_80_20.png              (80/20 split)")
    logger.info("   - split_70_15_15.png           (70/15/15 split)")
    logger.info("   - split_60_20_20.png           (60/20/20 split)")
    logger.info("   - splits_comparison_pie.png    (All splits side-by-side)")
    logger.info("   - splits_comparison_bars.png   (Bar chart comparison)")
    logger.info("   - splits_comparison_table.png  (Comparison table)")


if __name__ == '__main__':
    main()
