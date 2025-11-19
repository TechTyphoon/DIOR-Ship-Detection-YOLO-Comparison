#!/usr/bin/env python3
"""
Dataset Split Analysis - Show actual sample distribution comparison
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
sns.set_style("whitegrid")


def create_comparison_infographic(output_dir='runs/detect/metrics', dpi=300):
    """Create an infographic-style comparison of all splits."""
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Split data
    splits = {
        '82.7/17.3\n(Current)': [82.7, 17.3],
        '80/20\n(Standard)': [80, 20],
        '70/15/15\n(3-way)': [70, 15, 15],
        '60/20/20\n(3-way)': [60, 20, 20],
    }
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
    
    # Title
    fig.suptitle('Dataset Split Strategies - Complete Comparison', 
                 fontsize=18, weight='bold', y=0.98)
    
    # Color schemes
    colors_2way = ['#3498db', '#e74c3c']
    colors_3way = ['#3498db', '#e74c3c', '#2ecc71']
    
    # 1. Horizontal bar comparison
    ax1 = fig.add_subplot(gs[0, :])
    y_pos = np.arange(len(splits))
    split_names = list(splits.keys())
    
    ax1.set_title('Visual Percentage Comparison', fontsize=12, weight='bold', loc='left')
    
    for i, (split_name, percentages) in enumerate(splits.items()):
        start = 0
        colors = colors_2way if len(percentages) == 2 else colors_3way
        labels = ['Train', 'Val', 'Test'][:len(percentages)]
        
        for j, (pct, color, label) in enumerate(zip(percentages, colors, labels)):
            ax1.barh(i, pct, left=start, color=color, edgecolor='white', linewidth=2)
            # Add percentage text
            if pct >= 10:
                ax1.text(start + pct/2, i, f'{pct:.1f}%', 
                        ha='center', va='center', color='white', weight='bold', fontsize=11)
            start += pct
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(split_names, fontsize=11, weight='bold')
    ax1.set_xlim(0, 100)
    ax1.set_xlabel('Percentage (%)', fontsize=11, weight='bold')
    ax1.set_xticks([0, 20, 40, 60, 80, 100])
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3, axis='x')
    
    # 2-5. Individual detailed views
    positions = [(1, 0), (1, 1), (2, 0), (2, 1)]
    
    for idx, ((split_name, percentages), (row, col)) in enumerate(zip(splits.items(), positions)):
        ax = fig.add_subplot(gs[row, col])
        
        colors = colors_2way if len(percentages) == 2 else colors_3way
        labels = ['Train', 'Val', 'Test'][:len(percentages)]
        
        # Create pie chart
        wedges, texts, autotexts = ax.pie(
            percentages, 
            labels=labels, 
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            textprops={'fontsize': 10, 'weight': 'bold'},
            explode=[0.05] * len(percentages)
        )
        
        # Style text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(11)
            autotext.set_weight('bold')
        
        # Title with split info
        ax.set_title(split_name.replace('\n', ' '), fontsize=11, weight='bold', pad=10)
        
        # Add sample counts (for 1000 total samples)
        sample_info = '\n'.join([f'{l}: {int(1000*p/100):,}' for l, p in zip(labels, percentages)])
        ax.text(1.35, 0.5, sample_info, transform=ax.transAxes, 
               fontsize=9, verticalalignment='center', family='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'dataset_split_infographic.png'
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    logger.info(f"✓ Infographic saved to {output_path}")
    plt.close()


def create_recommendation_chart(output_dir='runs/detect/metrics', dpi=300):
    """Create a recommendation guide visualization."""
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('off')
    
    # Title
    title = "🎯 Dataset Split Recommendations\n"
    fig.text(0.5, 0.95, title, ha='center', fontsize=18, weight='bold')
    
    # Content
    recommendations = [
        ("Current: 82.7/17.3", [
            "✓ Close to standard 80/20",
            "✓ Good for most use cases",
            "✓ Focused training",
            "✓ Quick validation feedback",
            "✓ YOLOv8 compatible",
        ], "#3498db"),
        
        ("Standard: 80/20", [
            "✓ Most common split",
            "✓ Proven track record",
            "✓ Balanced approach",
            "✓ Recommended for general use",
            "✓ Good train/val balance",
        ], "#2ecc71"),
        
        ("Advanced: 70/15/15", [
            "✓ Separate test set",
            "✓ Unbiased evaluation",
            "✓ Better generalization test",
            "✓ 3-way split",
            "✓ Still good training data",
        ], "#f39c12"),
        
        ("Conservative: 60/20/20", [
            "✓ Maximum val/test coverage",
            "✓ Most rigorous evaluation",
            "✓ Best for critical applications",
            "✓ Equal val/test datasets",
            "✓ Less training data",
        ], "#e74c3c"),
    ]
    
    y_position = 0.85
    for title, points, color in recommendations:
        # Title box
        fig.text(0.08, y_position, title, fontsize=12, weight='bold', 
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.3, pad=0.8))
        
        y_position -= 0.05
        # Points
        for point in points:
            fig.text(0.12, y_position, point, fontsize=10, family='monospace')
            y_position -= 0.035
        
        y_position -= 0.02
    
    # Add footer
    fig.text(0.5, 0.02, 
            "For details, see: DATASET_SPLIT_GUIDE.md\nFor quick summary, see: DATASET_SPLIT_SUMMARY.md",
            ha='center', fontsize=9, style='italic', color='gray')
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'split_recommendations.png'
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    logger.info(f"✓ Recommendations chart saved to {output_path}")
    plt.close()


if __name__ == '__main__':
    logger.info("=" * 70)
    logger.info("GENERATING ADDITIONAL COMPARISON VISUALIZATIONS")
    logger.info("=" * 70)
    
    output_dir = 'runs/detect/metrics'
    
    logger.info("\n1. Creating comparison infographic...")
    create_comparison_infographic(output_dir)
    
    logger.info("2. Creating recommendations chart...")
    create_recommendation_chart(output_dir)
    
    logger.info("\n✅ Additional visualizations generated successfully!")
