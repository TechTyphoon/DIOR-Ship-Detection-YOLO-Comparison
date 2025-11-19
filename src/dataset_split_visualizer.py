#!/usr/bin/env python3
"""
DATASET_SPLIT_VISUALIZER - Dataset Split Comparison Visualizations
===================================================================

This module creates professional visualizations for different dataset split scenarios:
1. Current Split: 82.7% : 17.3% (Train : Val) - 2-way split
2. 80/20 Split: 80% : 20% (Train : Val) - 2-way split
3. 70/15/15 Split: 70% : 15% : 15% (Train : Val : Test) - 3-way split
4. 60/20/20 Split: 60% : 20% : 20% (Train : Val : Test) - 3-way split

Visualizations include:
- Pie charts for each split scenario
- Comparison bar charts
- Data distribution tables
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)
sns.set_style("whitegrid")

__all__ = [
    'plot_single_split',
    'plot_all_splits_comparison',
    'plot_split_distribution_table',
    'DatasetSplitVisualizer',
]


# Color palettes for consistency
COLORS_2WAY = ['#3498db', '#e74c3c']  # Blue, Red
COLORS_3WAY = ['#3498db', '#e74c3c', '#2ecc71']  # Blue, Red, Green


def plot_single_split(
    split_name: str,
    percentages: Dict[str, float],
    total_samples: int = 1000,
    output_dir: str = 'runs/detect/metrics',
    dpi: int = 300
) -> None:
    """
    Plot a single dataset split as a pie chart.
    
    Args:
        split_name: Name of the split (e.g., "82.7/17.3")
        percentages: Dict with split names as keys and percentages as values
                    e.g., {'Train': 82.7, 'Val': 17.3} or {'Train': 70, 'Val': 15, 'Test': 15}
        total_samples: Total number of samples for reference
        output_dir: Directory to save chart
        dpi: DPI for output image
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Calculate actual sample counts
    split_dict = {k: int(total_samples * v / 100) for k, v in percentages.items()}
    
    # Determine colors based on split type
    if len(percentages) == 2:
        colors = COLORS_2WAY
    else:
        colors = COLORS_3WAY
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create pie chart
    wedges, texts, autotexts = ax.pie(
        percentages.values(),
        labels=percentages.keys(),
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        textprops={'fontsize': 12, 'weight': 'bold'},
        explode=[0.05] * len(percentages)
    )
    
    # Enhance text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(13)
        autotext.set_weight('bold')
    
    for text in texts:
        text.set_fontsize(13)
        text.set_weight('bold')
    
    # Add title
    ax.set_title(f'Dataset Split: {split_name}', fontsize=16, weight='bold', pad=20)
    
    # Add sample counts as text box
    textstr = '\n'.join([f'{k}: {v:,} samples' for k, v in split_dict.items()])
    textstr += f'\n\nTotal: {total_samples:,} samples'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(1.3, 0.5, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='center', bbox=props, family='monospace')
    
    plt.tight_layout()
    
    # Sanitize filename
    filename = f"split_{split_name.replace('/', '_').replace(' ', '')}.png"
    output_path = Path(output_dir) / filename
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    logger.info(f"✓ Split chart '{split_name}' saved to {output_path}")
    plt.close()


def plot_all_splits_comparison(
    splits_data: Dict[str, Dict[str, float]],
    total_samples: int = 1000,
    output_dir: str = 'runs/detect/metrics',
    dpi: int = 300
) -> None:
    """
    Plot comparison of all dataset splits side by side.
    
    Args:
        splits_data: Dict of splits, e.g., {
            '82.7/17.3': {'Train': 82.7, 'Val': 17.3},
            '80/20': {'Train': 80, 'Val': 20},
            '70/15/15': {'Train': 70, 'Val': 15, 'Test': 15},
            '60/20/20': {'Train': 60, 'Val': 20, 'Test': 20}
        }
        total_samples: Total number of samples
        output_dir: Directory to save chart
        dpi: DPI for output image
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    n_splits = len(splits_data)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    fig.suptitle('Dataset Split Scenarios Comparison', fontsize=18, weight='bold', y=0.995)
    
    for idx, (split_name, percentages) in enumerate(splits_data.items()):
        ax = axes[idx]
        
        # Determine colors based on split type
        if len(percentages) == 2:
            colors = COLORS_2WAY
        else:
            colors = COLORS_3WAY
        
        # Calculate sample counts
        split_dict = {k: int(total_samples * v / 100) for k, v in percentages.items()}
        
        # Create pie chart
        wedges, texts, autotexts = ax.pie(
            percentages.values(),
            labels=percentages.keys(),
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            textprops={'fontsize': 10, 'weight': 'bold'},
            explode=[0.03] * len(percentages)
        )
        
        # Enhance text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(11)
            autotext.set_weight('bold')
        
        for text in texts:
            text.set_fontsize(11)
            text.set_weight('bold')
        
        # Add title with sample info
        title_text = f'{split_name}\n'
        title_text += ' : '.join([f'{k}={v:,}' for k, v in split_dict.items()])
        ax.set_title(title_text, fontsize=12, weight='bold', pad=15)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'splits_comparison_pie.png'
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    logger.info(f"✓ Splits comparison (pie charts) saved to {output_path}")
    plt.close()


def plot_split_distribution_table(
    splits_data: Dict[str, Dict[str, float]],
    total_samples: int = 1000,
    output_dir: str = 'runs/detect/metrics',
    dpi: int = 300
) -> None:
    """
    Plot a comparison table of all dataset splits.
    
    Args:
        splits_data: Dict of splits
        total_samples: Total number of samples
        output_dir: Directory to save chart
        dpi: DPI for output image
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Prepare data for table
    rows = []
    columns = ['Split Scenario']
    
    # Get all unique dataset parts
    all_parts = set()
    for percentages in splits_data.values():
        all_parts.update(percentages.keys())
    all_parts = sorted(list(all_parts))
    columns.extend(all_parts)
    columns.append('Total Samples')
    
    # Fill rows
    for split_name, percentages in splits_data.items():
        row = [split_name]
        for part in all_parts:
            if part in percentages:
                pct = percentages[part]
                samples = int(total_samples * pct / 100)
                row.append(f'{pct:.1f}% ({samples:,})')
            else:
                row.append('-')
        row.append(f'{total_samples:,}')
        rows.append(row)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc='center',
        loc='center',
        colWidths=[0.15, 0.15, 0.15, 0.15, 0.15, 0.15]
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Color header
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(rows) + 1):
        for j in range(len(columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
            else:
                table[(i, j)].set_facecolor('#ffffff')
    
    plt.title('Dataset Split Scenarios Comparison Table', fontsize=14, weight='bold', pad=20)
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'splits_comparison_table.png'
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    logger.info(f"✓ Splits comparison (table) saved to {output_path}")
    plt.close()


def plot_split_bar_chart(
    splits_data: Dict[str, Dict[str, float]],
    total_samples: int = 1000,
    output_dir: str = 'runs/detect/metrics',
    dpi: int = 300
) -> None:
    """
    Plot a bar chart comparison of all dataset splits.
    
    Args:
        splits_data: Dict of splits
        total_samples: Total number of samples
        output_dir: Directory to save chart
        dpi: DPI for output image
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Prepare data for bar chart
    split_names = list(splits_data.keys())
    
    # Get all unique dataset parts
    all_parts = set()
    for percentages in splits_data.values():
        all_parts.update(percentages.keys())
    all_parts = sorted(list(all_parts))
    
    # Prepare data
    x = np.arange(len(split_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Create bars for each part
    for idx, part in enumerate(all_parts):
        values = []
        for split_name in split_names:
            percentages = splits_data[split_name]
            values.append(percentages.get(part, 0))
        
        color = [COLORS_2WAY, COLORS_3WAY][len(all_parts) - 2][idx]
        bars = ax.bar(x + idx * width, values, width, label=part, color=color)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=9, weight='bold')
    
    ax.set_xlabel('Split Scenario', fontsize=12, weight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=12, weight='bold')
    ax.set_title('Dataset Split Scenarios - Bar Chart Comparison', fontsize=14, weight='bold')
    ax.set_xticks(x + width * (len(all_parts) - 1) / 2)
    ax.set_xticklabels(split_names, fontsize=11, weight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 100])
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'splits_comparison_bars.png'
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    logger.info(f"✓ Splits comparison (bar chart) saved to {output_path}")
    plt.close()


class DatasetSplitVisualizer:
    """Manager class for dataset split visualizations."""
    
    def __init__(self, output_dir: str = 'runs/detect/metrics', total_samples: int = 1000):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save visualizations
            total_samples: Total number of samples for reference
        """
        self.output_dir = output_dir
        self.total_samples = total_samples
        
        # Define all split scenarios
        self.splits_data = {
            '82.7/17.3 (Current)': {'Train': 82.7, 'Val': 17.3},
            '80/20': {'Train': 80, 'Val': 20},
            '70/15/15': {'Train': 70, 'Val': 15, 'Test': 15},
            '60/20/20': {'Train': 60, 'Val': 20, 'Test': 20},
        }
    
    def generate_all_visualizations(self) -> None:
        """Generate all split visualizations."""
        logger.info("=" * 70)
        logger.info("DATASET SPLIT COMPARISON VISUALIZATIONS")
        logger.info("=" * 70)
        logger.info(f"\nTotal Samples: {self.total_samples:,}")
        logger.info(f"Output Directory: {self.output_dir}\n")
        
        # Generate individual pie charts
        logger.info("Generating individual split pie charts...")
        for split_name, percentages in self.splits_data.items():
            plot_single_split(
                split_name,
                percentages,
                self.total_samples,
                self.output_dir
            )
        
        # Generate comparison visualizations
        logger.info("\nGenerating comparison visualizations...")
        plot_all_splits_comparison(
            self.splits_data,
            self.total_samples,
            self.output_dir
        )
        
        plot_split_distribution_table(
            self.splits_data,
            self.total_samples,
            self.output_dir
        )
        
        plot_split_bar_chart(
            self.splits_data,
            self.total_samples,
            self.output_dir
        )
        
        # Generate summary
        self._print_summary()
    
    def _print_summary(self) -> None:
        """Print summary of all splits."""
        logger.info("\n" + "=" * 70)
        logger.info("DATASET SPLIT SUMMARY")
        logger.info("=" * 70 + "\n")
        
        for split_name, percentages in self.splits_data.items():
            logger.info(f"📊 {split_name}:")
            for part, pct in percentages.items():
                samples = int(self.total_samples * pct / 100)
                logger.info(f"   {part:6} : {pct:5.1f}% ({samples:,} samples)")
            logger.info("")


if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create visualizer with default total samples
    visualizer = DatasetSplitVisualizer(
        output_dir='runs/detect/metrics',
        total_samples=1000
    )
    
    # Generate all visualizations
    visualizer.generate_all_visualizations()
