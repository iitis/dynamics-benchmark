#!/usr/bin/env python3
"""
This script plots time-to-solution (TTS) comparisons for quantum systems across different solvers.

USAGE:
    python plot_tta_comparison.py [OPTIONS]

    Options:
        --systems LIST        List of system IDs to plot (default: [1,2,5,6,7])
        --file-limit INT     Maximum number of files to process per system (default: 5)

    LIST format: comma-separated values, e.g., "1,2,5"

    Examples:
        python plot_tta_comparison.py
        python plot_tta_comparison.py --file-limit 10
        python plot_tta_comparison.py --systems 1,2,3 
"""

import argparse
from benchmarker.core import results_loader
from benchmarker.core import plotter

def parse_list(value: str) -> list:
    """Parse a comma-separated string into a list of integers."""
    return [int(x.strip()) for x in value.split(',')]

def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Plot time-to-solution comparisons for quantum systems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s
  %(prog)s --file-limit 10
  %(prog)s --systems 1,2,3 
        """
    )
    
    parser.add_argument(
        '--systems',
        type=parse_list,
        default='1,2,5,6,7',
        help='List of system IDs to plot. Default: [1,2,5,6,7]'
    )
    
    parser.add_argument(
        '--file-limit',
        type=int,
        default=5,
        help='Maximum number of files to process per system. Default: 5'
    )
    
    return parser

if __name__ == "__main__":
    # Parse command line arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Create plotter and generate plot
    plotter_instance = plotter.BenchmarkPlotter()
    plotter_instance.plot_tts(
        systems=args.systems,
        file_limit=args.file_limit,
    )
