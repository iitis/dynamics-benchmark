#!/usr/bin/env python3
"""
This script plots the success probability by annealing time (ta) for quantum systems.

USAGE:
    python plot_success_prob_by_ta.py [OPTIONS]

    Options:
        --system INT                   System ID to plot (1-8) (default: 1)
        --timepoints LIST             List of timepoints to consider (default: [2,3])

    LIST format: comma-separated values, e.g., "2,3,4"

    Examples:
        python plot_success_prob_by_ta.py --system 1
        python plot_success_prob_by_ta.py --system 2
        python plot_success_prob_by_ta.py --system 3 --timepoints 2,3,4
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
        description="Plot success probability by annealing time for quantum systems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --system 1
  %(prog)s --system 2 --annealing-times 10,200,500
  %(prog)s --system 3 --timepoints 2,3,4
        """
    )
    
    parser.add_argument(
        '--system',
        type=int,
        default=1,
        help='System ID to plot (1-8). Default: 1'
    )
    
    parser.add_argument(
        '--timepoints',
        type=parse_list,
        default='2,3',
        help='List of timepoints to consider. Default: [2,3]'
    )
    
    return parser

if __name__ == "__main__":
    # Parse command line arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Create plotter and generate plot
    plotter_instance = plotter.BenchmarkPlotter()
    plotter_instance.plot_success_prob_by_ta(
        system=args.system,
        timepoints_of_interest=args.timepoints
    )
