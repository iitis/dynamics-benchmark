#!/usr/bin/env python3
"""
This script plots the dynamics of quantum systems using different solvers.

USAGE:
    python plot_dynamics.py [OPTIONS]

    Options:
        --system INT        System ID to plot (1-8)
        --timepoints INT    Number of time points in the simulation (default: 3)
        --solver STR       Solver to use: "6.4", "1.4", "neal", or "VELOX" (default: "1.4")

    Examples:
        python plot_dynamics.py --system 1 --timepoints 3
        python plot_dynamics.py --system 2 --timepoints 5 --solver "6.4"
        python plot_dynamics.py --system 3 --timepoints 4 --solver "neal"
"""

import argparse
from benchmarker.core import results_loader
from benchmarker.core import plotter

def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Plot dynamics for quantum systems using different solvers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --system 1 --timepoints 3
  %(prog)s --system 2 --timepoints 5 --solver "6.4"
  %(prog)s --system 3 --timepoints 4 --solver "neal"
        """
    )
    
    parser.add_argument(
        '--system',
        type=int,
        required=True,
        help='System ID to plot (1-8)'
    )
    
    parser.add_argument(
        '--timepoints',
        type=int,
        default=3,
        help='Number of time points in the simulation. Default: 3'
    )
    
    parser.add_argument(
        '--solver',
        type=str,
        default='1.4',
        choices=['6.4', '1.4', 'neal', 'VELOX'],
        help='Solver to use for the simulation. Default: "1.4"'
    )
    
    return parser

if __name__ == "__main__":
    # Parse command line arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Create plotter and generate plot
    plotter_instance = plotter.BenchmarkPlotter()
    plotter_instance.plot_dynamics(
        system=args.system,
        timepoints=args.timepoints,
        solver=args.solver
    )










