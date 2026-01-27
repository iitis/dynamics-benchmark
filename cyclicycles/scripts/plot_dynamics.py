#!/usr/bin/env python3
"""
Script for plotting dynamics annealing results.
"""

import argparse
from pathlib import Path
from cyclicycles.plotter import Plotter
import sys
from pathlib import Path

# Add the src directory to Python path
src_dir = Path(__file__).resolve().parent.parent / 'src'
sys.path.insert(0, str(src_dir))

from cyclicycles.config import RESULT_DIR

def main():
    parser = argparse.ArgumentParser(description='Plot dynamics annealing results')
    parser.add_argument('--solver', type=str, default='6.4',
                       choices=['1.6', '1.7','1.8','1.9',"1.10", '4.1', '6.4'],
                       help='D-Wave solver results to plot')
    parser.add_argument('--instance_id', type=str, required=True,
                       help='ID of the dynamics instance to plot')
    parser.add_argument('--num_timepoints', type=int, required=True,
                       help='Number of timepoints for the dynamics instance')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='Directory to save plot. If not specified, displays plot.')
    parser.add_argument('--num_samples', type=int, choices=[100, 1000], default=1000,
                       help='Only include results with this exact number of samples.')
    parser.add_argument('--init', type=str, choices=['forward', 'zero', 'all'], default='all',
                       help='Which initialization method to show: forward (forward annealing initialized), '
                            'zero (standard zero initialization), or all (both)')
    parser.add_argument('--ancilla', action='store_true',
                       help='Plot results with ancilla transformation (default: without ancilla)')
    
    args = parser.parse_args()
    
    # Initialize plotter
    plotter = Plotter(RESULT_DIR)
    
    # Create save directory if specified
    save_dir = None
    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot dynamics instance
    plotter.plot_instance(
        solver_id=args.solver,
        instance_type='dynamics',
        instance_id=args.instance_id,
        num_timepoints=args.num_timepoints,
        save_dir=save_dir,
        num_samples=args.num_samples,
        init_type=args.init,
        use_ancilla=args.ancilla
    )

if __name__ == '__main__':
    main()
