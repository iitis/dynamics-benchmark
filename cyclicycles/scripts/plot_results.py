#!/usr/bin/env python3
"""
Script for plotting annealing results.
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
    parser = argparse.ArgumentParser(description='Plot annealing results')
    parser.add_argument('--solver', type=str, default='6.4',
                       choices=['1.6','1.7', '1.8','4.1', '6.4'],
                       help='D-Wave solver results to plot')
    parser.add_argument('--n_nodes', type=int, default=None,
                       help='Specific instance to plot. If not specified, plots all instances.')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='Directory to save plots. If not specified, displays plots.')
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
    
    # Plot results
    if args.n_nodes is not None:
        plotter.plot_instance(args.solver, args.n_nodes, save_dir, 
                            num_samples=args.num_samples, init_type=args.init,
                            use_ancilla=args.ancilla)
    else:
        # Find all instance directories
        import re
        ancilla_suffix = '_with_ancilla' if args.ancilla else '_no_ancilla'
        pattern = f"{args.solver}/N_*{ancilla_suffix}_realization_1"
        
        instance_dirs = list(plotter.result_dir.glob(pattern))
        
        # Extract node counts
        node_counts = set()
        for d in instance_dirs:
            match = re.search(r'N_(\d+)_', d.name)
            if match:
                node_counts.add(int(match.group(1)))
        
        for n_nodes in sorted(node_counts):
            plotter.plot_instance(args.solver, n_nodes, save_dir,
                                num_samples=args.num_samples, init_type=args.init,
                                use_ancilla=args.ancilla)

if __name__ == '__main__':
    main()