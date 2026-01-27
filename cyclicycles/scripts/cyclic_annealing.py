#!/usr/bin/env python3
"""
Script for running cyclic quantum annealing on dynamics instances.
Executes exactly one cyclic annealing run and exits.
"""

import argparse
import sys
from pathlib import Path
from cyclicycles.runner import Runner

def main():
    parser = argparse.ArgumentParser(description='Run cyclic quantum annealing on a single dynamics instance')
    parser.add_argument('--sampler', type=str, default='6.4',
                       choices=['1.6', '4.1', '6.4','1.10'],
                       help='D-Wave sampler to use (1.6=Advantage2, 4.1/6.4=Advantage)')
    parser.add_argument('--instance-id', type=str, required=True,
                       help='Dynamics instance ID (e.g., 1, 2, 3, etc.)')
    parser.add_argument('--num-timepoints', type=int, required=True,
                       help='Number of timepoints in the dynamics instance')
    parser.add_argument('--num-reads', type=int, default=1000,
                       help='Number of annealing reads per cycle (default: 1000)')
    parser.add_argument('--num-cycles', type=int, default=5,
                       help='Number of annealing cycles (default: 5)')
    parser.add_argument('--use-ancilla', action='store_true',
                       help='Use ancilla transformation')
    
    args = parser.parse_args()
    
    try:
        # Initialize runner with specified sampler
        runner = Runner(sampler=args.sampler)
        
        print(f"Running cyclic annealing with parameters:")
        print(f"  Instance ID: {args.instance_id}")
        print(f"  Timepoints: {args.num_timepoints}")
        print(f"  Reads per cycle: {args.num_reads}")
        print(f"  Cycles: {args.num_cycles}")
        print(f"  Sampler: {args.sampler}")
        print(f"  Use ancilla: {args.use_ancilla}")
        print("\nStarting annealing process...")
        
        # Run cyclic annealing on dynamics instance
        response, results, cycle_energies = runner.execute_cyclic_annealing(
            num_reads=args.num_reads,
            num_cycles=args.num_cycles,
            use_forward_init=True,
            instance_type='dynamics',
            instance_id=args.instance_id,
            num_timepoints=args.num_timepoints,
            use_ancilla_transformation=args.use_ancilla
        )
        
        print("\nCyclic annealing completed!")
        print(f"\nEnergy progression across cycles: {cycle_energies}")
        print(f"Best energy found: {results['best_energy']:.6f}")
        
        if 'save_path' in results:
            print(f"Results saved to: {results['save_path']}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {type(e).__name__}: {str(e)}", file=sys.stderr)
        return 1

if __name__ == '__main__':
    sys.exit(main())