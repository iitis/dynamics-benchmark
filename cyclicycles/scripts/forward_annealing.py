#!/usr/bin/env python3
"""
Script for running forward quantum annealing on problem instances.
"""

import argparse
from pathlib import Path
from cyclicycles.runner import Runner

def main():
    parser = argparse.ArgumentParser(description='Run forward quantum annealing')
    parser.add_argument('--n_nodes', type=str, default=None,
                       help='Number of nodes in the instance to run. If not specified, uses first available.')
    parser.add_argument('--num_reads', type=int, default=1000,
                       help='Number of annealing reads to perform')
    parser.add_argument('--sampler', type=str, default='6.4',
                       choices=['1.6', '4.1', '6.4'],
                       help='D-Wave sampler to use (1.6=Advantage2, 4.1/6.4=Advantage)')
    
    args = parser.parse_args()
    
    # Initialize runner with specified sampler
    runner = Runner(sampler=args.sampler)
    
    print(f"Running forward annealing with parameters:")
    print(f"Number of nodes: {args.n_nodes if args.n_nodes else 'First available'}")
    print(f"Number of reads: {args.num_reads}")
    print(f"Sampler: {args.sampler}")
    print("\nStarting annealing process...")
    
    # Run forward annealing
    response, results = runner.execute_instance(
        n_nodes=args.n_nodes,
        num_reads=args.num_reads
    )
    
    print("\nAnnealing completed!")
    print(f"Minimum energy found: {min(response.record.energy):.6f}")
    print(f"Results saved to: {results['save_path'] if 'save_path' in results else 'results directory'}")

if __name__ == '__main__':
    main()