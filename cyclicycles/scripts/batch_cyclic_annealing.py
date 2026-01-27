#!/usr/bin/env python3
"""
Batch runner for cyclic annealing on dynamics instances.

Runs cyclic annealing across:
- All instances (1-8)
- Multiple timepoints: 2, 6, 7, 8, 9, 10, 11, 12

Each run executes exactly one cyclic annealing sample with automatic retry logic.

Usage:
    python scripts/batch_cyclic_annealing.py
    python scripts/batch_cyclic_annealing.py --num-reads 1000 --num-cycles 30
"""

import sys
import os
import argparse
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cyclicycles.runner import Runner
from tqdm import tqdm


def run_cyclic_sample(instance_id, num_timepoints, num_reads=1000, num_cycles=30, max_retries=3):
    """
    Run a single cyclic annealing sample.
    
    Args:
        instance_id: Dynamic instance ID (1-8)
        num_timepoints: Number of timepoints
        num_reads: Number of annealing reads per cycle
        num_cycles: Number of annealing cycles
        max_retries: Maximum number of retries on error
        
    Returns:
        (success, results) tuple
    """
    runner = Runner()
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            print(f"    Running cyclic annealing (instance={instance_id}, "
                  f"timepoints={num_timepoints}, reads={num_reads}, cycles={num_cycles})...", 
                  end=" ", flush=True)
            
            response, results, cycle_energies = runner.execute_cyclic_annealing(
                num_reads=num_reads,
                num_cycles=num_cycles,
                use_forward_init=True,
                instance_type='dynamics',
                instance_id=str(instance_id),
                num_timepoints=num_timepoints,
                use_ancilla_transformation=False
            )
            
            print(f"✓ (best energy: {results['best_energy']:.4f})")
            return True, results
            
        except Exception as e:
            retry_count += 1
            print(f"✗ Error (attempt {retry_count}/{max_retries})")
            print(f"      {type(e).__name__}: {str(e)[:100]}")
            
            if retry_count < max_retries:
                wait_time = 5 * retry_count  # Exponential backoff: 5s, 10s, 15s
                print(f"      Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"      Max retries reached, skipping this configuration")
    
    return False, None


def main():
    parser = argparse.ArgumentParser(
        description='Batch runner for cyclic annealing on dynamics instances'
    )
    parser.add_argument('--num-reads', type=int, default=1000,
                       help='Number of annealing reads per cycle (default: 1000)')
    parser.add_argument('--num-cycles', type=int, default=30,
                       help='Number of annealing cycles (default: 30)')
    parser.add_argument('--instances', type=int, nargs='+', default=list(range(1, 9)),
                       help='Instance IDs to run (default: 1-8)')
    parser.add_argument('--timepoints', type=int, nargs='+', 
                       default=[2, 6, 7, 8, 9, 10, 11, 12],
                       help='Timepoints to run (default: 2, 6, 7, 8, 9, 10, 11, 12)')
    
    args = parser.parse_args()
    
    print(f"Batch Cyclic Annealing Runner")
    print(f"=" * 70)
    print(f"Parameters:")
    print(f"  - Instances: {sorted(args.instances)}")
    print(f"  - Timepoints: {sorted(args.timepoints)}")
    print(f"  - Reads per cycle: {args.num_reads}")
    print(f"  - Cycles: {args.num_cycles}")
    print(f"  - Max retries: 3 (exponential backoff: 5s, 10s, 15s)")
    print(f"=" * 70)
    
    total_configs = len(args.instances) * len(args.timepoints)
    print(f"\nTotal configurations to run: {total_configs}\n")
    
    # Track results
    successful = 0
    failed = 0
    
    # Iterate through all timepoints first, then instances
    for num_timepoints in sorted(args.timepoints):
        print(f"\n[Timepoints {num_timepoints}]")
        print("-" * 70)
        
        for instance_id in sorted(args.instances):
            success, results = run_cyclic_sample(
                instance_id=instance_id,
                num_timepoints=num_timepoints,
                num_reads=args.num_reads,
                num_cycles=args.num_cycles
            )
            
            if success:
                successful += 1
            else:
                failed += 1
            
            # Small delay between runs
            if instance_id < max(args.instances):
                time.sleep(1)
    
    # Summary
    print(f"\n" + "=" * 70)
    print(f"SUMMARY")
    print(f"=" * 70)
    print(f"Successful runs: {successful}/{total_configs}")
    print(f"Failed runs: {failed}/{total_configs}")
    print(f"Success rate: {100*successful/total_configs:.1f}%")
    print(f"=" * 70)


if __name__ == '__main__':
    main()
