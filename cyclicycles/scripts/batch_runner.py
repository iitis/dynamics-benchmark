"""
Batch runner script for cyclic annealing experiments.

Runs cyclic annealing and forward annealing for multiple instances and timepoints.
Automatically retries on errors and generates multiple samples per configuration.

Usage:
    python scripts/batch_runner.py
    python scripts/batch_runner.py --instances 1 2 3 --timepoints 3 4 --samples 5
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


def run_sample(runner, instance_id, num_timepoints, sample_num, max_retries=3):
    """
    Run a single sample pair (cyclic + forward annealing).
    
    Args:
        runner: Runner instance
        instance_id: Dynamic instance ID (1-8)
        num_timepoints: Number of timepoints (3-5)
        sample_num: Sample number (for logging)
        max_retries: Maximum number of retries on error
        
    Returns:
        (success, cycle_results, forward_results) tuple
    """
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            print(f"\n  Sample {sample_num}/{max_retries}: Instance {instance_id}, "
                  f"Timepoints {num_timepoints}")
            
            # Run cyclic annealing
            print("    Running cyclic annealing...", end=" ", flush=True)
            response_cyclic, results_cyclic, cycle_energies = runner.execute_cyclic_annealing(
                num_reads=1000,
                num_cycles=30,
                use_forward_init=True,
                instance_type='dynamics',
                instance_id=str(instance_id),
                num_timepoints=num_timepoints,
                use_ancilla_transformation=False
            )
            print(f"✓ (best energy: {results_cyclic['best_energy']:.4f})")
            
            # Small delay between requests
            time.sleep(2)
            
            # Run forward annealing
            print("    Running forward annealing...", end=" ", flush=True)
            response_forward, results_forward = runner.execute_instance(
                instance_type="dynamics",
                instance_id=str(instance_id),
                num_reads=1000,
                num_timepoints=num_timepoints,
                use_ancilla_transformation=False
            )
            print(f"✓ (best energy: {results_forward['energies'][0]:.4f})")
            
            return True, results_cyclic, results_forward
            
        except Exception as e:
            retry_count += 1
            print(f"✗ Error (attempt {retry_count}/{max_retries})")
            print(f"      {type(e).__name__}: {str(e)[:100]}")
            
            if retry_count < max_retries:
                wait_time = 5 * retry_count  # Exponential backoff
                print(f"      Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"      Max retries reached, skipping this configuration")
    
    return False, None, None


def main():
    parser = argparse.ArgumentParser(
        description='Batch runner for cyclic annealing experiments'
    )
    parser.add_argument('--instances', type=int, nargs='+', default=list([8]),
                       help='Instance IDs to run (default: 1-8)')
    parser.add_argument('--timepoints', type=int, nargs='+', default=[3, 4, 5],
                       help='Timepoints to run (default: 3, 4, 5)')
    parser.add_argument('--samples', type=int, default=3,
                       help='Number of samples per configuration (default: 3)')
    parser.add_argument('--solver', default='1.10',
                       help='Solver version (default: 1.10)')
    
    args = parser.parse_args()
    
    instances = sorted(set(args.instances))
    timepoints = sorted(set(args.timepoints))
    num_samples = args.samples
    
    print("=" * 70)
    print("BATCH RUNNER: Cyclic Annealing Experiments")
    print("=" * 70)
    print(f"Solver: {args.solver}")
    print(f"Instances: {instances}")
    print(f"Timepoints: {timepoints}")
    print(f"Samples per config: {num_samples}")
    print(f"Total runs: {len(instances) * len(timepoints) * num_samples}")
    print("=" * 70)
    
    # Initialize runner
    try:
        print("\nInitializing D-Wave runner...")
        runner = Runner(sampler=args.solver)
        print("✓ Runner initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize runner: {e}")
        return False
    
    # Run experiments
    total_configs = len(instances) * len(timepoints)
    completed_configs = 0
    failed_configs = 0
    
    for instance_id in instances:
        for num_timepoints in timepoints:
            config_name = f"Instance {instance_id}, Timepoints {num_timepoints}"
            print(f"\n[{completed_configs + 1}/{total_configs}] {config_name}")
            
            config_success = False
            for sample_num in range(1, num_samples + 1):
                success, cyclic_results, forward_results = run_sample(
                    runner, instance_id, num_timepoints, sample_num
                )
                if success:
                    config_success = True
                else:
                    print(f"    ✗ Sample {sample_num} failed after max retries")
            
            if config_success:
                print(f"  ✓ Configuration complete")
                completed_configs += 1
            else:
                print(f"  ✗ Configuration failed")
                failed_configs += 1
    
    # Summary
    print("\n" + "=" * 70)
    print("BATCH RUN COMPLETE")
    print("=" * 70)
    print(f"Completed configurations: {completed_configs}/{total_configs}")
    print(f"Failed configurations: {failed_configs}/{total_configs}")
    print(f"Success rate: {100 * completed_configs / total_configs:.1f}%")
    print("=" * 70)
    
    return completed_configs == total_configs


if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nBatch run interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
