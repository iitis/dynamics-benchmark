#!/usr/bin/env python3
"""

This script runs VeloxQ solver on QUBODynamics Hessian problem instances across multiple configurations
and saves the results to CSV files. It uses the VeloxQ Python SDK.

USAGE:
    python run_velox.py [OPTIONS]

    Options:
        --num-reps RANGE        Number of repetitions to test (default: [1024])
        --num-steps RANGE       Number of steps to test (default: 10-100:10,200-1000:100,2000-10000:1000)
        --instance-types RANGE  Instance types to test (default: [1,2,3,4,5,6,7,8])
        --num-shots INT         Number of shots per configuration (default: 50)
        --max-instances INT     Maximum number of instances per type to process, sorted by timepoints (default: None - process all)
        --output-prefix STR     Prefix for output CSV files (default: "new5")

    RANGE format: "start-end:step" or "val1,val2,val3"

    Examples:
        python run_velox.py --num-shots 100 --instance-types 1,2,3
        python run_velox.py --num-reps 512,1024 --num-steps 10-100:10
        python run_velox.py --output-prefix custom --max-instances 5
        python run_velox.py --max-instances 5 --instance-types 1,2 --num-shots 10

REQUIREMENTS:
    - VeloxQ SDK installed and configured
    - Problem instance files in the expected directory structure
    - Valid VeloxQ API credentials

CONFIGURATION:
    - The script expects problem instance files in: 
      ../../benchmarker/data/instances/hessian/{INSTANCE_TYPE}/
    - Results are saved to: results/hessian/{INSTANCE_TYPE}/

"""

import os
import re
import csv
import time
import argparse
from pathlib import Path
from typing import List, Optional 
import numpy as np
from tqdm import tqdm

# VeloxQ SDK imports
from veloxq_sdk import VeloxQSolver, VeloxQParameters
from veloxq_sdk.config import VeloxQAPIConfig


def parse_e0_from_file(filepath: str) -> Optional[float]:
    """Parse the E0 (ground state energy) value from the first line of a problem file.
    
    Args:
        filepath: Path to the problem file
        
    Returns:
        The E0 value if found, None otherwise
    """
    try:
        with open(filepath, 'r') as f:
            first_line = f.readline().strip()
            # Look for pattern like "# E0 = -123.456"
            match = re.search(r'# E0 = (-?\d+\.\d+)', first_line)
            if match:
                return float(match.group(1))
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
    return None


def calculate_gap(ground_truth: float, energy: float) -> float:
    """Calculate the relative gap between the found energy and ground truth.
    
    Args:
        ground_truth: Ground state energy
        energy: Found energy
        
    Returns:
        Relative gap as percentage
    """
    return (energy - ground_truth) / abs(ground_truth) * 100


def get_sorted_instances(path: str, max_instances: Optional[int] = None) -> List[str]:
    """Get sorted list of problem instance files from a directory, sorted by timepoints.
    
    Args:
        path: Directory path containing problem instances
        max_instances: Maximum number of instances to return (None for no limit)
        
    Returns:
        Sorted list of instance filenames, limited to max_instances if specified
    """
    if not os.path.exists(path):
        print(f"Warning: Path {path} does not exist")
        return []
    
    # Get all files in directory, excluding .json files
    instances = [f for f in os.listdir(path) if not f.endswith('.json')]
    
    # Sort by timepoints (extract from filename pattern: precision_X_timepoints_Y_ising.coo)
    def sort_key(filename: str) -> int:
        # Extract timepoints value using regex
        match = re.search(r'timepoints_(\d+)', filename)
        if match:
            return int(match.group(1))
        return 0
    
    instances.sort(key=sort_key)
    
    # Limit number of instances if specified
    if max_instances is not None and max_instances > 0:
        instances = instances[:max_instances]
        
    return instances


def get_problem_size(result) -> int:
    """Try to determine the problem size from the result.
    
    Args:
        result: VeloxSampleSet containing the results
        
    Returns:
        Number of variables in the problem
    """
    try:
        # Try to get the size from the sample shape
        if hasattr(result, 'sample') and result.sample.size > 0:
            if len(result.sample.shape) > 1:
                return result.sample.shape[1]  # Number of variables
            else:
                return len(result.sample)
    except Exception:
        pass
    return 0  # Default if size cannot be determined


def run_veloxq_benchmark(num_reps: Optional[List[int]] = None, num_steps: Optional[List[int]] = None, 
                        instance_types: Optional[List[int]] = None, 
                        num_shots: int = 50, output_prefix: str = "new5", max_instances: Optional[int] = None):
    """Main function to run the VeloxQ benchmark across multiple configurations.
    
    Args:
        num_reps: List of number of repetitions to test
        num_steps: List of number of steps to test
        instance_types: List of instance types to test
        num_shots: Number of shots per configuration
        output_prefix: Prefix for output CSV files
        max_instances: Maximum number of instances per type (None for no limit)
    """
    
    # Configuration parameters - use provided parameters or defaults
    NUM_REPS = num_reps if num_reps is not None else [2**10]  # [1024]
    NUM_STEPS = num_steps if num_steps is not None else (list(range(10, 101, 10)) + list(range(200, 1001, 100)) + list(range(2000, 10001, 1000)))
    INSTANCE_TYPES = instance_types if instance_types is not None else [1,2,3,4,5,6,7,8]
    NUM_SHOTS = num_shots
    
    # Set up base paths
    script_dir = Path(__file__).parent
    data_base_path = script_dir / "../../benchmarker/data/instances/hessian"
    results_base_path = script_dir / "results/hessian"
    
    print(f"VeloxQ Benchmark Script")
    print(f"Configurations: {len(NUM_REPS)} x {len(NUM_STEPS)} = {len(NUM_REPS) * len(NUM_STEPS)}")
    print(f"Instance types: {INSTANCE_TYPES}")
    print(f"Shots per configuration: {NUM_SHOTS}")
    print()
    print("NOTE: This script expects problem instance files in:")
    print(f"      {data_base_path}")
    print("      Adjust the path if your problem files are located elsewhere.")
    print("=" * 80)
    
    # Calculate total configurations for progress tracking
    all_configs = []
    for instance_type in INSTANCE_TYPES:
        instance_path = data_base_path / str(instance_type)
        instances = get_sorted_instances(str(instance_path), max_instances)
        for num_rep_val in NUM_REPS:
            for num_steps_val in NUM_STEPS:
                for instance in instances:
                    all_configs.append((instance_type, instance, num_rep_val, num_steps_val))
    
    # Single progress bar for all configurations
    progress_bar = tqdm(all_configs, desc="Processing configurations")
    
    # Group configurations by instance type for processing
    current_instance_type = None
    current_results_dir = None
    current_filename_best = None
    write_header = True
    
    for instance_type, instance, num_rep_val, num_steps_val in progress_bar:
        # Set up paths when switching to new instance type
        if current_instance_type != instance_type:
            current_instance_type = instance_type
            instance_path = data_base_path / str(instance_type)
            current_results_dir = results_base_path / str(instance_type)
            current_results_dir.mkdir(parents=True, exist_ok=True)
            current_filename_best = current_results_dir / f"{output_prefix}_best_results_hessian_{instance_type}.csv"
            write_header = not current_filename_best.exists()
        
        # Update progress bar description
        progress_bar.set_description(f"Type {instance_type}: {instance} (rep={num_rep_val}, steps={num_steps_val})")
        
        instance_filepath = instance_path / instance
        
        # Parse E0 from file
        E0 = parse_e0_from_file(str(instance_filepath))
        if E0 is None:
            progress_bar.set_postfix_str("E0 not found - skipping")
            continue
        
        # Create VeloxQ solver with parameters
        params = VeloxQParameters(
            num_rep=num_rep_val,
            num_steps=num_steps_val
        )
        solver = VeloxQSolver(parameters=params)
        
        # Track results across shots
        total_runtime = 0.0
        computation_time = 0.0
        best_energy = float('inf')
        best_state = None
        num_success = 0
        num_var = 0  # Will be determined from first successful result
        
        # Run multiple shots
        for shot in range(NUM_SHOTS):
            try:
                # Load and solve the problem
                start_time = time.time()
                
                # Use the sample method which submits job and waits for completion
                result = solver.sample(
                    str(instance_filepath),
                    name=f"{instance}_shot_{shot}.h5"
                )
                
                runtime = time.time() - start_time
                total_runtime += runtime
                
                # Extract results
                energies = result.energy
                states = result.sample
                
                # Get problem size from first successful result
                if num_var == 0:
                    num_var = get_problem_size(result)
                
                # Get computation time from result info
                computation_time += result.info["total_time"]
                
                # Find best energy and state from this shot
                min_energy_idx = np.argmin(energies)
                min_energy = energies[min_energy_idx]
                
                # Extract the best state (first column as in Julia version)
                if len(states.shape) > 1:
                    state = ';'.join(map(str, states[min_energy_idx, :]))
                else:
                    state = ';'.join(map(str, states[min_energy_idx]))
                
                # Check if this shot found the ground state (within tolerance)
                if np.isclose(min_energy, E0, rtol=1e-6):
                    num_success += 1
                                
                # Update best solution if this is better
                if min_energy < best_energy:
                    best_energy = min_energy
                    best_state = state
                    
            except Exception as e:
                progress_bar.set_postfix_str(f"Shot error: {str(e)[:20]}")
                continue
        
        # Calculate final statistics
        success_prob = num_success / NUM_SHOTS if NUM_SHOTS > 0 else 0
        avg_runtime = total_runtime / NUM_SHOTS if NUM_SHOTS > 0 else 0
        avg_computation_time = computation_time / NUM_SHOTS if NUM_SHOTS > 0 else 0
        gap = calculate_gap(E0, best_energy)
        
        # Update progress bar with results
        # Update progress bar with results
        progress_bar.set_postfix_str(f"Gap: {gap:.2f}%, Success: {success_prob:.2f}")
        
        # Create result record
        datapoint = {
            'type': instance_type,
            'instance': instance,
            'num_var': num_var,
            'NUM_SHOTS': NUM_SHOTS,
            'num_rep': num_rep_val,
            'num_steps': num_steps_val,
            'runtime': avg_runtime,
            'gpu_runtime': avg_computation_time,
            'gap': gap,
            'best_energy': best_energy,
            'gnd_energy': E0,
            'success_prob': success_prob,
            'best_solution': best_state or ""
        }
        
        # Save to CSV
        fieldnames = list(datapoint.keys())
        with open(str(current_filename_best), 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
                write_header = False
            writer.writerow(datapoint)
    
    progress_bar.close()


def parse_range(value: str) -> List[int]:
    """Parse a range string like '10-100:10' or '10,20,30' into a list of integers.
    
    Args:
        value: Range string in format 'start-end:step' or comma-separated values
        
    Returns:
        List of integers
    """
    if '-' in value and ':' in value:
        # Range format: start-end:step
        parts = value.split(':')
        if len(parts) != 2:
            raise ValueError(f"Invalid range format: {value}. Use 'start-end:step'")
        
        range_part, step = parts
        start, end = range_part.split('-')
        return list(range(int(start), int(end) + 1, int(step)))
    else:
        # Comma-separated format: 10,20,30
        return [int(x.strip()) for x in value.split(',')]


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Run VeloxQ benchmark on Hessian problem instances",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --num-shots 100 --instance-types 1,2,3
  %(prog)s --num-reps 512,1024 --num-steps 10-100:10 --output-prefix custom
  %(prog)s --num-steps 10,20,50,100 --num-shots 25
  %(prog)s --max-instances 5 --instance-types 1,2 --num-shots 10
        """
    )
    
    parser.add_argument(
        '--num-reps',
        type=parse_range,
        default=None,
        help='Number of repetitions to test. Format: "start-end:step" or "val1,val2,val3". Default: [1024]'
    )
    
    parser.add_argument(
        '--num-steps', 
        type=parse_range,
        default=None,
        help='Number of steps to test. Format: "start-end:step" or "val1,val2,val3". Default: 10-100:10,200-1000:100,2000-10000:1000'
    )
    
    parser.add_argument(
        '--instance-types',
        type=parse_range,
        default='1,2,3,4,5,6,7,8',
        help='Instance types to test. Format: "start-end:step" or "val1,val2,val3". Default: [1,2,3,4,5,6,7,8]'
    )
    
    parser.add_argument(
        '--num-shots',
        type=int,
        default=50,
        help='Number of shots per configuration. Default: 50'
    )
    
    parser.add_argument(
        '--max-instances',
        type=int,
        default=None,
        help='Maximum number of instances per type to process (sorted by timepoints). Default: None (process all)'
    )
    
    parser.add_argument(
        '--output-prefix',
        type=str,
        default='new5',
        help='Prefix for output CSV files. Default: "new5"'
    )
    
    return parser


if __name__ == "__main__":
    # Parse command line arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Uncomment and set your API configuration if needed
    # api_config = VeloxQAPIConfig.instance()
    # api_config.token = "YOUR_VELOXQ_API_TOKEN_HERE"
    
    print("Starting VeloxQ benchmark...")
    run_veloxq_benchmark(
        num_reps=args.num_reps,
        num_steps=args.num_steps,
        instance_types=args.instance_types,
        num_shots=args.num_shots,
        output_prefix=args.output_prefix,
        max_instances=args.max_instances
    )
    print("Benchmark completed!")
