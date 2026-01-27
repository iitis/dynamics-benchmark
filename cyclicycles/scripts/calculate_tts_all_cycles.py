#!/usr/bin/env python3
"""
Script for calculating and plotting Time To Solution (TTS) metrics for dynamics instances.
This version considers ground state found over ALL cycles, not just the final cycle.

TTS is calculated by finding the first cycle where ground state is reached across all cycles.
For each realization:
    - Find minimum energy in each cycle
    - Determine which cycle first reaches ground state (gap == 0)
    - Calculate TTS using cumulative time up to that cycle
    
If ground state is never reached, the realization is excluded from TTS calculation.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import json

# Add the src directory to Python path
src_dir = Path(__file__).resolve().parent.parent / 'src'
sys.path.insert(0, str(src_dir))

from cyclicycles.config import RESULT_DIR
from cyclicycles.instance import Instance
from cyclicycles.plotter import Plotter


def calculate_tts(p_success: float, p_target: float = 0.99, runtime_ms: float = 1.0) -> float | None:
    """Calculate Time To Solution.
    
    Args:
        p_success: Probability of success (0 to 1)
        p_target: Target success probability (default 0.99)
        runtime_ms: Runtime in milliseconds
        
    Returns:
        TTS in milliseconds, or None if p_success is 0 or 1
    """
    if p_success <= 0 or p_success >= 1:
        return None
    
    if p_success == 1.0:
        return None
    
    tts = np.log(1 - p_target) / np.log(1 - p_success) * runtime_ms
    return tts


def load_and_analyze_results_all_cycles(solver_versions: list[str], instance_id: str, num_timepoints: int, 
                                       use_ancilla: bool = False, filter_cycles: int = None) -> tuple[dict | None, dict | None, int | None, int, int]:
    """Load and analyze cyclic annealing results considering ALL cycles for ground state discovery.
    
    For cyclic annealing:
        - Load cycle_energies from each NPZ file
        - For each cycle in each realization, determine if ground state reached
        - Record the cycle where ground state was first found
        - Calculate TTS using cumulative timing up to that cycle
    
    Args:
        solver_versions: List of solver versions to aggregate
        instance_id: ID of the dynamics instance
        num_timepoints: Number of timepoints
        use_ancilla: Whether to load results with ancilla
        filter_cycles: Filter to specific cycle count
        
    Returns:
        tuple: (cyclic_analysis, num_qubits, num_cyclic_files)
    """
    instance = Instance(solver=solver_versions[0])
    
    # Get number of qubits
    dynamics_instances = instance.load_dynamics_instances(number_time_points=num_timepoints)
    if instance_id not in dynamics_instances:
        print(f"Error: Instance {instance_id} not found")
        return None, None, None
    
    num_qubits = dynamics_instances[instance_id]['num_variables']
    
    # Load offset and ground state info
    plotter = Plotter(RESULT_DIR)
    ground_energy = plotter.load_ground_state_energy(
        instance_type='dynamics',
        instance_id=instance_id,
        num_timepoints=num_timepoints
    )
    
    # Cyclic annealing analysis
    cyclic_analysis = None
    cyclic_tts_values = []  # Store TTS from each realization
    num_cyclic_files = 0
    
    ancilla_suffix = '_with_ancilla' if use_ancilla else '_no_ancilla'
    realization_pattern = f'dynamics_{instance_id}_timepoints_{num_timepoints}{ancilla_suffix}_realization_*'
    
    # Collect realization directories from all solver versions
    all_realization_dirs = []
    for solver_version in solver_versions:
        parent_dir = RESULT_DIR / solver_version
        realization_dirs = sorted(parent_dir.glob(realization_pattern))
        all_realization_dirs.extend(realization_dirs)
    
    if not all_realization_dirs:
        return cyclic_analysis, num_qubits, num_cyclic_files
    
    # Process each realization
    for cyclic_path in all_realization_dirs:
        # Extract number of cycles from directory name if filter_cycles is specified
        if filter_cycles is not None:
            path_name = cyclic_path.name
            if '_cycles_' in path_name:
                num_cycles = int(path_name.split('_')[0])
                if num_cycles != filter_cycles:
                    continue
            else:
                continue
        
        # Process all result files in this realization
        for result_file in cyclic_path.glob('*.npz'):
            data = np.load(result_file, allow_pickle=True)
            
            # Load cycle energies (minimum energy in each cycle)
            cycle_energies = data['cycle_energies']
            num_cycles_in_file = len(cycle_energies)
            
            # Get offset
            offset_val = float(data.get('offset', 0.0))
            
            # Get timing information - need per-cycle timing
            timing_info = data['timing'].item() if isinstance(data['timing'], np.ndarray) else data['timing']
            total_time_us = 0.0
            
            if isinstance(timing_info, dict) and 'qpu_access_time' in timing_info:
                total_time_us = timing_info['qpu_access_time']
            
            # Convert total time to per-cycle time (in milliseconds)
            time_per_cycle_ms = (total_time_us * 1e-3) / num_cycles_in_file if num_cycles_in_file > 0 else 0.0
            
            # Find which cycle first reached ground state
            tol = 1e-6
            ground_state_reached = False
            first_success_cycle = None
            cumulative_time_ms = 0.0
            
            for cycle_idx, energy in enumerate(cycle_energies):
                cumulative_time_ms = (cycle_idx + 1) * time_per_cycle_ms
                gap = energy + offset_val
                
                # Check if ground state found
                if ground_energy is not None:
                    ground_gap = ground_energy + offset_val
                    if np.abs(gap - ground_gap) < tol:
                        ground_state_reached = True
                        first_success_cycle = cycle_idx + 1
                        break
                else:
                    # If no ground state info, check if gap == 0
                    if np.abs(gap) < tol:
                        ground_state_reached = True
                        first_success_cycle = cycle_idx + 1
                        break
            
            # If ground state was reached, calculate TTS for this run
            if ground_state_reached:
                # For single run: TTS = cumulative_time (since we found it once)
                # But we need probability across multiple runs, so record the timing
                tts_value = cumulative_time_ms
                cyclic_tts_values.append(tts_value)
            
            num_cyclic_files += 1
    
    # Calculate average TTS across all runs that found ground state
    if cyclic_tts_values:
        # Success rate is number of runs that found ground state / total runs
        success_rate = len(cyclic_tts_values) / num_cyclic_files if num_cyclic_files > 0 else 0.0
        avg_time_to_solution = np.mean(cyclic_tts_values)
        
        # Calculate TTS using the success rate and average time
        tts_ca = calculate_tts(success_rate, runtime_ms=avg_time_to_solution)
        
        cyclic_analysis = {
            'p_success': success_rate,
            'runtime_ms': avg_time_to_solution,
            'tts': tts_ca,
            'num_successful': len(cyclic_tts_values),
            'total_runs': num_cyclic_files,
            'tts_values': cyclic_tts_values
        }
    
    return cyclic_analysis, num_qubits, num_cyclic_files


def main():
    parser = argparse.ArgumentParser(description='Calculate TTS for all cycles (ground state found across any cycle)')
    parser.add_argument('--solver', type=str, default="6",
                       help='D-Wave solver to analyze. Can be a base ID (e.g., 1, 4, 6) to aggregate all versions, '
                            'or a full version (e.g., 1.8, 4.1, 6.4)')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='Directory to save plot. If not specified, displays plot.')
    parser.add_argument('--ancilla', action='store_true',
                       help='Analyze results with ancilla transformation')
    parser.add_argument('--cycles', type=int, default=None,
                       help='Filter to only include samples with this many cycles')
    
    args = parser.parse_args()
    
    # Get all solver folders and filter to those matching the specified solver
    solver_versions = []
    for solver_dir in RESULT_DIR.iterdir():
        if solver_dir.is_dir() and not solver_dir.name.startswith('.'):
            if solver_dir.name.split('.')[0] == args.solver or solver_dir.name == args.solver:
                solver_versions.append(solver_dir.name)
    
    # Sort by version number
    def version_key(v):
        parts = v.split('.')
        try:
            return tuple(int(p) for p in parts)
        except:
            return (float('inf'),)
    
    solver_versions = sorted(solver_versions, key=version_key)
    
    if not solver_versions:
        print(f"Error: No solver versions found for solver ID: {args.solver}")
        return
    
    # Analyze all instances across all timepoints
    cyclic_data = []   # List of (num_qubits, tts, instance_id, num_timepoints)
    
    print(f"\nAnalyzing all dynamics instances (all cycles, solver: {args.solver} -> {solver_versions}, ancilla: {args.ancilla})\n")
    print(f"{'Instance ID':<20} {'Timepoints':<12} {'Qubits':<10} {'Cyclic TTS (ms)':<20} {'P_success':<12} {'Avg Time (ms)':<15} {'#CA':<5}")
    print("-" * 110)
    
    # Extract unique timepoints from all solver versions
    timepoints_set = set()
    for solver_version in solver_versions:
        result_base = RESULT_DIR / solver_version
        if result_base.exists():
            for entry in result_base.iterdir():
                if entry.is_dir():
                    name = entry.name
                    if 'timepoints_' in name:
                        parts = name.split('timepoints_')
                        if len(parts) > 1:
                            try:
                                timepoints_str = parts[1].split('_')[0]
                                if timepoints_str.isdigit():
                                    timepoints_set.add(int(timepoints_str))
                            except:
                                pass
    
    timepoints_list = sorted(list(timepoints_set))
    
    if not timepoints_list:
        print("No dynamics instances found")
        return
    
    # For each timepoint, analyze all instances
    for num_timepoints in timepoints_list:
        instance = Instance(solver=solver_versions[0])
        dynamics_instances = instance.load_dynamics_instances(number_time_points=num_timepoints)
        
        if not dynamics_instances:
            continue
        
        for instance_id in sorted(dynamics_instances.keys()):
            cyclic_analysis, num_qubits, num_cyclic_files = load_and_analyze_results_all_cycles(
                solver_versions, instance_id, num_timepoints, args.ancilla, filter_cycles=args.cycles
            )
            
            if num_qubits is None:
                continue
            
            cyclic_tts_str = "N/A"
            p_success_str = "N/A"
            avg_time_str = "N/A"
            cyclic_count_str = str(num_cyclic_files)
            
            if cyclic_analysis is not None:
                if cyclic_analysis['tts'] is not None:
                    cyclic_data.append((num_qubits, cyclic_analysis['tts'], instance_id, num_timepoints))
                    cyclic_tts_str = f"{cyclic_analysis['tts']:.2f}"
                    p_success_str = f"{cyclic_analysis['p_success']:.3f}"
                    avg_time_str = f"{cyclic_analysis['runtime_ms']:.2f}"
            
            print(f"{instance_id:<20} {num_timepoints:<12} {num_qubits:<10} {cyclic_tts_str:<20} {p_success_str:<12} {avg_time_str:<15} {cyclic_count_str:<5}")
    
    if not cyclic_data:
        print("\nNo valid TTS data found for any instances.")
        return
    
    # Average TTS values when there are multiple instances with the same number of qubits
    def average_by_qubits(data):
        """Group by num_qubits and average TTS values."""
        from collections import defaultdict
        grouped = defaultdict(list)
        for num_qubits, tts, instance_id, num_timepoints in data:
            grouped[num_qubits].append((tts, instance_id, num_timepoints))
        
        averaged = []
        for num_qubits in sorted(grouped.keys()):
            values = grouped[num_qubits]
            tts_values = [v[0] for v in values]
            avg_tts = np.mean(tts_values)
            instance_info = [f"{v[1]}({v[2]}tp)" for v in values]
            averaged.append((num_qubits, avg_tts, instance_info))
        
        return averaged
    
    # Average data
    cyclic_avg = average_by_qubits(cyclic_data) if cyclic_data else []
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Determine y-axis limits
    all_tts_values = [x[1] for x in cyclic_avg]
    if all_tts_values:
        y_min = min(all_tts_values) * 0.5
        y_max = max(all_tts_values) * 2
    else:
        y_min, y_max = 0.1, 1000
    
    # Cyclic annealing plot
    if cyclic_avg:
        qubits_ca = [x[0] for x in cyclic_avg]
        tts_ca = [x[1] for x in cyclic_avg]
        labels_ca = ['\n'.join(x[2]) for x in cyclic_avg]
        
        ax.scatter(qubits_ca, tts_ca, s=100, alpha=0.7, color='darkred', label='Cyclic Annealing (All Cycles)', zorder=3)
        ax.plot(qubits_ca, tts_ca, '--', alpha=0.5, color='darkred', linewidth=2)
        
        # Add instance labels
        for i, label in enumerate(labels_ca):
            ax.annotate(label, (qubits_ca[i], tts_ca[i]), 
                       xytext=(5, -15), textcoords='offset points', fontsize=5, color='darkred')
    
    ax.set_xlabel('Number of Qubits', fontsize=12, fontweight='bold')
    ax.set_ylabel('TTS (milliseconds)', fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    
    ancilla_suffix = " (with ancilla)" if args.ancilla else " (without ancilla)"
    solver_display = f"{args.solver} ({','.join(solver_versions)})" if len(solver_versions) > 1 else args.solver
    ax.set_title(f'TTS Comparison (Considering All Cycles) - Solver {solver_display}{ancilla_suffix}', 
                 fontsize=14, fontweight='bold', pad=20)
    fig.tight_layout()
    
    # Save or display
    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        ancilla_str = "_with_ancilla" if args.ancilla else "_no_ancilla"
        save_path = save_dir / f'tts_all_cycles_{args.solver}{ancilla_str}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
        
        # Export data to JSON
        if cyclic_avg:
            cycles_str = f"_{args.cycles}cycles" if args.cycles else ""
            json_filename = f'dynamics_tts_all_cycles_{args.solver}{ancilla_str}{cycles_str}.json'
            json_path = save_dir / json_filename
            
            cyclic_data_points = [{'x': x[0], 'y': x[1]} for x in cyclic_avg]
            
            export_data = {
                'metadata': {
                    'solver': args.solver,
                    'solver_versions': solver_versions,
                    'with_ancilla': args.ancilla,
                    'cycles_filter': args.cycles,
                    'description': 'TTS considering all cycles: x = number of qubits, y = Time To Solution (milliseconds)',
                    'note': 'Ground state discovery across any cycle, not just final cycle'
                },
                'data': cyclic_data_points,
                'num_points': len(cyclic_data_points)
            }
            
            with open(json_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            print(f"Data exported to: {json_path}")
    else:
        plt.show()


if __name__ == '__main__':
    main()
