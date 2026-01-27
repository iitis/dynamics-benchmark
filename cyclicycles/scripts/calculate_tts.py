#!/usr/bin/env python3
"""
Script for calculating and plotting Time To Solution (TTS) metrics for dynamics instances.

TTS is calculated as:
    TTS = ln(1 - P_target) / ln(1 - P_success) * T_r

Where:
    - P_target is the target success probability (typically 0.99)
    - P_success is the probability of finding the ground state (gap = 0)
    - T_r is the runtime in milliseconds

Logic copied from plot_instance:
    - When gap (energy + offset) == 0, ground state is found
    - Count all samples that reach gap == 0
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


def load_and_analyze_results(solver_versions: list[str], instance_id: str, num_timepoints: int, 
                            use_ancilla: bool = False, filter_cycles: int = None) -> tuple[dict | None, dict | None, int | None, int, int]:
    """Load and analyze forward and cyclic annealing results across all solver versions.
    
    Uses the same logic as plot_instance: when gap (energy + offset) == 0, ground state is found.
    
    Args:
        solver_versions: List of solver versions to aggregate (e.g., ['4.1'] or ['1.8', '1.9', '1.10'])
        instance_id: ID of the dynamics instance
        num_timepoints: Number of timepoints
        use_ancilla: Whether to load results with ancilla
        
    Returns:
        tuple: (forward_analysis, cyclic_analysis, num_qubits, num_forward_files, num_cyclic_files)
    """
    plotter = Plotter(RESULT_DIR)
    instance = Instance(solver=solver_versions[0])
    
    # Get number of qubits
    dynamics_instances = instance.load_dynamics_instances(number_time_points=num_timepoints)
    if instance_id not in dynamics_instances:
        print(f"Error: Instance {instance_id} not found")
        return None, None, None
    
    num_qubits = dynamics_instances[instance_id]['num_variables']
    
    # Load results to get offset
    cyclic_df, forward_stats, offset, cyclic_best_percentage = plotter.load_results(
        solver_id=solver_versions[0],
        instance_type='dynamics',
        instance_id=instance_id,
        num_timepoints=num_timepoints,
        use_ancilla=use_ancilla
    )
    
    # Convert ground state to SPIN space (gap space)
    # gap = energy + offset, where energy is in SPIN space
    # When loaded from CSV, gnd_energy is in BINARY space
    # When we add offset: gnd_energy_binary + offset gives us the SPIN-space value
    gnd_energy_spin = 0.0
    
    # Forward annealing analysis - iterate over all result files across all solver versions
    forward_analysis = None
    total_time_ms_fw = 0.0
    total_samples_fw = 0
    successful_samples_fw = 0
    num_calls_fw = 0
    num_forward_files = 0
    
    ancilla_suffix = '_with_ancilla' if use_ancilla else '_no_ancilla'
    
    # Load forward results from all solver versions
    for solver_version in solver_versions:
        forward_path = RESULT_DIR / 'forward' / solver_version / f'dynamics_{instance_id}_timepoints_{num_timepoints}{ancilla_suffix}'
        
        if forward_path.exists():
            for result_file in forward_path.glob('*.npz'):
                data = np.load(result_file, allow_pickle=True)
                
                # Get timing
                timing_info = data['timing'].item() if isinstance(data['timing'], np.ndarray) else data['timing']
                if isinstance(timing_info, dict) and 'qpu_access_time' in timing_info:
                    total_time_ms_fw += timing_info['qpu_access_time'] * 1e-3  # Convert μs to ms
                
                # Get energies and calculate gap
                energies = data['energies']
                num_occurrences = data['num_occurrences']
                offset_val = float(data.get('offset', 0.0))
                
                # Gap = energy + offset (in SPIN space with offset)
                gaps = energies + offset_val
                # Ground state found when gap == 0 (within tolerance)
                tol = 1e-6
                found_indices = np.where(np.abs(gaps) < tol)[0]
                
                if len(found_indices) > 0:
                    successful_samples_fw += int(np.sum(num_occurrences[found_indices]))
                
                total_samples_fw += int(np.sum(num_occurrences))
                num_calls_fw += 1
                num_forward_files += 1
    
    if num_calls_fw > 0:
        p_success_fw = successful_samples_fw / total_samples_fw if total_samples_fw > 0 else 0.0
        runtime_per_call_ms = total_time_ms_fw / num_calls_fw
        tts_fw = calculate_tts(p_success_fw, runtime_ms=runtime_per_call_ms)
        
        forward_analysis = {
            'p_success': p_success_fw,
            'runtime_ms': runtime_per_call_ms,
            'tts': tts_fw,
            'num_calls': num_calls_fw,
            'total_time_ms': total_time_ms_fw,
            'successful_samples': successful_samples_fw,
            'total_samples': total_samples_fw
        }
    
    # Cyclic annealing analysis - iterate over all realizations across all solver versions
    cyclic_analysis = None
    cyclic_tts_values = []  # Store TTS from each realization
    num_cyclic_files = 0
    
    realization_pattern = f'dynamics_{instance_id}_timepoints_{num_timepoints}{ancilla_suffix}_realization_*'
    
    # Collect realization directories from all solver versions
    all_realization_dirs = []
    for solver_version in solver_versions:
        parent_dir = RESULT_DIR / solver_version
        realization_dirs = sorted(parent_dir.glob(realization_pattern))
        all_realization_dirs.extend(realization_dirs)
    
    if not all_realization_dirs:
        return forward_analysis, cyclic_analysis, num_qubits, num_forward_files, num_cyclic_files
    
    # Process each realization
    for cyclic_path in all_realization_dirs:
        total_time_ms_ca = 0.0
        total_samples_ca = 0
        successful_samples_ca = 0
        num_files_ca = 0
        
        # Extract number of cycles from directory name if filter_cycles is specified
        if filter_cycles is not None:
            path_name = cyclic_path.name
            if '_cycles_' in path_name:
                num_cycles = int(path_name.split('_')[0])
                if num_cycles != filter_cycles:
                    continue
            else:
                continue
        
        for result_file in cyclic_path.glob('*.npz'):
            data = np.load(result_file, allow_pickle=True)
            if data['num_cycles'] != 5:
                continue
            # Get timing
            timing_info = data['timing'].item() if isinstance(data['timing'], np.ndarray) else data['timing']
            if isinstance(timing_info, dict) and 'qpu_access_time' in timing_info:
                total_time_ms_ca += timing_info['qpu_access_time'] * 1e-3  # Convert μs to ms
            
            # Get energies and calculate gap
            energies = data['energies']
            num_occurrences = data['num_occurrences']
            offset_val = float(data.get('offset', 0.0))
            
            # Gap = energy + offset (in SPIN space with offset)
            gaps = energies + offset_val
            
            # Ground state found when gap == 0 (within tolerance)
            tol = 1e-6
            found_indices = np.where(np.abs(gaps) < tol)[0]
            
            if len(found_indices) > 0:
                successful_samples_ca += int(np.sum(num_occurrences[found_indices]))
            
            total_samples_ca += int(np.sum(num_occurrences))
            num_files_ca += 1
        
        # Track total cyclic files
        num_cyclic_files += num_files_ca
        if num_files_ca > 0:
            p_success_ca = successful_samples_ca / total_samples_ca if total_samples_ca > 0 else 0.0
            runtime_total_ms = total_time_ms_ca / num_files_ca # Total time by num files
            tts_ca = calculate_tts(p_success_ca, runtime_ms=runtime_total_ms)
            
            if tts_ca is not None:
                cyclic_tts_values.append(tts_ca)
    
    # Average TTS across all realizations
    if cyclic_tts_values:
        avg_tts_ca = np.mean(cyclic_tts_values)
        
        cyclic_analysis = {
            'p_success': None,  # Will be recalculated from average
            'runtime_ms': None,  # Will be recalculated from average
            'tts': avg_tts_ca,
            'num_realizations': len(cyclic_tts_values),
            'tts_values': cyclic_tts_values
        }
    
    return forward_analysis, cyclic_analysis, num_qubits, num_forward_files, num_cyclic_files


def main():
    parser = argparse.ArgumentParser(description='Calculate and plot TTS metrics for all dynamics instances')
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
            # Check if this folder starts with the specified solver ID
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
    forward_data = []  # List of (num_qubits, tts, instance_id, num_timepoints)
    cyclic_data = []   # List of (num_qubits, tts, instance_id, num_timepoints)
    cyclic_no_solution = []   # List of (num_qubits, p_success, instance_id, num_timepoints) when TTS can't be calculated
    
    print(f"\nAnalyzing all dynamics instances (solver: {args.solver} -> {solver_versions}, ancilla: {args.ancilla})\n")
    print(f"{'Instance ID':<20} {'Timepoints':<12} {'Qubits':<10} {'Forward TTS (ms)':<20} {'#FW':<5} {'Cyclic TTS (ms)':<20} {'#CA':<5}")
    print("-" * 105)
    
    # Extract unique timepoints from all solver versions
    timepoints_set = set()
    for solver_version in solver_versions:
        result_base = RESULT_DIR / solver_version
        if result_base.exists():
            for entry in result_base.iterdir():
                if entry.is_dir():
                    # Look for patterns like: dynamics_{id}_timepoints_{num}...
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
    print(timepoints_list)
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
            forward_analysis, cyclic_analysis, num_qubits, num_forward_files, num_cyclic_files = load_and_analyze_results(
                solver_versions, instance_id, num_timepoints, args.ancilla, filter_cycles=args.cycles
            )
            
            if num_qubits is None:
                continue
            
            forward_tts_str = "N/A"
            cyclic_tts_str = "N/A"
            forward_count_str = ""
            cyclic_count_str = ""
            
            if forward_analysis is not None:
                forward_count_str = str(num_forward_files)
                if forward_analysis['tts'] is not None:
                    forward_data.append((num_qubits, forward_analysis['tts'], instance_id, num_timepoints))
                    forward_tts_str = f"{forward_analysis['tts']:.2f}"
            else:
                forward_count_str = "0"
                continue
            
            if cyclic_analysis is not None:
                cyclic_count_str = str(num_cyclic_files)
                if cyclic_analysis['tts'] is not None:
                    cyclic_data.append((num_qubits, cyclic_analysis['tts'], instance_id, num_timepoints))
                    cyclic_tts_str = f"{cyclic_analysis['tts']:.2f}"
                else:
                    # Have data but no solution found
                    # For cyclic, we need to calculate average p_success from the tts_values
                    if cyclic_analysis.get('tts_values'):
                        # We need to recover p_success - but we don't have it directly, so use 0 as placeholder
                        cyclic_no_solution.append((num_qubits, 0.0, instance_id, num_timepoints))
                        cyclic_tts_str = "no_sol"
            else:
                cyclic_count_str = "0"
                continue
            
            print(f"{instance_id:<20} {num_timepoints:<12} {num_qubits:<10} {forward_tts_str:<20} {forward_count_str:<5} {cyclic_tts_str:<20} {cyclic_count_str:<5}")
    
    if not forward_data and not cyclic_data and not cyclic_no_solution:
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
            # Store all instance info for labeling
            instance_info = [f"{v[1]}({v[2]}tp)" for v in values]
            averaged.append((num_qubits, avg_tts, instance_info))
        
        return averaged
    
    # Average data
    forward_avg = average_by_qubits(forward_data) if forward_data else []
    cyclic_avg = average_by_qubits(cyclic_data) if cyclic_data else []
    
    # Create single plot with both curves
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Determine common y-axis limits for both plots
    all_tts_values = [x[1] for x in forward_avg] + [x[1] for x in cyclic_avg]
    if all_tts_values:
        y_min = min(all_tts_values) * 0.5
        y_max = max(all_tts_values) * 2
    else:
        y_min, y_max = 0.1, 1000
    
    # Forward annealing plot
    if forward_avg:
        qubits_fw = [x[0] for x in forward_avg]
        tts_fw = [x[1] for x in forward_avg]
        labels_fw = ['\n'.join(x[2]) for x in forward_avg]
        
        ax.scatter(qubits_fw, tts_fw, s=100, alpha=0.7, color='steelblue', label='Forward Annealing', zorder=3)
        ax.plot(qubits_fw, tts_fw, '--', alpha=0.5, color='steelblue', linewidth=2)
        
        # Add instance labels for forward annealing
        for i, label in enumerate(labels_fw):
            ax.annotate(label, (qubits_fw[i], tts_fw[i]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=5, color='steelblue')
    
    # Cyclic annealing plot
    if cyclic_avg:
        qubits_ca = [x[0] for x in cyclic_avg]
        tts_ca = [x[1] for x in cyclic_avg]
        labels_ca = ['\n'.join(x[2]) for x in cyclic_avg]
        
        ax.scatter(qubits_ca, tts_ca, s=100, alpha=0.7, color='darkred', label='Cyclic Annealing', zorder=3)
        ax.plot(qubits_ca, tts_ca, '--', alpha=0.5, color='darkred', linewidth=2)
        
        # Add instance labels for cyclic annealing
        for i, label in enumerate(labels_ca):
            ax.annotate(label, (qubits_ca[i], tts_ca[i]), 
                       xytext=(5, -15), textcoords='offset points', fontsize=5, color='darkred')
    
    # Cyclic annealing - no solution cases (plot as crosses)
    if cyclic_no_solution:
        qubits_ca_no = [x[0] for x in cyclic_no_solution]
        # For y-axis, place at bottom of visible range
        y_no_ca = [y_min] * len(qubits_ca_no)
        ax.scatter(qubits_ca_no, y_no_ca, s=150, alpha=0.7, color='darkred', marker='x', linewidths=2.5, zorder=3)
    
    ax.set_xlabel('Number of Qubits', fontsize=12, fontweight='bold')
    ax.set_ylabel('TTS (milliseconds)', fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    
    # Add note about crosses
    if cyclic_no_solution:
        ax.text(0.98, 0.02, 'Crosses (×) indicate cyclic annealing instances where no ground state was found',
               transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
               horizontalalignment='right', style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    ancilla_suffix = " (with ancilla)" if args.ancilla else " (without ancilla)"
    solver_display = f"{args.solver} ({','.join(solver_versions)})" if len(solver_versions) > 1 else args.solver
    ax.set_title(f'TTS Comparison - Solver {solver_display}{ancilla_suffix}', 
                 fontsize=14, fontweight='bold', pad=20)
    fig.tight_layout()
    
    # Save or display
    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        ancilla_str = "_with_ancilla" if args.ancilla else "_no_ancilla"
        save_path = save_dir / f'tts_comparison_{args.solver}{ancilla_str}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
        
        # Export dynamics (cyclic annealing) data to JSON
        if cyclic_avg:
            cycles_str = f"_{args.cycles}cycles" if args.cycles else ""
            json_filename = f'dynamics_tts_{args.solver}{ancilla_str}{cycles_str}.json'
            json_path = save_dir / json_filename
            
            cyclic_data_points = [{'x': x[0], 'y': x[1]} for x in cyclic_avg]
            
            export_data = {
                'metadata': {
                    'solver': args.solver,
                    'solver_versions': solver_versions,
                    'with_ancilla': args.ancilla,
                    'cycles_filter': args.cycles,
                    'description': 'Cyclic annealing TTS data: x = number of qubits, y = Time To Solution (milliseconds)'
                },
                'data': cyclic_data_points,
                'num_points': len(cyclic_data_points)
            }
            
            with open(json_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            print(f"Dynamics data exported to: {json_path}")
    else:
        plt.show()


if __name__ == '__main__':
    main()
