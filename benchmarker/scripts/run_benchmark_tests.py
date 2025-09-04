#!/usr/bin/env python3

import argparse
from benchmarker.core.runner import BenchmarkRunner
from benchmarker.core.case import QuantumTestCase
from benchmarker.core.plotter import BenchmarkPlotter
from typing import List

def create_test_cases(systems: List[int], samplers: List[str], timepoints_range: List[int], 
                     annealing_times: List[int], num_reps: int, num_repeats: int = 1) -> List[QuantumTestCase]:
    """
    Create test cases with the given parameters.
    
    Args:
        systems: List of system IDs to test
        samplers: List of sampler names to use
        timepoints_range: List of timepoints to test
        annealing_times: List of annealing times (ta) to test
        num_reps: Number of repetitions per sample
        num_repeats: Number of times to repeat each test case
    
    Returns:
        List of QuantumTestCase instances
    """
    return [
        QuantumTestCase(
            system=system,
            sampler=sampler,
            timepoints=timepoints,
            ta=ta,
            num_reps=num_reps
        ) 
        for system in systems
        for ta in annealing_times
        for timepoints in timepoints_range
        for sampler in samplers
        for _ in range(num_repeats)
    ]

def main():
    parser = argparse.ArgumentParser(description='Run quantum dynamics benchmark tests')
    parser.add_argument('--systems', type=int, nargs='+', default=[1, 2, 3, 4],
                      help='System IDs to test')
    parser.add_argument('--samplers', type=str, nargs='+', default=['1.6'],
                      help='Sampler names to use')
    parser.add_argument('--timepoints-range', type=int, nargs='+', default=range(2, 3),
                      help='Range of timepoints to test')
    parser.add_argument('--annealing-times', type=int, nargs='+', default=[10, 100, 200],
                      help='Annealing times (ta) to test')
    parser.add_argument('--num-reps', type=int, default=1000,
                      help='Number of repetitions per sample')
    parser.add_argument('--num-repeats', type=int, default=5,
                      help='Number of times to repeat each test case')
    parser.add_argument('--file-limit', type=int, default=5,
                      help='Maximum number of files to include in the analysis')
    
    args = parser.parse_args()

    # Create test cases
    test_cases = create_test_cases(
        systems=args.systems,
        samplers=args.samplers,
        timepoints_range=args.timepoints_range,
        annealing_times=args.annealing_times,
        num_reps=args.num_reps,
        num_repeats=args.num_repeats
    )

    # Run benchmarks
    runner = BenchmarkRunner(test_cases=test_cases)
    results = runner.run_and_save()

    # Plot results
    plotter = BenchmarkPlotter()
    plotter.plot_tts(num_reps=args.num_reps, file_limit=args.file_limit)
    print("Benchmark tests completed successfully")

if __name__ == '__main__':
    main()
