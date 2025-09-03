#!/usr/bin/env python3
"""
This script calculates and prints the success probability ratios between different
D-Wave architectures for native and non-native systems.

USAGE:
    python print_success_ratios.py [OPTIONS]

    Options:
        --annealing-times LIST    List of annealing times in microseconds (default: [10,100,200,500])
        --timepoints INT         Number of timepoints to consider (default: 2)
        --file-limit INT        Maximum number of files to process (default: 20)

    Examples:
        python print_success_ratios.py
        python print_success_ratios.py --annealing-times 10,200,500
        python print_success_ratios.py --timepoints 3 --file-limit 10
"""

import argparse
import pandas as pd
from benchmarker.core import results_loader

def parse_list(value: str) -> list:
    """Parse a comma-separated string into a list of integers."""
    return [int(x.strip()) for x in value.split(',')]

def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Calculate success probability ratios between D-Wave architectures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s
  %(prog)s --annealing-times 10,200,500
  %(prog)s --timepoints 3 --file-limit 10
        """
    )
    
    parser.add_argument(
        '--annealing-times',
        type=parse_list,
        default='10,100,200,500',
        help='List of annealing times in microseconds. Default: [10,100,200,500]'
    )
    
    parser.add_argument(
        '--timepoints',
        type=int,
        default=2,
        help='Number of timepoints to consider. Default: 2'
    )
    
    parser.add_argument(
        '--file-limit',
        type=int,
        default=20,
        help='Maximum number of files to process. Default: 20'
    )
    
    return parser

def calculate_success_ratios(annealing_times: list, timepoints: int, file_limit: int) -> pd.DataFrame:
    """Calculate success probability ratios between architectures."""
    # Configuration
    topologies = ['1.4', '6.4']  # Updated from '1.6' to '1.4' to match the loader
    native_systems = [1, 2, 4, 5, 6, 7]
    non_native_systems = [3, 8]
    
    loader = results_loader.ResultsLoader()
    ratio_dfs = []

    # Process each group of systems (native and non-native)
    for systems in [native_systems, non_native_systems]:
        dfs = []
        for system in systems:
            for topology in topologies:
                for ta in annealing_times:
                    df = loader.get_dwave_success_rates(
                        system, 
                        topology=topology, 
                        ta=ta, 
                        grouped=True,
                        file_limit=file_limit
                    )
                    df = df[df['timepoints'] == timepoints]  # Single timepoint comparison
                    df['system'] = system
                    df['ta'] = ta
                    dfs.append(df)

        # Combine and process the data
        combined_df = pd.concat(dfs)
        combined_df = combined_df[['topology', 'success_prob', 'ta']].groupby(by=['topology', 'ta']).mean()
        df_reset = combined_df.reset_index()

        # Calculate ratios between architectures
        df_14 = df_reset[df_reset['topology'] == '1.4'].set_index('ta')[['success_prob']]
        df_64 = df_reset[df_reset['topology'] == '6.4'].set_index('ta')[['success_prob']]
        ratio_dfs.append(df_14 / df_64)

    # Combine native and non-native ratios
    ratio_df = ratio_dfs[0].copy()
    ratio_df['p_ratio_non_native'] = ratio_dfs[1]['success_prob']
    ratio_df = ratio_df.rename({'success_prob': 'p_ratio_native'}, axis=1)
    
    return ratio_df

def main():
    parser = create_parser()
    args = parser.parse_args()
    
    # Calculate ratios
    ratio_df = calculate_success_ratios(
        annealing_times=args.annealing_times,
        timepoints=args.timepoints,
        file_limit=args.file_limit
    )
    
    # Print results in LaTeX format
    print("\nSuccess Probability Ratios (Advantage2/Advantage):")
    print("================================================")
    print(ratio_df.to_latex(float_format=lambda x: '{:.3f}'.format(x)))

if __name__ == "__main__":
    main()