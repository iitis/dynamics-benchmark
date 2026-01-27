#!/usr/bin/env python3
"""
Script for running forward quantum annealing on problem instances.
"""

import argparse
from pathlib import Path
from cyclicycles.runner import Runner,Instance
from dwave.system import DWaveSampler


def main():
    parser = argparse.ArgumentParser(description='Run forward quantum annealing')
    parser.add_argument('--n_nodes', type=str, default='2084',
                       help='Number of nodes in the instance to run. If not specified, uses first available.')
    
    parser.add_argument('--sampler', type=str, default='6.4',
                       choices=['1.6', '4.1', '6.4'],
                       help='D-Wave sampler to use (1.6=Advantage2, 4.1/6.4=Advantage)')
    
    args = parser.parse_args()
    
    # Initialize runner with specified sampler
    runner = Runner(sampler=args.sampler)
    
    print(f"Checking spin glass problem:")
    print(f"Number of nodes: {args.n_nodes if args.n_nodes else 'First available'}")
    print(f"Sampler: {args.sampler}")
    
    i = Instance()
    
    J_terms = i.load_instances()[args.n_nodes]

    sampler = args.sampler
    if sampler == "1.10":  # zephyr
        qpu = DWaveSampler(solver="Advantage2_system1.10")
    elif sampler == "6.4":
        qpu = DWaveSampler(solver="Advantage_system6.4")
    elif sampler == "4.3":
        qpu = DWaveSampler(solver="Advantage_system4.3")
    else:
        raise ValueError(f"Invalid solver id: {sampler}")
    
    edge_list = qpu.edgelist
    for key,_ in J_terms.items():
        if not (key in edge_list or (key[1],key[0]) in edge_list):
            print(key)
if __name__ == '__main__':
    main()