#!/usr/bin/env python3
"""
Script for generating spin glass problems that are compatible with D-Wave QPUs.
"""

import argparse
import numpy as np
from pathlib import Path
from dwave.system import DWaveSampler
from typing import Dict, List, Tuple, Set
from cyclicycles.config import INSTANCE_DIR


def get_qpu_graph(sampler_id: str) -> Set[Tuple[int, int]]:
    """Get the hardware graph for a specific D-Wave QPU.
    
    Args:
        sampler_id: The QPU identifier ('1.7' or '6.4')
        
    Returns:
        Set of tuples representing available qubit connections
    """
    if sampler_id == "1.10":  # Advantage2
        qpu = DWaveSampler(solver="Advantage2_system1.10")
    elif sampler_id == "6.4":  # Advantage
        qpu = DWaveSampler(solver="Advantage_system6.4")
    else:
        raise ValueError(f"Invalid sampler id: {sampler_id}")
    
    # Create a set of available edges (both directions)
    edges = set()
    for u, v in qpu.edgelist:
        edges.add((u, v))
        edges.add((v, u))  # Add reversed edge for easier lookup
    
    return edges


def generate_spin_glass(n_nodes: int, available_edges: Set[Tuple[int, int]], 
                       seed: int | None = None) -> Dict[Tuple[int, int], float]:
    """Generate a spin glass problem with n_nodes that respects QPU connectivity.
    
    Creates a maximally connected spin glass instance using all available QPU connections
    between the specified nodes
    
    Args:
        n_nodes: Number of nodes in the problem
        available_edges: Set of allowed qubit connections from the QPU
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary mapping edge tuples to J coupling values [-1, 1]
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Get a list of valid edges for our problem size
    valid_edges = [(u, v) for u, v in available_edges if u < n_nodes and v < n_nodes]
    
    # Use all available edges to create a maximally connected spin glass
    selected_edges = range(len(valid_edges))
    
    # Generate random J values for all valid edges
    J = {}
    for idx in selected_edges:
        u, v = valid_edges[idx]
        # Generate random coupling in [-1, 1]
        J[(u, v)] = np.random.uniform(-1, 1)
    
    return J


def save_instance(J: Dict[Tuple[int, int], float], n_nodes: int, sampler_id: str, 
                 realization: int):
    """Save a spin glass instance to a file.
    
    Args:
        J: Dictionary of coupling values
        n_nodes: Number of nodes
        sampler_id: QPU identifier
        realization: Realization number
    """    
    # Save J terms - wrap dict in object array to maintain numpy compatibility
    J_obj = np.array([J], dtype=object)
    
    # Also save in the solver-specific directory
    solver_dir = INSTANCE_DIR / sampler_id / f'N_{n_nodes}_realization_{realization}'
    solver_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(solver_dir / 'J.npz', J=J_obj[0])


def main():
    parser = argparse.ArgumentParser(description='Generate spin glass problems compatible with D-Wave QPUs')
    parser.add_argument('--n_nodes', type=int, required=True,
                       help='Number of nodes in the instance to generate')
    parser.add_argument('--sampler', type=str, required=True,
                       choices=['1.7'],
                       help='D-Wave sampler to use (1.6=Advantage2, 6.4=Advantage)')
    parser.add_argument('--realization', type=int, default=1,
                       help='Realization number (default: 1)')
    parser.add_argument('--seed', type=int,
                       help='Random seed for reproducibility')

    
    args = parser.parse_args()
    
    print(f"Generating spin glass problem:")
    print(f"Number of nodes: {args.n_nodes}")
    print(f"Sampler: {args.sampler}")
    print(f"Realization: {args.realization}")
    
    # Get QPU graph
    available_edges = get_qpu_graph(args.sampler)
    
    # Generate problem
    J = generate_spin_glass(args.n_nodes, available_edges, seed=args.seed)
    
    # Save the instance
    save_instance(J, args.n_nodes, args.sampler, args.realization)
    
    print(f"Successfully generated and saved instance with {len(J)} couplings")


if __name__ == '__main__':
    main()