"""
Utilities for loading and visualizing problem instances on Pegasus topology.

This module provides functions to load problem instances from data files,
extract their graph structure, generate embeddings, and create visualizations
on the QPU topology.
"""

import os
import sys
import json
import pickle
from pathlib import Path
import numpy as np
import dimod
import dwave_networkx as dnx
from minorminer import find_embedding

# Add parent directory to path to access config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from cyclicycles.config import INSTANCE_DIR, DYNAMICS_INSTANCE_DIR
from .draw_utils import graph_to_dot


def load_problem_instance(solver='4.1', n_nodes=None, realization=1):
    """
    Load a problem instance from disk.
    
    Args:
        solver: Solver version (e.g., '4.1', '6.4')
        n_nodes: Number of nodes in the problem (e.g., 263, 678, 1312, 2084, 5627)
        realization: Realization number (default 1)
        
    Returns:
        Dict with 'h' (linear terms), 'J' (quadratic terms), and 'offset'
    """
    if not n_nodes:
        raise ValueError("n_nodes must be specified")
    
    instance_path = INSTANCE_DIR / solver / f'N_{n_nodes}_realization_{realization}'
    
    if not instance_path.exists():
        raise FileNotFoundError(f"Instance not found: {instance_path}")
    
    # Load J terms
    j_file = instance_path / 'J.npz'
    if not j_file.exists():
        raise FileNotFoundError(f"J file not found: {j_file}")
    
    J = np.load(j_file, allow_pickle=True)['J'].item()
    
    # Return problem structure
    return {
        'h': {},  # Assuming no linear terms
        'J': J,
        'offset': 0.0
    }


def load_dynamic_problem(instance_id=1, num_timepoints=5):
    """
    Load a dynamic problem instance.
    
    Args:
        instance_id: ID of the instance (1-8)
        num_timepoints: Number of time points
        
    Returns:
        Dict with 'h', 'J', 'offset', and 'num_variables'
    """
    instance_path = DYNAMICS_INSTANCE_DIR / str(instance_id)
    
    if not instance_path.exists():
        raise FileNotFoundError(f"Dynamic instance not found: {instance_path}")
    
    # Find the JSON file with matching timepoints
    json_files = list(instance_path.glob(f"*_timepoints_{num_timepoints}.json"))
    
    if not json_files:
        raise FileNotFoundError(
            f"No JSON file found for {num_timepoints} timepoints in {instance_path}"
        )
    
    json_file = json_files[0]
    
    with open(json_file, 'r') as f:
        bqm_data = json.load(f)
    
    # Convert to BQM and extract problem structure
    bqm = dimod.BQM.from_serializable(bqm_data)
    
    return {
        'h': dict(bqm.linear),
        'J': dict(bqm.quadratic),
        'offset': bqm.offset,
        'num_variables': len(bqm.linear)
    }


def get_problem_graph_structure(problem):
    """
    Extract graph structure from a problem instance.
    
    Converts h and J into nodes and edges lists suitable for visualization.
    
    Args:
        problem: Dict with 'h' and 'J' keys
        
    Returns:
        Dict with 'nodes' and 'edges' for visualization
    """
    edges = {}
    nodes = {}
    
    # Extract nodes from linear terms
    if problem.get('h'):
        nodes['active'] = list(problem['h'].keys())
    else:
        nodes['active'] = []
    
    # Extract edges and additional nodes from quadratic terms
    edge_list = []
    for (i, j), coupling in problem.get('J', {}).items():
        edge_list.append((i, j))
        # Ensure nodes are tracked
        if i not in nodes['active']:
            nodes['active'].append(i)
        if j not in nodes['active']:
            nodes['active'].append(j)
    
    edges['problem'] = edge_list
    edges['unused'] = []
    
    return {
        'nodes': nodes,
        'edges': edges
    }


def get_hardware_topology(m=16):
    """
    Get the hardware topology for Pegasus.
    
    Args:
        m: Pegasus graph parameter (default 16)
        
    Returns:
        Dict with 'edges' and 'nodes' lists
    """
    pegasus_graph = dnx.pegasus_graph(m)
    
    return {
        'nodes': list(pegasus_graph.nodes()),
        'edges': list(pegasus_graph.edges())
    }


def generate_embedding(problem, m=16, timeout=10):
    """
    Generate an embedding of the problem onto Pegasus hardware.
    
    Args:
        problem: Dict with 'h' and 'J' keys representing the problem graph
        m: Pegasus graph parameter (default 16)
        timeout: Timeout in seconds for embedding search
        
    Returns:
        Dict mapping logical qubits to chains (lists of physical qubits)
    """
    # Build problem graph edges
    problem_edges = [(i, j) for (i, j) in problem.get('J', {}).keys()]
    
    # Get hardware graph
    hardware = dnx.pegasus_graph(m)
    
    # Find embedding
    embedding = find_embedding(problem_edges, hardware, timeout=timeout)
    
    return embedding


def visualize_embedding(problem, embedding, output_svg, output_dot=None, m=16):
    """
    Visualize an embedding of a problem onto Pegasus hardware.
    
    Args:
        problem: Dict with 'h' and 'J' keys
        embedding: Dict mapping logical qubits to chains of physical qubits
        output_svg: Path to write SVG file to
        output_dot: Optional path to save intermediate DOT file
        m: Pegasus graph parameter
    """
    if not output_dot:
        output_dot = output_svg.replace('.svg', '.dot')
    
    # Get problem and hardware graphs
    problem_graph = get_problem_graph_structure(problem)
    hardware_graph = get_hardware_topology(m)
    
    # Convert embedding to edge/node visualization structure
    chain_edges = []
    coupling_edges = []
    chain_nodes = []
    embedded_nodes = []
    
    # Extract all physical qubits and chains
    for logical_qubit, chain in embedding.items():
        chain_nodes.append(chain)
        embedded_nodes.extend(chain)
        
        # Add edges within chain
        for i in range(len(chain) - 1):
            chain_edges.append((chain[i], chain[i+1]))
    
    # Find a chain with length 3 or 4 to highlight
    highlighted_chain_edges = []
    for logical_qubit, chain in embedding.items():
        if 4 <= len(chain) <= 5:
            # Highlight this chain
            for i in range(len(chain) - 1):
                highlighted_chain_edges.append((chain[i], chain[i+1]))
            break  # Just highlight the first one we find
    
    # Update edges with chain edges
    problem_graph['edges']['chain'] = chain_edges
    if highlighted_chain_edges:
        problem_graph['edges']['chain_highlight'] = highlighted_chain_edges
    problem_graph['nodes']['chain_qubit'] = embedded_nodes
    
    edges = []
    for edge in problem_graph["edges"]['problem']:
        edges.append((embedding[edge[0]][0], embedding[edge[1]][0]))
    problem_graph["edges"]['problem'] = edges
    # Export to DOT with embedding visualization
    with open(output_dot, 'w') as f:
        graph_to_dot(f, problem_graph, hardware_graph, model=problem, m=m)
    
    # Use neato to render with fixed Pegasus coordinates
    os.system(f"neato -Tsvg {output_dot} -o {output_svg}")
    
    # Clean up DOT if not requested to keep
    if not (os.environ.get('KEEP_DOT') == '1'):
        try:
            os.remove(output_dot)
        except OSError:
            pass


def export_problem_to_dot(problem, output_file, solver_version='4.1', m=16):
    """
    Export a problem instance to DOT format for visualization.
    
    Args:
        problem: Dict with 'h' and 'J' keys
        output_file: Path to write DOT file to
        solver_version: D-Wave solver version (used for hardware graph)
        m: Pegasus graph parameter
    """
    # Get problem graph structure
    problem_graph = get_problem_graph_structure(problem)
    
    # Create hardware graph (placeholder - could load real hardware graph)
    hardware_graph = {
        'nodes': list(range(5000)),  # Pegasus M=16 has ~5000 qubits
        'edges': []
    }
    
    with open(output_file, 'w') as f:
        graph_to_dot(f, problem_graph, hardware_graph, model=problem, m=m)


def export_problem_to_svg(problem, output_svg, output_dot=None, m=16):
    """
    Export a problem instance to SVG format for visualization.
    
    Uses Graphviz neato to layout and render the graph.
    
    Args:
        problem: Dict with 'h' and 'J' keys
        output_svg: Path to write SVG file to
        output_dot: Optional path to save intermediate DOT file
        m: Pegasus graph parameter
    """
    if not output_dot:
        output_dot = output_svg.replace('.svg', '.dot')
    
    # Export to DOT
    export_problem_to_dot(problem, output_dot, m=m)
    
    # Use neato to render with fixed coordinates
    os.system(f"neato -Tsvg {output_dot} -o {output_svg}")
    
    # Clean up DOT if not requested to keep
    if not (os.environ.get('KEEP_DOT') == '1'):
        try:
            os.remove(output_dot)
        except OSError:
            pass


if __name__ == '__main__':
    # Example usage
    try:
        # Load a problem instance
        problem = load_problem_instance(solver='4.1', n_nodes=263, realization=1)
        
        output_dir = os.path.dirname(__file__)
        output_file = os.path.join(output_dir, 'problem_instance_263.svg')
        
        # Export to visualization
        export_problem_to_svg(problem, output_file)
        print(f"Visualization saved to {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")
