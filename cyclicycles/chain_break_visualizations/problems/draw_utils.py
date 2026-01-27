"""
Utilities for drawing problem instance graphs on Pegasus topology.

This module provides functions to visualize spin glass problems and their
mappings onto D-Wave QPU hardware graphs.
"""

import math
import dwave_networkx as dnx


# Color mappings for different problem types and node roles
edge_colormap = {
    'problem': 'blue',
    'unused': 'lightgrey',
    'chain': 'orange',
    'chain_highlight': 'red',
}

node_colormap = {
    'active': 'lightblue',
    'inactive': 'lightgrey',
    'chain_qubit': 'darkblue',
    'chain_head': 'darkblue',
    'chain_tail': 'lightgreen',
}


def which_edge(edge, edges):
    """
    Find which category an edge belongs to.
    
    Args:
        edge: Tuple (a, b) representing the edge
        edges: Dict mapping category names to lists of edges
        
    Returns:
        Category name or None if not found
    """
    for category, edge_list in edges.items():
        if edge in edge_list or edge[::-1] in edge_list:
            return category
    return None


def which_node(node, nodes):
    """
    Find which category a node belongs to.
    
    Args:
        node: Node identifier
        nodes: Dict mapping category names to lists of nodes
        
    Returns:
        Category name or None if not found
    """
    for category, node_list in nodes.items():
        if node in node_list:
            return category
    return None


def arc_to_dot(a, b, color):
    """Generate DOT string for an edge."""
    return f'    "{a}" -- "{b}"[penwidth = 2.5, color={color}]'


def thick_arc_to_dot(a, b, color, penwidth=6):
    """Generate DOT string for a thick edge (highlighted)."""
    return f'    "{a}" -- "{b}"[penwidth = {penwidth}, color={color}]'


def weighted_arc_to_dot(a, b, color, weight):
    """Generate DOT string for a weighted edge."""
    return f'    "{a}" -- "{b}"[penwidth = 2.5, color={color}, label="", xlabel="{weight}"]'


def node_to_dot(a, color, coord):
    """Generate DOT string for a node."""
    return f'    "{a}" [style=filled, fillcolor={color}, label="", shape=circle, height=0.25, width=0.25, pos="{coord[0]},{coord[1]}!"]'


def weighted_node_to_dot(a, color, weight, coord):
    """Generate DOT string for a weighted node."""
    return f'    "{a}" [style=filled, fillcolor={color}, label="", xlabel="{weight}", forcelabels=true, shape=circle, height=0.25, width=0.25, pos="{coord[0]},{coord[1]}!"]'


def graph_to_dot(file, problem_graph, hardware_graph, model=None, qbit_values=None, m=16, label=''):
    """
    Convert a problem graph and its embedding to DOT format for visualization.
    
    Args:
        file: File object to write DOT output to
        problem_graph: Dict with 'edges' and 'nodes' keys for problem structure
        hardware_graph: Dict with 'edges' and 'nodes' keys for hardware topology
        model: Optional dict with 'couplings' and 'biases' for the problem
        qbit_values: Optional dict of qubit values to display
        m: Pegasus graph parameter (default 16)
        label: Label for the graph
    """
    map_arcs = dnx.pegasus_graph(m, coordinates=True, nice_coordinates=True)
    map_nodes = dnx.pegasus_layout(map_arcs)
    
    def c(a):
        return dnx.pegasus_coordinates(m).nice_to_linear(a)
    
    print("graph G{outputorder=edgesfirst;", file=file)
    
    resize = math.sqrt(len(map_nodes)) * 2  # heuristic for better looking
    
    # Draw hardware edges first (light grey background)
    for a in map_arcs:
        for b in map_arcs[a]:
            a_new = c(a)
            b_new = c(b)
            if a_new < b_new:
                print(arc_to_dot(a_new, b_new, 'lightgrey'), file=file)
    
    # Draw problem edges (blue)
    if problem_graph and 'edges' in problem_graph and 'problem' in problem_graph['edges']:
        for edge in problem_graph['edges']['problem']:
            a, b = edge
            print(arc_to_dot(a, b, 'blue'), file=file)
    
    # Draw chain edges (orange by default, or red if highlighted)
    if problem_graph and 'edges' in problem_graph:
        if 'chain' in problem_graph['edges']:
            for edge in problem_graph['edges']['chain']:
                a, b = edge
                # Check if this edge is part of a highlighted chain
                if 'chain_highlight' in problem_graph['edges'] and edge in problem_graph['edges']['chain_highlight']:
                    print(thick_arc_to_dot(a, b, 'red', penwidth=8), file=file)
                else:
                    print(arc_to_dot(a, b, 'orange'), file=file)
    
    # Draw all nodes
    for a in map_arcs:
        a_new = c(a)
        coord = [map_nodes[a][0] * resize, map_nodes[a][1] * resize]
        
        # Determine node color
        color = 'lightgrey'  # default
        
        # Check if in chain_qubit (dark blue, highest priority)
        if problem_graph and 'nodes' in problem_graph and 'chain_qubit' in problem_graph['nodes']:
            if a_new in problem_graph['nodes']['chain_qubit']:
                color = 'darkblue'
        
        # Check if in active (light blue, lower priority)
        if color == 'lightgrey' and problem_graph and 'nodes' in problem_graph and 'active' in problem_graph['nodes']:
            if a_new in problem_graph['nodes']['active']:
                color = 'lightblue'
        
        print(node_to_dot(a_new, color, coord), file=file)
    
    print("}", file=file)
