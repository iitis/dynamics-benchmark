"""
Problem visualization utilities for quantum annealing instances.

This package provides tools for visualizing spin glass problem instances
on D-Wave QPU topologies, including graph structure visualization and
embedding analysis.
"""

from .draw_utils import (
    graph_to_dot,
    arc_to_dot,
    thick_arc_to_dot,
    weighted_arc_to_dot,
    node_to_dot,
    weighted_node_to_dot,
    which_edge,
    which_node,
    edge_colormap,
    node_colormap,
)

from .real_graphs import (
    load_problem_instance,
    load_dynamic_problem,
    get_problem_graph_structure,
    get_hardware_topology,
    generate_embedding,
    export_problem_to_dot,
    export_problem_to_svg,
    visualize_embedding,
)

__all__ = [
    'graph_to_dot',
    'arc_to_dot',
    'weighted_arc_to_dot',
    'node_to_dot',
    'weighted_node_to_dot',
    'which_edge',
    'which_node',
    'edge_colormap',
    'node_colormap',
    'load_problem_instance',
    'load_dynamic_problem',
    'get_problem_graph_structure',
    'get_hardware_topology',
    'generate_embedding',
    'export_problem_to_dot',
    'export_problem_to_svg',
    'visualize_embedding',
]
