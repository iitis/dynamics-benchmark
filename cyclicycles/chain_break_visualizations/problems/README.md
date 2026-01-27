# Problem Visualization Utilities

This module provides tools for visualizing spin glass problem instances on D-Wave Pegasus QPU topologies.

## Overview

The problem visualization utilities are designed to complement the existing multiplier visualization tools in the parent `chain_break_visualizations` directory. While the multiplier tools visualize circuit embedding structures, these utilities focus on visualizing the problem instances themselves.

## Files

### `draw_utils.py`

Provides low-level graph drawing utilities:

- **`graph_to_dot(file, problem_graph, hardware_graph, ...)`** - Main function that converts a problem graph and hardware topology to Graphviz DOT format
- **`arc_to_dot(a, b, color)`** - Generate DOT string for an edge
- **`weighted_arc_to_dot(a, b, color, weight)`** - Generate DOT string for a weighted edge
- **`node_to_dot(a, color, coord)`** - Generate DOT string for a node
- **`weighted_node_to_dot(a, color, weight, coord)`** - Generate DOT string for a weighted node
- **`which_edge(edge, edges)`** - Categorize an edge based on problem structure
- **`which_node(node, nodes)`** - Categorize a node based on problem structure

**Color schemes:**

```python
edge_colormap = {
    'problem': 'blue',      # Edges in the problem
    'unused': 'lightgrey',  # Unused edges
    'chain': 'red',         # Chain edges (for embeddings)
}

node_colormap = {
    'active': 'lightblue',     # Active problem nodes
    'inactive': 'lightgrey',   # Inactive nodes
    'chain_head': 'darkblue',  # Chain header nodes
    'chain_tail': 'lightgreen',# Chain tail nodes
}
```

### `real_graphs.py`

Provides high-level problem loading and visualization:

- **`load_problem_instance(solver, n_nodes, realization)`** - Load a static problem instance from disk
- **`load_dynamic_problem(instance_id, num_timepoints)`** - Load a dynamic problem instance
- **`get_problem_graph_structure(problem)`** - Extract graph structure (nodes and edges) from a problem
- **`export_problem_to_dot(problem, output_file, solver_version, m)`** - Export problem to DOT format
- **`export_problem_to_svg(problem, output_svg, output_dot, m)`** - Export problem to SVG format using Graphviz

### `examples.py`

Demonstration script showing how to use the visualization utilities:

```bash
# Visualize a static instance
python examples.py --solver 4.1 --n-nodes 263 --realization 1

# Visualize a dynamic instance
python examples.py --dynamic --instance-id 1 --timepoints 5 --output-dir ./output
```

## Usage Examples

### Basic Usage

```python
from chain_break_visualizations.problems import (
    load_problem_instance,
    export_problem_to_svg,
)

# Load a problem instance
problem = load_problem_instance(solver='4.1', n_nodes=263, realization=1)

# Export to visualization
export_problem_to_svg(problem, 'problem_viz.svg')
```

### Advanced Usage with Custom Colors

```python
from chain_break_visualizations.problems import (
    get_problem_graph_structure,
    graph_to_dot,
)

# Load and process problem
problem = load_problem_instance(solver='4.1', n_nodes=678)
problem_graph = get_problem_graph_structure(problem)

# Create custom visualization
with open('output.dot', 'w') as f:
    graph_to_dot(f, problem_graph, hardware_graph, model=problem)
```

### Working with Dynamic Instances

```python
from chain_break_visualizations.problems import load_dynamic_problem

# Load dynamic problem
problem = load_dynamic_problem(instance_id=1, num_timepoints=5)

# Access problem structure
print(f"Number of variables: {problem['num_variables']}")
print(f"Number of couplings: {len(problem['J'])}")
```

## Problem Structure Format

Problems are represented as dictionaries with the following structure:

```python
problem = {
    'h': {},              # Linear terms {qubit: bias}
    'J': {...},           # Quadratic terms {(i,j): coupling}
    'offset': 0.0,        # Energy offset
    'num_variables': 100  # (optional) Total number of variables
}
```

Graph structures for visualization use:

```python
graph = {
    'nodes': {
        'active': [node_ids...],      # Active problem nodes
        'inactive': [node_ids...],    # Inactive nodes
    },
    'edges': {
        'problem': [(i,j) pairs...],  # Edges in the problem
        'unused': [(i,j) pairs...],   # Unused edges
        'chain': [(i,j) pairs...],    # Chain edges (optional)
    }
}
```

## Hardware Topology

The visualization uses D-Wave's Pegasus topology (default m=16):
- ~5000 qubits
- Hierarchical structure with tile-based layout
- Uses `dwave_networkx` for coordinate management

## Requirements

- `dwave-networkx`: QPU topology and layout utilities
- `dimod`: BQM (Binary Quadratic Model) handling
- `graphviz`: Neato layout engine for rendering (system dependency)

## Output Formats

### DOT Format
Human-readable Graphviz format suitable for manual inspection or further processing:
```
graph G{outputorder=edgesfirst;
    "0" -- "1"[penwidth = 2.5, color=blue]
    "0" [style=filled, fillcolor=lightblue, ...]
    ...
}
```

### SVG Format
Rendered visualization generated from DOT using Graphviz neato engine. SVG files can be:
- Opened in web browsers
- Embedded in documents
- Converted to other formats (PDF, PNG, etc.)

## Design Notes

These utilities follow the same patterns as the multiplier visualization tools:
- Similar naming conventions (`draw_utils.py`, `real_graphs.py`)
- Compatible with Pegasus topology visualization
- Extensible color and layout schemes
- DOT format as intermediate representation

However, they focus on problem instance visualization rather than circuit structure, making them complementary tools for analyzing quantum annealing workflows.

## Integration with Main Project

Import from the problems submodule:

```python
from chain_break_visualizations.problems import load_problem_instance
```

Or if working within the multipliers package:

```python
from ..problems import load_problem_instance
```

## Future Enhancements

Potential improvements:
- Interactive HTML visualization with Plotly
- Embedding visualization showing chain structure
- Comparison visualizations (before/after embedding)
- Statistical analysis of problem structure
- Support for custom color schemes
- Performance optimization for large instances
