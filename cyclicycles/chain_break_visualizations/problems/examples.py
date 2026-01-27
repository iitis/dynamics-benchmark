"""
Example script demonstrating problem instance visualization.

This script shows how to:
1. Load problem instances
2. Generate graph structures
3. Create visualizations on Pegasus topology
"""

import os
import sys
from pathlib import Path

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, '..', '..', 'src'))
sys.path.insert(0, os.path.join(current_dir, '..', '..'))

try:
    from chain_break_visualizations.problems import (
        load_problem_instance,
        load_dynamic_problem,
        get_problem_graph_structure,
        export_problem_to_svg,
        generate_embedding,
        visualize_embedding,
    )
except ImportError:
    # Fallback for direct imports
    from draw_utils import graph_to_dot
    from real_graphs import (
        load_problem_instance,
        load_dynamic_problem,
        get_problem_graph_structure,
        export_problem_to_svg,
        generate_embedding,
        visualize_embedding,
    )


def visualize_static_instance(solver='4.1', n_nodes=263, realization=1, output_dir=None):
    """
    Visualize a static problem instance.
    
    Args:
        solver: Solver version
        n_nodes: Number of nodes
        realization: Realization number
        output_dir: Directory to save output (default: current directory)
    """
    if output_dir is None:
        output_dir = os.path.dirname(__file__)
    
    try:
        print(f"Loading problem instance: solver={solver}, n_nodes={n_nodes}, "
              f"realization={realization}")
        problem = load_problem_instance(solver=solver, n_nodes=n_nodes, 
                                       realization=realization)
        
        # Display problem structure
        graph_struct = get_problem_graph_structure(problem)
        num_active = len(graph_struct['nodes'].get('active', []))
        num_edges = len(graph_struct['edges'].get('problem', []))
        
        print(f"  Active nodes: {num_active}")
        print(f"  Problem edges: {num_edges}")
        
        # Generate embedding
        print("Generating embedding...")
        embedding = generate_embedding(problem, timeout=30)
        print(f"  Embedding chains: {len(embedding)}")
        
        # Generate visualization with embedding
        output_file = os.path.join(output_dir, 
                                   f"problem_N{n_nodes}_v{solver.replace('.', '_')}_embedded.svg")
        print(f"Generating embedded visualization: {output_file}")
        
        visualize_embedding(problem, embedding, output_file)
        print(f"✓ Visualization saved to {output_file}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False


def visualize_dynamic_instance(instance_id=1, num_timepoints=5, output_dir=None):
    """
    Visualize a dynamic problem instance.
    
    Args:
        instance_id: Instance ID (1-8)
        num_timepoints: Number of time points
        output_dir: Directory to save output (default: current directory)
    """
    if output_dir is None:
        output_dir = os.path.dirname(__file__)
    
    try:
        print(f"Loading dynamic instance: id={instance_id}, "
              f"timepoints={num_timepoints}")
        problem = load_dynamic_problem(instance_id=instance_id, 
                                      num_timepoints=num_timepoints)
        
        # Display problem structure
        graph_struct = get_problem_graph_structure(problem)
        num_active = len(graph_struct['nodes'].get('active', []))
        num_edges = len(graph_struct['edges'].get('problem', []))
        
        print(f"  Active nodes: {num_active}")
        print(f"  Problem edges: {num_edges}")
        print(f"  Total variables: {problem.get('num_variables', 'Unknown')}")
        
        # Generate embedding
        print("Generating embedding...")
        embedding = generate_embedding(problem, timeout=30)
        print(f"  Embedding chains: {len(embedding)}")
        
        # Generate visualization with embedding
        output_file = os.path.join(output_dir, 
                                   f"problem_dynamic_{instance_id}_t{num_timepoints}_embedded.svg")
        print(f"Generating embedded visualization: {output_file}")
        
        visualize_embedding(problem, embedding, output_file)
        print(f"✓ Visualization saved to {output_file}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Visualize problem instances on Pegasus topology'
    )
    parser.add_argument('--solver', default='1.10',
                       help='Solver version (e.g., 4.1, 6.4)')
    parser.add_argument('--n-nodes', type=int, default=263,
                       help='Number of nodes in static instance')
    parser.add_argument('--realization', type=int, default=1,
                       help='Realization number')
    parser.add_argument('--dynamic', action='store_true',default=True,
                       help='Visualize dynamic instance instead')
    parser.add_argument('--instance-id', type=int, default=1,
                       help='Dynamic instance ID (1-8)')
    parser.add_argument('--timepoints', type=int, default=5,
                       help='Number of timepoints for dynamic instance')
    parser.add_argument('--output-dir', default=None,
                       help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    if args.dynamic:
        success = visualize_dynamic_instance(
            instance_id=args.instance_id,
            num_timepoints=args.timepoints,
            output_dir=args.output_dir
        )
    else:
        success = visualize_static_instance(
            solver=args.solver,
            n_nodes=args.n_nodes,
            realization=args.realization,
            output_dir=args.output_dir
        )
    
    sys.exit(0 if success else 1)
