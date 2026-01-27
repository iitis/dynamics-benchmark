#!/usr/bin/env python3
"""
Script for visualizing D-Wave QPU chain breaks and embeddings.

This script:
1. Performs a single sample on a problem instance
2. Retrieves the embedding used by the sampler
3. Identifies chain breaks (qubits in a chain that have different values)
4. Visualizes the QPU topology with:
   - Red/orange for broken chains
   - Blue/green for working chains and their used qubits
   - Grey for unused qubits
5. Outputs the visualization to an HTML file for interactive viewing
"""

import sys
from pathlib import Path
import numpy as np
import argparse
import json

# Add the src directory to Python path
src_dir = Path(__file__).resolve().parent.parent / 'src'
sys.path.insert(0, str(src_dir))

from cyclicycles.runner import Runner
from cyclicycles.config import RESULT_DIR, ensure_dir
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import networkx as nx

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: plotly not available. Using matplotlib for visualization.")


def get_embedding_and_response(runner, instance_type='dynamics', instance_id='1', 
                                num_timepoints=5, num_reads=1):
    """
    Execute a single sample and extract the embedding and response.
    
    Args:
        runner: Runner instance
        instance_type: 'static' or 'dynamics'
        instance_id: instance ID for dynamics instances
        num_timepoints: number of timepoints for dynamics instances
        num_reads: number of samples (keep low to get quick response)
    
    Returns:
        dict: Contains embedding, response, h, J, chain_breaks info
    """
    print(f"Executing sample on {instance_type} instance {instance_id}...")
    
    # Get the response
    response, result_data = runner.execute_instance(
        instance_type=instance_type,
        instance_id=instance_id,
        num_timepoints=num_timepoints,
        num_reads=num_reads
    )
    
    # Get embedding from the sampler
    # The sampler used is either EmbeddingComposite or FixedEmbeddingComposite
    embedding = None
    if hasattr(runner.dw_sampler, 'embedding'):
        embedding = runner.dw_sampler.embedding
    elif hasattr(runner.dw_sampler, 'child') and hasattr(runner.dw_sampler.child, 'embedding'):
        embedding = runner.dw_sampler.child.embedding
    else:
        # Try to extract from composite structure
        composite = runner.dw_sampler
        while hasattr(composite, 'child'):
            if hasattr(composite, 'embedding'):
                embedding = composite.embedding
                break
            composite = composite.child
    
    if embedding is None:
        print("Warning: Could not extract embedding from sampler. Creating embedding mapping...")
        # Fallback: create a simple embedding from response
        embedding = {}
    
    # Get the best sample
    best_idx = np.argmin(response.record.energy)
    best_sample = response.record.sample[best_idx]
    
    # Identify chain breaks
    chain_breaks = identify_chain_breaks(embedding, best_sample, response.variables)
    
    print(f"Found {len(embedding)} logical variables")
    print(f"Identified {len(chain_breaks)} broken chains")
    if chain_breaks:
        print(f"Broken chains: {chain_breaks}")
    
    return {
        'embedding': embedding,
        'response': response,
        'best_sample': best_sample,
        'chain_breaks': chain_breaks,
        'result_data': result_data,
        'variables': list(response.variables)
    }


def identify_chain_breaks(embedding, sample, variables):
    """
    Identify which chains are broken (have conflicting qubit values).
    
    Args:
        embedding: dict mapping logical variables to lists of physical qubits
        sample: dict mapping variables to their values
        variables: list of variable names
    
    Returns:
        dict: Maps logical variable to list of broken physical qubits in that chain
    """
    chain_breaks = {}
    
    for logical_var in embedding:
        chain = embedding[logical_var]
        if not chain or len(chain) <= 1:
            continue
        
        # Get the values of qubits in this chain
        chain_values = []
        for phys_qubit in chain:
            if phys_qubit in sample:
                chain_values.append(sample[phys_qubit])
        
        # Check if all values are the same (no break)
        if chain_values and len(set(chain_values)) > 1:
            # Chain is broken
            chain_breaks[logical_var] = chain
    
    return chain_breaks


def get_qpu_topology(sampler):
    """
    Extract QPU topology from the sampler.
    
    Returns:
        tuple: (edges list, qubits list)
    """
    try:
        # Get the sampler properties
        if hasattr(sampler, 'properties'):
            props = sampler.properties
        elif hasattr(sampler, 'child') and hasattr(sampler.child, 'properties'):
            props = sampler.child.properties
        else:
            # Navigate through composite chain
            current = sampler
            while hasattr(current, 'child'):
                if hasattr(current, 'properties'):
                    props = current.properties
                    break
                current = current.child
            else:
                if hasattr(current, 'properties'):
                    props = current.properties
                else:
                    raise ValueError("Could not find sampler properties")
        
        # Extract edges and qubits
        edges = []
        qubits = set()
        
        if 'couplers' in props:
            edges = props['couplers']
            qubits = set()
            for e in edges:
                qubits.add(e[0])
                qubits.add(e[1])
        
        if 'qubits' in props:
            qubits = set(props['qubits'])
        
        qubits = sorted(list(qubits))
        return edges, qubits
    
    except Exception as e:
        print(f"Error extracting topology: {e}")
        return [], []


def create_plotly_visualization(edges, qubits, embedding, chain_breaks, output_file):
    """
    Create an interactive Plotly visualization of the QPU topology.
    
    Args:
        edges: list of (qubit1, qubit2) tuples
        qubits: list of all qubits
        embedding: dict mapping logical variables to physical qubits
        chain_breaks: dict mapping logical variables to broken qubit chains
        output_file: path to save HTML file
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly not available, using matplotlib instead")
        return create_matplotlib_visualization(edges, qubits, embedding, chain_breaks, output_file)
    
    print(f"Creating Plotly visualization...")
    
    # Create graph
    G = nx.Graph()
    G.add_nodes_from(qubits)
    G.add_edges_from(edges)
    
    # Use spring layout for visualization
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Prepare edge traces
    edge_traces = []
    
    # Regular edges (unused)
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        showlegend=False
    )
    edge_traces.append(edge_trace)
    
    # Create color mapping for nodes
    node_colors = {}
    node_text = {}
    
    # Mark all qubits as grey initially
    for q in qubits:
        node_colors[q] = 'lightgrey'
        node_text[q] = f"Qubit {q}<br>Unused"
    
    # Mark used qubits
    all_used_qubits = set()
    for logical_var, chain in embedding.items():
        all_used_qubits.update(chain)
        for qubit in chain:
            if logical_var not in chain_breaks:
                # Chain is working
                node_colors[qubit] = 'lightblue'
                node_text[qubit] = f"Qubit {qubit}<br>Chain for var {logical_var}"
            else:
                # Qubit is in a broken chain
                node_colors[qubit] = 'salmon'
                node_text[qubit] = f"Qubit {qubit}<br>BROKEN CHAIN var {logical_var}"
    
    # Create node trace
    node_x = []
    node_y = []
    node_color = []
    node_hover = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_color.append(node_colors.get(node, 'lightgrey'))
        node_hover.append(node_text.get(node, f"Qubit {node}"))
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_hover,
        marker=dict(
            size=12,
            color=node_color,
            line=dict(width=1, color='black')
        ),
        showlegend=False
    )
    
    # Create the figure
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title="D-Wave QPU Topology - Chain Break Visualization",
                        showlegend=True,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        plot_bgcolor='white',
                        height=800,
                        width=1000
                    ))
    
    # Add legend
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(size=12, color='lightblue', line=dict(width=1, color='black')),
        name='Used Qubits (Working Chains)',
        showlegend=True
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(size=12, color='salmon', line=dict(width=1, color='black')),
        name='BROKEN CHAIN Qubits',
        showlegend=True
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(size=12, color='lightgrey', line=dict(width=1, color='black')),
        name='Unused Qubits',
        showlegend=True
    ))
    
    # Save figure
    fig.write_html(str(output_file))
    print(f"Interactive visualization saved to: {output_file}")


def create_matplotlib_visualization(edges, qubits, embedding, chain_breaks, output_file):
    """
    Create a matplotlib visualization of the QPU topology.
    
    Args:
        edges: list of (qubit1, qubit2) tuples
        qubits: list of all qubits
        embedding: dict mapping logical variables to physical qubits
        chain_breaks: dict mapping logical variables to broken qubit chains
        output_file: path to save image file
    """
    print(f"Creating matplotlib visualization...")
    
    # Create graph
    G = nx.Graph()
    G.add_nodes_from(qubits)
    G.add_edges_from(edges)
    
    # Use spring layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax, width=0.5, edge_color='lightgrey', alpha=0.5)
    
    # Separate qubits by type
    unused_qubits = set(qubits)
    working_chain_qubits = []
    broken_chain_qubits = []
    
    for logical_var, chain in embedding.items():
        unused_qubits -= set(chain)
        if logical_var in chain_breaks:
            broken_chain_qubits.extend(chain)
        else:
            working_chain_qubits.extend(chain)
    
    # Draw qubits by type
    # Unused qubits - grey
    nx.draw_networkx_nodes(G, pos, nodelist=list(unused_qubits), ax=ax,
                          node_color='lightgrey', node_size=200, 
                          node_edgecolor='black', linewidths=1)
    
    # Working chain qubits - blue
    nx.draw_networkx_nodes(G, pos, nodelist=working_chain_qubits, ax=ax,
                          node_color='lightblue', node_size=200,
                          node_edgecolor='darkblue', linewidths=2)
    
    # Broken chain qubits - red/orange
    nx.draw_networkx_nodes(G, pos, nodelist=broken_chain_qubits, ax=ax,
                          node_color='salmon', node_size=200,
                          node_edgecolor='darkred', linewidths=2)
    
    # Draw labels for broken qubits and some key ones
    labels = {}
    if broken_chain_qubits:
        for q in broken_chain_qubits[:20]:  # Limit labels to avoid clutter
            labels[q] = str(q)
    
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=8)
    
    # Create legend
    legend_elements = [
        mpatches.Patch(facecolor='lightblue', edgecolor='darkblue', label='Used Qubits (Working Chains)'),
        mpatches.Patch(facecolor='salmon', edgecolor='darkred', label='BROKEN CHAIN Qubits'),
        mpatches.Patch(facecolor='lightgrey', edgecolor='black', label='Unused Qubits')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=12)
    
    ax.set_title("D-Wave QPU Topology - Chain Break Visualization", fontsize=16, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    
    # Save figure
    output_file_png = str(output_file).replace('.html', '.png')
    plt.savefig(output_file_png, dpi=150, bbox_inches='tight')
    print(f"Matplotlib visualization saved to: {output_file_png}")
    plt.close()


def save_chain_break_report(data, output_file):
    """
    Save a detailed report about chain breaks in JSON format.
    
    Args:
        data: dict with embedding and chain break info
        output_file: path to save JSON file
    """
    report = {
        'num_logical_variables': len(data['embedding']),
        'num_broken_chains': len(data['chain_breaks']),
        'embedding': {str(k): v for k, v in data['embedding'].items()},
        'chain_breaks': {str(k): v for k, v in data['chain_breaks'].items()},
        'best_energy': float(np.min(data['response'].record.energy)),
        'num_samples': len(data['response'].record.energy),
    }
    
    output_file_json = str(output_file).replace('.html', '.json')
    with open(output_file_json, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Chain break report saved to: {output_file_json}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize D-Wave QPU chain breaks and embeddings'
    )
    parser.add_argument('--solver', type=str, default='1.10',
                       choices=['1.6', '1.7', '1.8', '1.9', '1.10', '4.1', '6.4'],
                       help='D-Wave solver to use (default: 1.10)')
    parser.add_argument('--instance-type', type=str, default='dynamics',
                       choices=['static', 'dynamics'],
                       help='Type of instance (default: dynamics)')
    parser.add_argument('--instance-id', type=str, default='3',
                       help='Instance ID for dynamics instances (default: 3)')
    parser.add_argument('--num-timepoints', type=int, default=5,
                       help='Number of timepoints for dynamics instances (default: 5)')
    parser.add_argument('--output-dir', type=str, default='./chain_break_visualizations',
                       help='Output directory for visualizations (default: ./chain_break_visualizations)')
    parser.add_argument('--num-reads', type=int, default=10,
                       help='Number of samples to read (default: 10)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    
    # Initialize runner
    print(f"Initializing D-Wave runner with solver {args.solver}...")
    runner = Runner(sampler=args.solver)
    
    # Get embedding and response
    data = get_embedding_and_response(
        runner,
        instance_type=args.instance_type,
        instance_id=args.instance_id,
        num_timepoints=args.num_timepoints,
        num_reads=args.num_reads
    )
    
    # Get QPU topology
    print("Extracting QPU topology...")
    edges, qubits = get_qpu_topology(runner.dw_sampler)
    print(f"QPU has {len(qubits)} qubits and {len(edges)} couplers")
    
    # Create filename
    if args.instance_type == 'dynamics':
        filename = f"chain_breaks_{args.instance_type}_{args.instance_id}_timepoints_{args.num_timepoints}"
    else:
        filename = f"chain_breaks_{args.instance_type}"
    
    output_file = output_dir / f"{filename}.html"
    
    # Create visualizations
    if edges and qubits:
        if PLOTLY_AVAILABLE:
            create_plotly_visualization(edges, qubits, data['embedding'], 
                                       data['chain_breaks'], output_file)
        else:
            create_matplotlib_visualization(edges, qubits, data['embedding'],
                                           data['chain_breaks'], output_file)
    else:
        print("Warning: Could not extract full QPU topology. Skipping visualization.")
    
    # Save detailed report
    save_chain_break_report(data, output_file)
    
    print("\n" + "="*60)
    print("CHAIN BREAK SUMMARY")
    print("="*60)
    print(f"Instance: {args.instance_type} {args.instance_id}")
    print(f"Solver: {args.solver}")
    print(f"Total logical variables: {len(data['embedding'])}")
    print(f"Total broken chains: {len(data['chain_breaks'])}")
    if data['chain_breaks']:
        print(f"Broken chain variables: {list(data['chain_breaks'].keys())}")
    print(f"Best energy found: {np.min(data['response'].record.energy):.6f}")
    print(f"Output directory: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
