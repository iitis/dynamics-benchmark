"""
Script to run a dynamic problem instance and visualize with D-Wave Inspector.

Usage:
    python scripts/visualize_with_inspector.py --instance-id 3 --timepoints 5
    python scripts/visualize_with_inspector.py --solver 4.1 --n-nodes 263 --static
"""

import sys
import os
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cyclicycles.instance import Instance
from cyclicycles.config import DYNAMICS_INSTANCE_DIR, INSTANCE_DIR
import dimod
import json

try:
    from dwave.system import DWaveSampler, EmbeddingComposite
    from dwave.inspector import show
    INSPECTOR_AVAILABLE = True
except ImportError:
    INSPECTOR_AVAILABLE = False
    print("Warning: dwave-system and dwave-inspector not available")
    print("Install with: pip install dwave-system dwave-inspector")


def load_dynamic_problem(instance_id=1, num_timepoints=5):
    """Load a dynamic problem instance."""
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
    
    # Convert to BQM
    bqm = dimod.BQM.from_serializable(bqm_data)
    return bqm


def load_static_problem(solver='4.1', n_nodes=263, realization=1):
    """Load a static problem instance."""
    instance = Instance(solver=solver)
    j_terms = instance.load_instances(realization_number=realization)
    
    if str(n_nodes) not in j_terms:
        raise ValueError(f"N_{n_nodes} not found in solver {solver}")
    
    J = j_terms[str(n_nodes)]
    h = {}
    offset = 0.0
    
    bqm = dimod.BQM(h, J, offset, 'SPIN')
    return bqm


def visualize_with_inspector(bqm, solver_config=None):
    """
    Run problem and visualize with D-Wave Inspector.
    
    Args:
        bqm: Dimod BQM object
        solver_config: Optional solver configuration dict
    """
    if not INSPECTOR_AVAILABLE:
        print("Error: dwave-system and dwave-inspector required")
        return False
    
    try:
        print(f"BQM: {len(bqm)} variables, {len(bqm.quadratic)} interactions")
        print("Connecting to D-Wave sampler...")
        
        # Create sampler with embedding composite
        sampler = EmbeddingComposite(DWaveSampler())
        
        # Submit problem
        print("Submitting problem to D-Wave...")
        sampleset = sampler.sample(
            bqm,
            num_reads=100,
            chain_break_fraction=True,
            return_embedding=True,
        )
        
        print(f"Received {len(sampleset)} samples")
        print(f"Chain break fraction: {sampleset.info.get('chain_break_fraction', 'N/A')}")
        
        # Show in inspector
        print("Opening D-Wave Inspector...")
        show(sampleset)
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run a problem and visualize with D-Wave Inspector'
    )
    
    parser.add_argument('--dynamic', action='store_true',
                       help='Load dynamic instance')
    parser.add_argument('--instance-id', type=int, default=3,
                       help='Dynamic instance ID (1-8)')
    parser.add_argument('--timepoints', type=int, default=5,
                       help='Number of timepoints')
    
    parser.add_argument('--static', action='store_true',
                       help='Load static instance')
    parser.add_argument('--solver', default='4.1',
                       help='Solver version')
    parser.add_argument('--n-nodes', type=int, default=263,
                       help='Number of nodes')
    parser.add_argument('--realization', type=int, default=1,
                       help='Realization number')
    
    args = parser.parse_args()
    
    try:
        if args.dynamic or (not args.static and not args.dynamic):
            # Default to dynamic
            print(f"Loading dynamic instance {args.instance_id} with {args.timepoints} timepoints...")
            bqm = load_dynamic_problem(args.instance_id, args.timepoints)
        else:
            print(f"Loading static instance N_{args.n_nodes} from solver {args.solver}...")
            bqm = load_static_problem(args.solver, args.n_nodes, args.realization)
        
        success = visualize_with_inspector(bqm)
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
