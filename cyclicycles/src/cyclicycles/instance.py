from pathlib import Path
import re
import numpy as np
from cyclicycles.config import INSTANCE_DIR, DYNAMICS_INSTANCE_DIR
import json
import dimod

class Instance:
    def __init__(self,solver='4.1'):
        self.J_terms = {}
        self.solver = solver
        self.instance_dir = INSTANCE_DIR / solver

        
    def load_instances(self, realization_number: int = 1):
        """Load all instances with N_ in folder name from the data directory.
        
        Args:
            realization_number (int): The realization number to load. Defaults to 1.
            
        Returns:
            dict: Dictionary with node numbers as keys and J terms as values.
        """
        # Get all directories that match the pattern N_*_realization_*
        for path in self.instance_dir.glob(f'N_*_realization_{realization_number}'):
            if not path.is_dir():
                continue
                
            # Extract N from the directory name using regex
            match = re.match(r'N_(\d+)_realization_', path.name)
            if not match:
                continue
                
            n_nodes = match.group(1)
            
            # Load J terms
            j_path = path / 'J.npz'
            if j_path.exists():
                self.J_terms[n_nodes] = np.load(j_path, allow_pickle=True)['J'].item()
        
        return self.J_terms
    
    def load_dynamics_instances(self, number_time_points: int = 5):
        """Load all dynamic instances from the dynamics directory.
        
        Args:
            number_time_points (int): Number of time points. Defaults to 5.
            
        Returns:
            dict: Dictionary with instance IDs as keys and BQM dicts (h, J, offset) as values.
        """
        dynamics_instances = {}
        
        if not DYNAMICS_INSTANCE_DIR.exists():
            print(f"Warning: Dynamics instance directory not found: {DYNAMICS_INSTANCE_DIR}")
            return dynamics_instances
        
        # Get all subdirectories (instance IDs)
        for instance_path in DYNAMICS_INSTANCE_DIR.iterdir():
            if not instance_path.is_dir():
                continue
            
            instance_id = instance_path.name
            
            # Find the file with the correct timepoints (ignoring precision)
            # Files follow pattern: precision_{x}_timepoints_{timepoints}.json
            matching_files = list(instance_path.glob(f"*_timepoints_{number_time_points}.json"))
            
            if not matching_files:
                continue
            
            # Use the first (and should be only) matching file
            file_path = matching_files[0]
            
            try:
                with open(file_path, 'r') as f:
                    bqm_data = json.load(f)
                
                # Convert to BQM and extract h, J, offset
                bqm = dimod.BQM.from_serializable(bqm_data)
                
                dynamics_instances[instance_id] = {
                    'h': dict(bqm.linear),
                    'J': dict(bqm.quadratic),
                    'offset': bqm.offset,
                    'num_variables': len(bqm.linear)
                }
            except Exception as e:
                print(f"Warning: Could not load dynamics instance {instance_id}: {e}")
                continue
        
        return dynamics_instances
    
    def load_all_instances(self, realization_number: int = 1, include_dynamics: bool = True):
        """Load both static and dynamic instances.
        
        Args:
            realization_number (int): The realization number for static instances. Defaults to 1.
            include_dynamics (bool): Whether to include dynamic instances. Defaults to True.
            
        Returns:
            tuple: (static_instances_dict, dynamics_instances_dict)
        """
        static = self.load_instances(realization_number)
        dynamics = self.load_dynamics_instances() if include_dynamics else {}
        
        return static, dynamics
    
    def remove_linear_terms_with_ancilla(self, h: dict, J: dict, ancilla_ratio: int = 1, offset: float = 0.0):
        """Transform an Ising problem with linear terms to one with only quadratic terms using ancilla qubits.
        
        This implements the reduction technique from supplementary material S1, where linear biases h_i
        are eliminated by introducing ancilla qubits s_{N+k} (for k=1..num_ancilla).
        
        Each group of `ancilla_ratio` original qubits shares one ancilla qubit. The ancilla qubit s_anc
        is coupled to qubits in its group with coupling h_i, effectively moving the linear bias to a
        quadratic term: h_i * s_i * s_anc
        
        Args:
            h (dict): Dictionary of linear terms {qubit: bias}
            J (dict): Dictionary of quadratic terms {(i,j): coupling}
            ancilla_ratio (int): Number of original qubits per ancilla qubit. Default is 1 (one ancilla per qubit).
                                If > 1, multiple qubits share the same ancilla (reduces qubit count).
            offset (float): Energy offset from the original problem
            
        Returns:
            dict: Dictionary containing:
                - 'h': Empty dict (no linear terms)
                - 'J': New quadratic terms dict (original J + ancilla couplings)
                - 'offset': Original offset
                - 'num_variables': Total number of variables (original + ancilla)
                - 'num_original': Number of original variables
                - 'num_ancilla': Number of ancilla variables
                - 'ancilla_mapping': Dict mapping ancilla qubit to list of original qubits it couples to
        """
        if not h:
            return {
                'h': {},
                'J': J,
                'offset': offset,
                'num_variables': len(set([i for (i,j) in J.keys()] + [j for (i,j) in J.keys()])),
                'num_original': len(set([i for (i,j) in J.keys()] + [j for (i,j) in J.keys()])),
                'num_ancilla': 0,
                'ancilla_mapping': {}
            }
        
        # Get original qubits from h dict
        original_qubits = sorted(list(h.keys()))
        num_original = len(original_qubits)
        
        # Calculate number of ancilla qubits needed
        num_ancilla = (num_original + ancilla_ratio - 1) // ancilla_ratio
        max_qubit = max([max(i, j) for (i, j) in J.keys()]) if J else max(original_qubits)
        

        J_new = dict(J)
        ancilla_mapping = {}
        h_new = {}
        
        for ancilla_idx in range(num_ancilla):
            # Ancilla qubit index
            ancilla_qubit = max_qubit + 1 + ancilla_idx
            
            # Which original qubits does this ancilla couple to?
            start_qubit_idx = ancilla_idx * ancilla_ratio
            end_qubit_idx = min((ancilla_idx + 1) * ancilla_ratio, num_original)
            coupled_qubits = original_qubits[start_qubit_idx:end_qubit_idx]
            
            ancilla_mapping[ancilla_qubit] = coupled_qubits
            h_new[ancilla_qubit] = -2
            # Add coupling between ancilla and each original qubit in its group
            for original_qubit in coupled_qubits:
                coupling_strength = h[original_qubit]

                # Store as (min, max) tuple for consistent ordering
                if coupling_strength > 0.0:
                    edge = tuple(sorted([original_qubit, ancilla_qubit]))
                    J_new[edge] = coupling_strength
        offset += num_ancilla * 2
        
        return {
            'h': h_new,  # No linear terms in the new problem
            'J': J_new,
            'offset': offset,
            'num_variables': num_original + num_ancilla,
            'num_original': num_original,
            'num_ancilla': num_ancilla,
            'ancilla_mapping': ancilla_mapping
        }
