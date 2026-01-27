from pathlib import Path
import numpy as np
import json
import pickle
import signal
import threading
from dwave.system import DWaveSampler, EmbeddingComposite, FixedEmbeddingComposite
import dimod
from .instance import Instance
from .config import RESULT_DIR, INSTANCE_DIR, DATA_DIR, ensure_dir
import dwave.inspector
import re
import time
class TimeoutException(Exception):
    """Raised when an operation times out."""
    pass

def timeout_handler(signum, frame):
    """Handler for timeout signal."""
    raise TimeoutException("Operation timed out")

class Runner:
    def __init__(self, sampler='6.4'):
        self.time_path = DATA_DIR / 'time.json'
        ensure_dir(self.time_path.parent)
        if not self.time_path.exists():
            with self.time_path.open('w') as f:
                json.dump({"time_ms": 0}, f)
                
    
        self.sampler = sampler
        # Default annealing schedule and h_gain
        self.anneal_schedule = [
            [0.0,   1.0],   # start at s=1  (Bx ~ 0)
            [0.1,   1.0],   # turn on Bz while keeping s=1
            [0.6,   0.35],  # ramp down to s_min -> Bx high
            [300.0, 1.0],   # ramp back to s=1 (Bx -> 0)
        ]  # times in microseconds

        self.h_gain_schedule = [
            [0.0,   0.0],   # Bz=0 at t=0 per table
            [0.1,   1],  # raise Bz by 0.1 μs
            [0.6,   1],  # keep Bz ~ constant while Bx rises
            [300.0, 0],   # bring Bz back to 0 by the end
        ]

        # Configure D-Wave sampler based on solver ID
        if self.sampler == "1.10":  # zephyr
            self.qpu = DWaveSampler(solver="Advantage2_system1.10")
        elif self.sampler == "6.4":
            self.qpu = DWaveSampler(solver="Advantage_system6.4")
        elif self.sampler == "4.3":
            self.qpu = DWaveSampler(solver="Advantage_system4.3")
        else:
            raise ValueError(f"Invalid solver id: {self.sampler}")
        self.dw_sampler = self.qpu
        

    def _log_access_time(self, access_time_us: float):
        """Log the D-Wave access time to time.json.
        
        Args:
            access_time_ms (float): Access time in milliseconds.
        """
        try:
            with self.time_path.open('r') as f:
                time_dict = json.load(f)
            time_dict['time_ms'] += access_time_us * 1e-3
            with self.time_path.open('w') as f:
                json.dump(time_dict, f)
        except Exception as e:
            print(f"Error logging access time: {e}")

    def _get_or_create_embedding(self, instance_type: str, instance_id: str | None = None,
                                 num_timepoints: int = 5, n_nodes: str | None = None,
                                 h: dict | None = None, J: dict | None = None) -> EmbeddingComposite:
        """Get embedding composite for the problem instance.
        
        This method returns an EmbeddingComposite, which computes a heuristic embedding
        for the given problem. The embedding is computed fresh each time to avoid issues
        with caching and serialization of D-Wave objects.
        
        Args:
            instance_type (str): Either 'static' or 'dynamics'.
            instance_id (str, optional): ID of the dynamics instance.
            num_timepoints (int): Number of timepoints for dynamics instances.
            n_nodes (str, optional): Number of nodes for static instances.
            h (dict, optional): Linear terms.
            J (dict, optional): Quadratic terms.
            
        Returns:
            EmbeddingComposite: Composite sampler with heuristic embedding.
        """
        max_retries = 3
        timeout_sec = 30
        
        for attempt in range(max_retries):
            print(f"Creating EmbeddingComposite for heuristic embedding (attempt {attempt + 1}/{max_retries})...", end=' ', flush=True)
            
            # Set up timeout using threading
            result_container = {'composite': None, 'exception': None}
            
            def create_embedding():
                result_container['composite'] = EmbeddingComposite(self.dw_sampler)
            
            thread = threading.Thread(target=create_embedding, daemon=True)
            thread.start()
            thread.join(timeout=timeout_sec)
            
            if thread.is_alive():
                print(f"TIMEOUT (>{timeout_sec}s)")
                if attempt < max_retries - 1:
                    print("  Retrying...")
                    continue
                else:
                    raise TimeoutException(f"EmbeddingComposite creation timed out after {timeout_sec}s")
            
            if result_container['exception']:
                raise result_container['exception']
            
            if result_container['composite'] is None:
                raise RuntimeError("Failed to create EmbeddingComposite")
            
            print("OK")
            return result_container['composite']
                
           
        
        raise RuntimeError(f"Failed to create EmbeddingComposite after {max_retries} attempts")

    def _load_and_prepare_problem(self, n_nodes: str | None, instance_type: str, instance_id: str | None,
                                  num_timepoints: int, use_ancilla_transformation: bool = False, 
                                  ancilla_ratio: int = 1):
        """Load and prepare a problem instance for annealing.
        
        This method consolidates the logic for loading both static and dynamic instances,
        converting to SPIN vartype for D-Wave, and applying ancilla transformation if requested.
        
        IMPORTANT: Ancilla transformation is applied AFTER BINARY->SPIN conversion because
        the vartype conversion itself introduces new linear h-terms from the original quadratic
        terms. The ancilla method eliminates these h-terms in SPIN space.
        
        Args:
            n_nodes (str, optional): Number of nodes for static instances.
            instance_type (str): Either 'static' or 'dynamics'.
            instance_id (str, optional): ID for dynamics instances.
            num_timepoints (int): Number of timepoints for dynamics instances.
            use_ancilla_transformation (bool): If True, apply ancilla transformation to eliminate linear h-terms in SPIN space.
            ancilla_ratio (int): Ratio for ancilla transformation (qubits per ancilla).
            
        Returns:
            dict: Contains h, J, offset, vartype, and metadata about the problem.
        """
        instance = Instance(solver=self.sampler)
        
        if instance_type == 'dynamics':
            if instance_id is None:
                raise ValueError("instance_id must be specified for dynamics instances")
            
            dynamics_instances = instance.load_dynamics_instances(number_time_points=num_timepoints)
            if instance_id not in dynamics_instances:
                raise ValueError(f"Dynamics instance {instance_id} not found")
            
            dyn_instance = dynamics_instances[instance_id]
            h_binary = dyn_instance['h']
            J_binary = dyn_instance['J']
            offset = dyn_instance['offset']
            
            # Step 1: Convert from BINARY to SPIN for D-Wave
            bqm_binary = dimod.BQM(h_binary, J_binary, offset, vartype='BINARY')
            bqm_spin = bqm_binary.copy()
            bqm_spin.change_vartype(dimod.SPIN)

            h_spin = dict(bqm_spin.linear)
            J_spin = dict(bqm_spin.quadratic)
            offset_spin = bqm_spin.offset

            # Step 2: Apply ancilla transformation if requested (removes h-terms in SPIN space)
            if use_ancilla_transformation:
                transformed = instance.remove_linear_terms_with_ancilla(h_spin, J_spin, ancilla_ratio=ancilla_ratio, offset=float(offset_spin))
                h_spin = transformed['h']
                J_spin = transformed['J']
                offset_spin = transformed['offset']
            
            return {
                'h': h_spin,
                'J': J_spin,
                'offset': offset_spin,
                'vartype': 'SPIN',
                'instance_type': 'dynamics',
                'instance_id': instance_id,
                'num_timepoints': num_timepoints,
                'n_nodes': None,
                'used_ancilla': use_ancilla_transformation,
                'ancilla_ratio': ancilla_ratio if use_ancilla_transformation else None
            }
        
        else:  # static instances
            J_terms = instance.load_instances()
            
            if not J_terms:
                raise ValueError("No static instances found")
            
            # Select instance
            if n_nodes is None:
                n_nodes = list(J_terms.keys())[0]
            elif n_nodes not in J_terms:
                raise ValueError(f"Instance with {n_nodes} nodes not found")
            
            J = J_terms[n_nodes]
            h = {}  # No linear terms for static instances
            offset = 0.0
            
            # Ancilla transformation not applicable to static instances (no h terms)
            if use_ancilla_transformation:
                print("Note: Ancilla transformation not applicable to static instances (no linear terms)")
            
            return {
                'h': h,
                'J': J,
                'offset': offset,
                'vartype': 'SPIN',
                'instance_type': 'static',
                'instance_id': None,
                'num_timepoints': None,
                'n_nodes': n_nodes,
                'used_ancilla': False,
                'ancilla_ratio': None
            }

    def  execute_cyclic_annealing(self, n_nodes: str | None = None, num_cycles: int = 5, num_reads: int = 1000, 
                             use_forward_init: bool = False, instance_type: str = 'static', instance_id: str | None = None,
                             num_timepoints: int = 5, use_ancilla_transformation: bool = False, ancilla_ratio: int = 1):
        """Execute cyclic annealing on a problem instance.
        
        Args:
            n_nodes (str, optional): Number of nodes to select specific static instance.
                If None, executes first available instance.
            num_cycles (int): Number of cyclic annealing iterations. Defaults to 5.
            num_reads (int): Number of samples per cycle. Defaults to 1000.
            use_forward_init (bool): If True, run forward annealing first and use its best
                solution as initial state. Defaults to False.
            instance_type (str): Either 'static' or 'dynamics'. Defaults to 'static'.
            instance_id (str, optional): ID of the dynamics instance (required if instance_type='dynamics').
            num_timepoints (int): Number of timepoints for dynamics instances. Defaults to 5.
            use_ancilla_transformation (bool): If True, apply ancilla transformation to dynamics instances.
            ancilla_ratio (int): Ratio for ancilla transformation (qubits per ancilla). Defaults to 1.
            
        Returns:
            tuple: (final_response, result_data, cycle_energies)
        """
    
        # Load and prepare problem first to get instance details
        problem = self._load_and_prepare_problem(n_nodes, instance_type, instance_id, num_timepoints,
                                                  use_ancilla_transformation, ancilla_ratio)
        
        h = problem['h']
        J = problem['J']
        offset = problem['offset']
        n_nodes = problem['n_nodes']
        
        # Setup embedding (cached) for dynamics instances
        if instance_type == 'dynamics':
            self.dw_sampler = self._get_or_create_embedding(instance_type, instance_id, num_timepoints, n_nodes, h, J)
        
        used_qubits = set([i for (i,j) in J.keys()] + [j for (i,j) in J.keys()])
        initial_state = {qubit: 0 if qubit in used_qubits else 3 for qubit in self.qpu.nodelist}

        # Initialize state for first cycle
        num_variables = len(used_qubits)
        cycle_energies = []
        best_state = initial_state
        best_energy = float('inf')
        
        # Run forward annealing first if requested
        if use_forward_init:
            print("Running forward annealing for initialization...")
            forward_response, _ = self.execute_instance(n_nodes=n_nodes, num_reads=num_reads, 
                                                       instance_type=instance_type, instance_id=instance_id,
                                                       num_timepoints=num_timepoints,
                                                       use_ancilla_transformation=use_ancilla_transformation,
                                                       ancilla_ratio=ancilla_ratio)
            
            # Get best solution from forward annealing
            min_energy_idx = np.argmin(forward_response.record.energy)
            forward_energy = forward_response.record.energy[min_energy_idx]
            forward_state = {qubit: int(forward_response.record.sample[min_energy_idx][i])
                           for i, qubit in enumerate(forward_response.variables)}
            
            print(f"Forward annealing found solution with energy: {forward_energy:.6f}")
            
            # Use forward solution as initial state
            best_state = forward_state
            best_energy = forward_energy
            cycle_energies.append(forward_energy)  # Count forward annealing as first cycle

        final_response = None
        for cycle in range(num_cycles):
            # Set up reverse annealing parameters
            reverse_params = dict(
                anneal_schedule=self.anneal_schedule,
                initial_state=best_state,
                reinitialize_state=True,
                h_gain_schedule=self.h_gain_schedule
            )
            tries = 0
            success = False
                # Execute reverse annealing
            response = self.dw_sampler.sample_ising(
                h=h,
                J=J,
                num_reads=num_reads,
                anneal_schedule=self.anneal_schedule,
                initial_state=best_state,
                reinitialize_state=True,
                h_gain_schedule=self.h_gain_schedule,
                chain_break_fraction=False
            )
            final_response = response  # Keep track of last response
            #dwave.inspector.show(final_response)
            # Log access time
            try:
                if 'timing' in response.info and 'qpu_access_time' in response.info['timing']:
                    self._log_access_time(response.info['timing']['qpu_access_time'])
            except:
                self._log_access_time(400.0)

        
            # Find best solution from this cycle
            min_energy_idx = np.argmin(response.record.energy)
            cycle_min_energy = response.record.energy[min_energy_idx]
            cycle_energies.append(cycle_min_energy)
            
            print(f"Cycle {cycle + 1}/{num_cycles} - Minimum energy: {cycle_min_energy:.6f}")
            
            # Update best state if we found a better solution
            if cycle_min_energy < best_energy:
                best_energy = cycle_min_energy
                best_state = {qubit: int(response.record.sample[min_energy_idx][i])
                              for i,qubit in enumerate(response.variables)}
        
        # Save final results
        # Add ancilla suffix to directory name to separate results
        ancilla_suffix = '_with_ancilla' if use_ancilla_transformation else '_no_ancilla'
        
        if instance_type == 'dynamics':
            results_dir = ensure_dir(RESULT_DIR / str(self.sampler) / f'dynamics_{instance_id}_timepoints_{num_timepoints}{ancilla_suffix}_realization_1')
        else:
            results_dir = ensure_dir(RESULT_DIR / str(self.sampler) / f'N_{n_nodes}{ancilla_suffix}_realization_1')
        
        # Find next available file number
        existing_files = list(results_dir.glob('[0-9]*.npz'))
        next_number = 1 if not existing_files else max(int(re.findall('[0-9]*',f.stem)[0]) for f in existing_files) + 1
        results_path = results_dir / f'{next_number}.npz'
        
        # Add a suffix if forward initialization was used
        if use_forward_init:
            results_path = results_dir / f'{next_number}_forward_init.npz'
        
        if final_response is None:
            raise RuntimeError("No annealing cycles were completed")
            
        # Get last response info for metadata
        final_response_info = {
            'energies': final_response.record.energy,
            'solutions': final_response.record.sample,
            'num_occurrences': final_response.record.num_occurrences,
            'timing': final_response.info['timing']
        }
        
        # Save results with cyclic annealing information
        result_data = {
            **final_response_info,
            'anneal_schedule': self.anneal_schedule,
            'h_gain_schedule': self.h_gain_schedule,
            'cycle_energies': np.array(cycle_energies),
            'num_cycles': num_cycles,
            'best_state': best_state,
            'best_energy': best_energy,
            'used_forward_init': use_forward_init,
            'offset': offset,
            'instance_type': instance_type,
            'num_timepoints': num_timepoints if instance_type == 'dynamics' else None,
            'used_ancilla': problem['used_ancilla'],
            'ancilla_ratio': problem['ancilla_ratio'] if problem['used_ancilla'] else None
        }
        
        np.savez_compressed(results_path, **result_data)
        print(f"\nResults saved as: {results_path}")
        
        return final_response, result_data, cycle_energies
        
    def execute_instance(self, n_nodes: str | None = None, num_reads: int = 1000, instance_type: str = 'static', 
                        instance_id: str | None = None, num_timepoints: int = 5, use_ancilla_transformation: bool = False,
                        ancilla_ratio: int = 1):
        """Execute a single annealing run on a problem instance.
        
        Args:
            n_nodes (str, optional): Number of nodes to select specific static instance.
                If None, executes first available instance.
            num_reads (int, optional): Number of samples to collect. Defaults to 1000.
            instance_type (str): Either 'static' or 'dynamics'. Defaults to 'static'.
            instance_id (str, optional): ID of the dynamics instance (required if instance_type='dynamics').
            num_timepoints (int): Number of timepoints for dynamics instances. Defaults to 5.
            use_ancilla_transformation (bool): If True, apply ancilla transformation to dynamics instances.
            ancilla_ratio (int): Ratio for ancilla transformation (qubits per ancilla). Defaults to 1.
            
        Returns:
            tuple: (response object, instance_info)
        """
        # Load and prepare problem first to get instance details
        problem = self._load_and_prepare_problem(n_nodes, instance_type, instance_id, num_timepoints,
                                                  use_ancilla_transformation, ancilla_ratio)
        
        h = problem['h']
        J = problem['J']
        offset = problem['offset']
        n_nodes = problem['n_nodes']
        
        # Setup embedding (cached) for dynamics instances
        if instance_type == 'dynamics':
            self.dw_sampler = self._get_or_create_embedding(instance_type, instance_id, num_timepoints, n_nodes, h, J)


        #dwave.inspector.show(response)

        # Log access time
        success = False
        tries = 0
            
        response = self.dw_sampler.sample_ising(
            h=h,
            J=J,
            num_reads=num_reads,
            chain_break_fraction=False
        )
        try:
            if 'timing' in response.info and 'qpu_access_time' in response.info['timing']:
                self._log_access_time(response.info['timing']['qpu_access_time'])
        except:
            self._log_access_time(300.0)
        success = True
        """  except Exception as e:
                print(f"Retry {tries}")
                print(e)
                tries += 1
                if tries > 20:
                    raise Exception("Too many retries for sampling")
        """
        # Save results
        # Add ancilla suffix to directory name to separate results
        ancilla_suffix = '_with_ancilla' if use_ancilla_transformation else '_no_ancilla'
        
        if instance_type == 'dynamics':
            results_dir = ensure_dir(RESULT_DIR / 'forward' / str(self.sampler) / f'dynamics_{problem["instance_id"]}_timepoints_{problem["num_timepoints"]}{ancilla_suffix}')
        else:
            results_dir = ensure_dir(RESULT_DIR / 'forward' / str(self.sampler) / f'N_{n_nodes}{ancilla_suffix}_realization_1')
         
        # Find the next available file number
        existing_files = list(results_dir.glob('[0-9]*.npz'))
        if not existing_files:
            next_number = 1
        else:
            # Extract numbers from filenames and find the maximum
            numbers = [int(f.stem) for f in existing_files]
            next_number = max(numbers) + 1
            
        results_path = results_dir / f'{next_number}.npz'
        
        # Extract relevant information
        result_data = {
            'energies': response.record.energy,
            'solutions': response.record.sample,
            'num_occurrences': response.record.num_occurrences,
            'timing': response.info['timing'],
            'offset': offset,
            'instance_type': problem['instance_type'],
            'num_timepoints': problem['num_timepoints'],
            'used_ancilla': problem['used_ancilla']
        }
        
        # Save results
        np.savez_compressed(results_path, **result_data)
        print(f"Results saved as: {results_path}")
        
        return response, result_data
