from pathlib import Path
import json
import pickle
from typing import Any, Optional, Union
from dwdynamics import ComplexDynamicsProblem, Objective
from ..config import INSTANCES_DIR, ensure_dir


class BenchmarkInstance:
    def __init__(self, instance_id: int, number_time_points: int, 
                 objective: Any = Objective.hessian, 
                 base_path: Optional[Path] = None):
        """
        Initialize a benchmark instance.
        
        Args:
            instance_id: The ID of the instance
            number_time_points: Number of time poin ts to use
            objective: The objective function to use (hessian or norm)
            base_path: Optional custom path to instances directory
        """
        self.instance_id = instance_id
        self.objective = objective
        self.objective_path = 'norm' if objective == Objective.norm else 'hessian'
        self.number_time_points = number_time_points
        self.base_path = base_path if base_path else INSTANCES_DIR
        
        # Load data and create problem immediately
        file_path = self.base_path / f"{self.instance_id}.pckl"
        with file_path.open('rb') as f:
            instance_dict = pickle.load(f)
        self.H = instance_dict['H']
        self.psi0 = instance_dict['psi0']
        self.precision = instance_dict['precision']
        self.problem = ComplexDynamicsProblem(
            hamiltonian=self.H,
            initial_state=self.psi0,
            times=tuple(range(number_time_points)),
            num_bits_per_var=self.precision
        )
        self.qubo = self.problem.qubo(objective=self.objective)
        self.save_qubo()

    def save_qubo(self, save_path: Optional[Path] = None):
        """
        Save the QUBO to a JSON file.
        
        Args:
            save_path: Optional custom save path. If None, uses the default instances directory.
        """
        if self.qubo is None:
            raise ValueError("QUBO not created yet.")
            
        # Use provided path or construct default path
        save_dir = save_path if save_path else self.base_path
        qubo_path = save_dir / self.objective_path / str(self.instance_id)
        ensure_dir(qubo_path)
        
        file_path = qubo_path / f"precision_{self.precision}_timepoints_{self.number_time_points}.json"
        qubo_path.mkdir(exist_ok=True)
        with file_path.open('w') as f:
            json.dump(self.qubo.to_serializable(), f)
