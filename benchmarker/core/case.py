from abc import ABC, abstractmethod
from typing import Any, Dict
from dwave.system import DWaveSampler, EmbeddingComposite
import neal
import qutip as qp
import numpy as np
from .results import BenchmarkResult
from .save_utils import save_benchmark_result
from .instance import BenchmarkInstance
from ..config import RESULTS_DIR, ensure_dir
import json
import dwave.inspector

class TestCase(ABC):
    """Abstract base class for all test cases.
    
    Defines the common interface and shared functionality for running benchmarks.
    
    Args:
        system (int): System ID to benchmark
        sampler (str): Solver to use for sampling
        timepoints (int): Number of time points to simulate
    """
    name: str
    system: int
    sampler: str
    precision: int
    timepoints: int

    def __init__(self, system: int, sampler: str, timepoints: int):
        self.system = system
        self.sampler = sampler
        self.timepoints = timepoints

    @abstractmethod
    def run(self) -> BenchmarkResult:
        pass

    def run_and_save(self) -> BenchmarkResult:
        result = self.run()
        save_benchmark_result(
            self.system,
            self.sampler,
            self.precision,
            self.timepoints,
            result
        )
        return result

class QuantumTestCase(TestCase):
    """Base class for quantum annealing test cases.
    
    This class implements quantum annealing-based benchmarking for both hardware
    and simulated quantum annealers.

    Args:
        system (int): System ID to benchmark
        sampler (str): Quantum annealer to use. Available options:
            - 'neal': Software-based simulated quantum annealing
            - '1.6': D-Wave Advantage2 system1.6 (Zephyr)
            - '6.4': D-Wave Advantage system6.4
        timepoints (int): Number of time points to simulate
        ta (int): Annealing time in microseconds
        num_reps(int): Number of samples 

    Example:
        >>> case = QuantumTestCase(
        ...     system=1,
        ...     sampler='neal',
        ...     timepoints=3,
        ...     ta=200
        ... )
        >>> result = case.run()
    """
    name = "Quantum Benchmark"

    def __init__(self, system: int, sampler: str, timepoints: int,ta:int, num_reps=1000):
        super().__init__(system, sampler, timepoints)
        self.ta =ta
        self.num_reps =num_reps

    def create_instance(self):
        """Create a benchmark instance for the current system."""
        return BenchmarkInstance(
            instance_id=int(self.system),
            number_time_points=self.timepoints,
        )
    
    def save_access_time(self, at: int):
        """
        Saves access time to benchmarker/data/time.json.

        Args:
            at (int): Access time in microseconds.

        Returns:
            None
        """
        time_path = RESULTS_DIR.parent / 'time.json'
        ensure_dir(time_path.parent)
        
        with time_path.open('r') as f:
            at_dict = json.load(f)
        at_dict['time_ms'] += at
        with time_path.open('w') as f:
            json.dump(at_dict, f)

    def sample_qubo(self, qubo):
        """Sample from QUBO using quantum hardware."""
        if self.sampler == 'neal':
            # If no sampler specified, use SimulatedAnnealingSampler
            sampler = neal.SimulatedAnnealingSampler()
            return sampler.sample(qubo, num_reads=self.num_reps)

        # Configure D-Wave sampler based on solver ID
        elif self.sampler == "1.8":  # zephyr
            dw_sampler = EmbeddingComposite(DWaveSampler(solver="Advantage2_system1.8"))
        elif self.sampler == "6.4":
            dw_sampler = EmbeddingComposite(DWaveSampler(solver="Advantage_system6.4"))
        else:
            raise ValueError(f"Invalid solver id: {self.sampler}")
        
        sampleset =  dw_sampler.sample(qubo, num_reads=self.num_reps, annealing_time=self.ta, return_embedding=True)
        dwave.inspector.show(sampleset)
        return sampleset
    def run(self) -> BenchmarkResult:
        """Common run logic for all quantum test cases."""
        instance = self.create_instance()
        sampleset = self.sample_qubo(instance.qubo)
        self.precision = instance.precision 
        if self.sampler == 'neal':
            computation_time = sampleset.info['timing']['sampling_ns'] * 1e-3
        else:
            computation_time = sampleset.info['timing']['qpu_access_time'] * 1e-3
            self.save_access_time(computation_time)


        return BenchmarkResult(
            result = sampleset,
            system = self.system,
            ta=self.ta,
            computation_time=computation_time
        ) 

class GroupTestCase(ABC):
    """Abstract base class for running benchmarks on groups of systems.
    
    Enables batch processing of multiple systems with the same configuration.
    
    Args:
        system_list (list): List of system IDs to benchmark
        num_timepoints (int): Number of time points to simulate
        sampler (str): Quantum annealer to use (see QuantumTestCase for options)
        ta (int, optional): Annealing time in microseconds. Defaults to 0.
    """
    def __init__(self, system_list: list, num_timepoints: int, sampler: str, ta: int = 0):
        self.system_list = system_list
        self.sampler = sampler
        self.num_timepoints = num_timepoints
        self.ta = ta

    def run_all_systems(self):
        """Run benchmarks for all native systems."""
        results = []
        for system_id in self.system_list:
            self.system = system_id
            TC = QuantumTestCase(self.system, self.sampler, self.num_timepoints, self.ta)
            results.append(TC.run_and_save())
        return results



class PegasusNativeSystemsCase(GroupTestCase):
    """Test case for native Pegasus systems.
    
    Handles benchmarking of systems that are natively embeddable on 
    the Pegasus architecture (systems 1, 2, 4, 5, 6, 7).
    """
    name = "Pegasus Native Benchmark"
    def __init__(self,num_timepoints: int, sampler: str, ta:int=0):
        native_systems = [1, 2, 4, 5, 6, 7]
        super().__init__(native_systems,num_timepoints,sampler,ta)



class NonNativeTestCase(QuantumTestCase):
    """Test case for non-native systems.
    
    Handles benchmarking of systems that require non-trivial embedding
    on quantum hardware (systems 3, 8).
    """
    name = "Non-Native Benchmark"
    non_native_systems = [3, 8]


    def get_system_type(self) -> str:
        return "non-native"

    def run_all_systems(self):
        """Run benchmarks for all non-native systems."""
        results = []
        for system_id in self.non_native_systems:
            self.system = system_id
            results.append(self.run_and_save())
        return results
