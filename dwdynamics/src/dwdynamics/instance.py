from dwdynamics import ComplexDynamicsProblem, Objective, helpers,draw_utils
import json
import os
from dwave.system import DWaveSampler, EmbeddingComposite
from dimod import BQM
import pandas as pd
from io import StringIO
import numpy as np
import subprocess
import pickle
import qutip as qp
import neal

class Instance:
    def __init__(
            self,
            instance_id: int,
            objective = Objective.hessian
    ):
        self._id = instance_id
        self.basepath ='../' if os.getcwd()[-9:] == 'notebooks' else '' # for execution in jupyter notebooks
        self.objective = objective
        self.objective_path = 'norm' if objective == Objective.norm else 'hessian'

    
    def create_instance(self, precision: int, number_time_points:int, save = False):
        """
        Creates an instance based on the provided parameters and saves the QUBO to /data/instances.

        Args:
            precision (int): Number of bits per variable.
            number_time_points (int): Number of time points.
            save (bool): If True, saves the QUBO to file.

        Returns:
            None
        """
        self.precision = precision
        self.number_time_points = number_time_points

        file_name = os.path.join(self.basepath,'data','instances', f"{self._id}.pckl")

        with open(file_name,'rb') as f:
            instance_dict = pickle.load(f)
        self.H = instance_dict['H']
        self.psi0 = instance_dict['psi0']

        self.problem = ComplexDynamicsProblem(
            hamiltonian= self.H, 
            initial_state = self.psi0,            
            times=tuple(range(number_time_points)),             
            num_bits_per_var=precision                
        )
        self.qubo = self.problem.qubo(objective=self.objective)           
        #assert self.qubo.num_variables == self.problem.hamiltonian.shape[0] * len(self.problem.times) * self.problem.num_bits_per_var * 2

        # save instances in the form 
        # systemid_{d}_precision_{d}_timepoints_{d}.json
        if save:
            path = f"data/instances/{self.objective_path}/{self._id}"

            file_name = os.path.join(self.basepath, path, f"precision_{precision}_timepoints_{number_time_points}.json")
            os.makedirs(path, exist_ok=True)
            with open(file_name,'w') as f:
                json.dump(self.qubo.to_serializable(),f)


    def generate_and_save_sampleset(self,solver_id="6.4", ta=200,num_samples=1000):
        """
        Runs 1000 samples on the indicated D-Wave machine and saves the result.

        Args:
            solver_id (str): D-Wave solver identifier.
            ta (int): Annealing time.

        Returns:
            dimod.SampleSet: Resulting sample set from the D-Wave run.
        """
        if solver_id == "5.4":
            dw_sampler = EmbeddingComposite(DWaveSampler( solver="Advantage_system5.4", region="eu-central-1", ))
        elif solver_id == "1.6": # zephyr
            dw_sampler = EmbeddingComposite(DWaveSampler( solver="Advantage2_system1.6"))
        elif solver_id == "6.4": 
            dw_sampler = EmbeddingComposite(DWaveSampler(solver="Advantage_system6.4"))
        elif solver_id == "neal":
            dw_sampler = neal.SimulatedAnnealingSampler()
        else:
            raise ValueError("Invalid solver id")


        if solver_id == 'neal':
            self.dw_result = dw_sampler.sample(self.qubo, num_reads=num_samples, annealing_time=ta)
        else:
            self.dw_result = dw_sampler.sample(self.qubo, num_reads=num_samples, annealing_time=ta, return_embedding=True)
            self.save_access_time(int(self.dw_result.info['timing']['qpu_access_time']))

        path = f"data/results/{self.objective_path}/{self._id}/{solver_id}"
        path = os.path.join(self.basepath, path)
        os.makedirs(path, exist_ok=True)
        idx = helpers.get_last_index(os.listdir(path)) +1

        file_name = os.path.join(path, f"precision_{self.precision}_timepoints_{self.number_time_points}_{idx}.json")

        with open(file_name,'w') as f:
            json.dump(self.dw_result.to_serializable(),f)
        return self.dw_result

    def to_xubo(self):
        """
        Convert a BQM to a xubo readable file and run the xubo script.

        Returns:
            None
        """
        bqm = self.qubo.spin
        # map linear terms
        lin_map = {}
        for i,key in enumerate(sorted(bqm.linear)):
            lin_map[key] = i
        
        output = [f'# QUBITS {len(bqm.linear)}\n']
        output += [f'# offset {bqm.offset}\n']
        output += [f'# quibitmap {lin_map}\n']
        output += [f'# vartype {bqm.vartype}\n']
        output += [f"{lin_map[k]} {lin_map[k]} {v}\n" for k, v in sorted(bqm.linear.items())]
        output += [f"{lin_map[k[0]]} {lin_map[k[1]]} {v}\n" for k, v in bqm.quadratic.items()]
        print(os.getcwd())
        path = f'data/xubo/ising/{self._id}/'
        os.makedirs(path,exist_ok=True)

        with open(os.path.join(self.basepath,path,f'precision_{self.precision}_timepoints_{self.number_time_points}.ising'), 'w') as f:
            f.writelines(output)
            f.close()   

        script_path = os.path.join(self.basepath, 'scripts/run_xubo.sh')
        subprocess.check_call(
            f"{script_path} %s %s %s" % (str(self._id), str(self.precision), str(self.number_time_points)),
            shell=True
)
        

    def get_xubo_df(self)->pd.DataFrame:
        """
        Load and parse the xubo output file for the current instance.

        Returns:
            pd.DataFrame: DataFrame containing energy and state information.
        """
        filename = f'precision_{self.precision}_timepoints_{self.number_time_points}.xubo'
        path = os.path.join(self.basepath, f'data/xubo/output/{self._id}', filename)
        if not os.path.exists(path):
            print("running xubo")
            self.to_xubo()
        content = ""
        i = 1
        with open(path, 'r') as f:
            line = f.readline()
            while line:
                line = f.readline()
                if i >= 17:
                    content += line
                i+=1
        assert content[0:6] == 'Energy', 'File does not have correct structure'
        return pd.read_csv(StringIO(content),sep=r'\s+',dtype={'Energy':np.float64,'State':'str'})

    def save_access_time(self, at: int):
        """
        Saves access time to file /data/time.json.

        Args:
            at (int): Access time in microseconds.

        Returns:
            None
        """
        with open(os.path.join(self.basepath, 'data','time.json'), 'r') as f:
            at_dict = json.load(f)
        at_dict['time_us'] += at
        with open(os.path.join(self.basepath, 'data', 'time.json'), 'r+') as f:
            json.dump(at_dict, f)

    def verify_sample(self, sample: str)->bool:
        """
        Verify a sample against the baseline quantum expectation values.

        Args:
            sample (str): Sample string.

        Returns:
            bool: True if sample matches baseline, False otherwise.
        """
        SZ = np.array([[1, 0], [0, -1]])
        exact_vec = self.problem.interpret_sample(sample)
        exact_expect = [(state.conj() @ SZ @ state).real for state in exact_vec]
        times = [i for i in range(self.number_time_points)]
        baseline = qp.mesolve(qp.Qobj(self.H), qp.basis(2, 0),times, e_ops=[qp.sigmaz()]).expect[0]
        return np.allclose(baseline, exact_expect)

    def get_qubo(self) -> BQM:
        """
        Get the QUBO for the current instance.

        Returns:
            BQM: Binary Quadratic Model (QUBO).
        """
        return self.qubo
    