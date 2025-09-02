from typing import Dict, List, Optional
from pathlib import Path
import json
import pandas as pd
from .results import BenchmarkResult
import numpy as np
import dimod
from collections import defaultdict
import re
import math
from scipy.stats import linregress
from ..config import HESSIAN_RESULTS_DIR, ensure_dir


class ResultsLoader:
    def __init__(self, base_path: Optional[Path] = None):
        """
        Initialize the ResultsLoader.
        
        Args:
            base_path: Optional custom base path. If None, uses default from config.
        """
        self.base_path = base_path if base_path else HESSIAN_RESULTS_DIR
    
    def load_result(self, system: int, solver: str, precision: int, timepoints: int) -> Optional[BenchmarkResult]:
        """Load a specific benchmark result"""
        path = self.base_path / str(system) / str(solver) / f"precision_{precision}_timepoints_{timepoints}.json"
        if not path.exists():
            return None
            
        with open(path, 'r') as f:
            data = json.load(f)
            return BenchmarkResult(
                result=pd.DataFrame(data['result']),
                system=data['system'],
                ta=data['ta'],
                computation_time=data['computation_time']
            )
    
    def load_all_results(self, system: int) -> Dict[str, List[BenchmarkResult]]:
        """Load all results for a given system"""
        results = {}
        system_path = self.base_path / str(system)
        
        if not system_path.exists():
            return results
            
        for solver in system_path.iterdir():
            if not solver.is_dir():
                continue
                
            results[solver.name] = []
            
            for result_file in solver.glob('*.json'):
                with result_file.open('r') as f:
                    data = json.load(f)
                    results[solver.name].append(BenchmarkResult(
                        result=pd.DataFrame(data['result']),
                        system=data['system'],
                        ta=data['ta'],
                        computation_time=data['computation_time']
                    ))
                        
        return results
    

    def return_tts(self,p_success: float,t:float, p_target=0.99)->float:
        """
        Calculate the time-to-solution (TTS) for a given success probability.

        Args:
            p_success (float): Success probability per run.
            t (float): Time per run.
            p_target (float): Target cumulative success probability (default 0.99).

        Returns:
            float: Estimated time to reach target success probability.
        """
        if p_success == 0:
            return np.inf
        if p_success == 1:
            return t
        return (math.log(1-p_target) / math.log(1-p_success))*t


    def get_dwave_tts(self, system: int, topology: str = "6.4", file_limit: float = np.inf, num_reps=0,ta=0) -> pd.DataFrame:
        """Get D-Wave time-to-solution data for a specific system and topology."""
        path = self.base_path / str(system) / topology
        df_dict = defaultdict(list)
        file_counter = defaultdict(int)

        if not path.exists():
            return pd.DataFrame()

        for file_path in path.glob('*.json'):
           
            with file_path.open('r') as f:
                s = dimod.SampleSet.from_serializable(json.load(f))
            if num_reps > 0 and num_reps != s.to_pandas_dataframe()['num_occurrences'].sum():
                continue
          
            # Append Metadata
            if topology == 'neal':
                qpu_access_time = s.info['timing']['sampling_ns'] * 1e-6
                annealing_time = 0

            else:
                qpu_access_time = s.info['timing']['qpu_access_time'] * 1e-3
                annealing_time = s.info['timing']['qpu_anneal_time_per_sample']

            precision = int(re.findall(r'(?<=precision_)\d+', file_path.name)[0])
            timepoints = int(re.findall(r'(?<=timepoints_)\d+', file_path.name)[0])
            if file_counter[(system, timepoints)] >= file_limit or annealing_time != ta:
                continue
            file_counter[(system,timepoints)] += 1
            
            df_dict['runtime'].append(qpu_access_time)
            df_dict['ta'].append(annealing_time)
            df_dict['precision'].append(precision)
            df_dict['timepoints'].append(timepoints)
            #if topology=='1.4':
             #   num_vars = sum([len(value) for value in s.info['embedding_context']['embedding'].values()]) 
              #  df_dict['num_var'].append(num_vars)

            #else:
                #df_dict['num_var'].append(len(s.variables))
            df_dict['num_var'].append(len(s.variables))
            sampleset = s.to_pandas_dataframe()
            sampleset['energy'] = round(sampleset['energy'],14)
            if len(sampleset[sampleset.energy== 0]) == 0:
                success_rate = 0.0
            else:
                success_rate = int(sampleset[sampleset.energy == 0]['num_occurrences'].sum())
            success_rate /= sampleset['num_occurrences'].sum()
            
            df_dict['success'].append(success_rate)
        
        df = pd.DataFrame.from_dict(df_dict)
        if df.empty:
            raise ValueError('no data found')

        #df =df.groupby(['ta','precision','timepoints','num_var']).agg(
         #   success_prob=('success','mean'),
         #   runtime=('runtime','mean'),
        #).reset_index()

        df = df[['precision','timepoints','num_var','success','runtime']].groupby(['precision','timepoints','num_var']).mean().reset_index()
        df['tts99'] = df.apply(lambda row: self.return_tts(row['success'],row.runtime),axis=1)

        df['source'] = topology
        df['system'] = system
        return df
    

    def get_velox_results(self, system: int,timepoints = None) -> pd.DataFrame:
        """
        Load Velox results for a given system from CSV and parse relevant fields.

        Args:
            system: System identifier

        Returns:
            DataFrame containing parsed results
        """
        path = self.base_path / str(system) / 'velox' / f'best_results_hessian_{system}_native.csv'
        
        if not path.exists():
            print("path does not exist")
            return pd.DataFrame()
        
        df = pd.read_csv(path)
        df_dict = defaultdict(list)
        
        # Process each row
        for row in df.itertuples():
            # Extract precision and timepoints using regex
            precision, row_timepoints = re.findall(r'\d+', str(row.instance))
            df_dict['precision'].append(int(precision))
            df_dict['timepoints'].append(int(row_timepoints))
            
            # Convert and append other fields
            df_dict['num_steps'].append(int(str(row.num_steps)))
            df_dict['runtime'].append(float(str(row.runtime)) * 1e3)
            df_dict['gap'].append(float(str(row.gap)))
            df_dict['num_rep'].append(int(str(row.num_rep)))
            df_dict['success_prob'].append(float(str(row.success_prob)))
            df_dict['solution'].append(str(row.best_solution).replace("-1", "0").replace(';', ''))
            df_dict['num_var'].append(int(str(row.num_var)))
        
        df = pd.DataFrame(df_dict)
        if timepoints is not None:
            df = df[df.timepoints == timepoints]
            return df
        return df

    def get_velox_tts(self,system:int)->pd.DataFrame:
        """
        Compute Velox success rates for a given system.

        Args:
            system (int): System identifier.

        Returns:
            pd.DataFrame: DataFrame containing aggregated success rates and runtimes.
        """
        df = self.get_velox_results(system=system)
        df['tts99'] = df.apply(lambda row: self.return_tts(row['success_prob'],row.runtime),axis=1)
        df = df[['precision','timepoints','num_var','tts99']].groupby(['precision','timepoints','num_var']).min().reset_index()
        df['system'] = system
        df['source'] = 'VELOX'
        return df
    

    def get_dwave_success_rates(self, system: int, topology: str = "6.4", 
                               ta: int = 200, grouped: bool = True, 
                               file_limit: float = np.inf) -> pd.DataFrame:
        """
        Load D-Wave success rates for a given system and topology.

        Args:
            system: System identifier
            topology: Topology string
            ta: Annealing time
            grouped: If True, group results by precision, timepoints, topology
            file_limit: Maximum number of files to process

        Returns:
            DataFrame containing success rates and related info
        """
        path = self.base_path / str(system) / topology
        
        if not path.exists():
            return pd.DataFrame()
            
        df_dict = defaultdict(list)
        file_counter = 0

        for file_path in path.glob('*.json'):
            if file_counter >= file_limit:
                break
                
            try:
                with file_path.open('r') as f:
                    sample_set = dimod.SampleSet.from_serializable(json.load(f))
            except Exception:
                continue
                
            # Handle different timing information based on solver
            if topology == 'neal':
                qpu_access_time = sample_set.info['timing']['sampling_ns']
                annealing_time = 0
            else:
                qpu_access_time = sample_set.info['timing']['qpu_access_time']
                annealing_time = sample_set.info['timing']['qpu_anneal_time_per_sample']
                if not ta == annealing_time:
                    continue
            
            file_counter += 1
            
            # Extract metadata from filename
            precision = int(re.findall(r'(?<=precision_)\d+', file_path.name)[0])
            timepoints = int(re.findall(r'(?<=timepoints_)\d+', file_path.name)[0])
            
            # Process sample set data
            df = sample_set.to_pandas_dataframe()
            df['energy'] = abs(round(df['energy'], 10))
            df = df[['energy', 'num_occurrences']].groupby('energy').sum().reset_index()
            
            # Calculate success rate
            total_occurrences = df['num_occurrences'].sum()
            success_occurrences = df[df.energy == 0]['num_occurrences'].iloc[0] if len(df[df.energy == 0]) > 0 else 0
            success_rate = float(success_occurrences) / total_occurrences
            
            # Calculate access time per sample
            access_time = qpu_access_time / total_occurrences * 1e-3
            
            # Store results
            df_dict['precision'].append(precision)
            df_dict['timepoints'].append(timepoints)
            df_dict['topology'].append(topology)
            df_dict['success_prob'].append(success_rate)
            df_dict['runtime'].append(access_time)
            df_dict['num_var'].append(len(sample_set.variables))
            
        # Convert results to DataFrame
        results_df = pd.DataFrame(df_dict)
        
        if results_df.empty:
            return results_df
            
        # Group results if requested
        if grouped:
            results_df = results_df.groupby(['precision', 'timepoints', 'topology']).mean().reset_index()
            
        return results_df
    

    def get_velox_sample_set(self, system: int, timepoints: int):
        path = self.base_path / str(system) / f'new_best_results_hessian_{system}_native.csv'
        df = pd.read_csv(path)
        df_dict= defaultdict(list)
        for row in df.itertuples():
            _, row_timepoints = re.findall(r'\d+',str(row.instance))
            if timepoints != row_timepoints:
                continue 
            df_dict['gap'].append(float(row.gap))
            df_dict['solution'].append(row.best_solution.replace("-1","0").replace(';',''))
        return pd.DataFrame(df_dict).sort_values('gap',ascending=True).drop_duplicates()


    def get_dwave_sample_set(self, system: int, timepoints: int, topology: str = "1.4") -> dimod.SampleSet:
        """
        Get the best D-Wave sample set for a given system configuration.

        Args:
            system: System identifier
            timepoints: Number of timepoints
            topology: D-Wave topology to use

        Returns:
            The best sample set found (lowest energy)

        Raises:
            ValueError: If no samples are found for the given configuration
        """
        path = self.base_path / str(system) / topology
        
        if not path.exists():
            raise ValueError(f"Path does not exist: {path}")
            
        min_energy = np.inf
        best_sample = None
        
        for file_path in path.glob('*.json'):
            file_timepoints = int(re.findall(r'(?<=timepoints_)\d+', file_path.name)[0])
            if file_timepoints != timepoints:
                continue

            with file_path.open('r') as f:
                sample_set = dimod.SampleSet.from_serializable(json.load(f))
            
            # Convert to dataframe to get first sample's energy
            first_sample = sample_set.to_pandas_dataframe().iloc[0]
            sample_energy = abs(first_sample['energy'])
            
            # If we find a solution with zero energy, return it immediately
            if sample_energy < 1e-12: 
                return sample_set
                
            # Otherwise, keep track of the lowest energy solution
            if sample_energy < min_energy:
                best_sample = sample_set
                min_energy = sample_energy
        
        if best_sample is None:
            raise ValueError(f'No samples found for system={system}, timepoints={timepoints}, topology={topology}')
            
        return best_sample
    

    def result_string_to_dict(self, input_string:str)->dict[int,int]:
        """
        Convert a string of bits to a dictionary mapping index to bit value.

        Args:
            input_string (str): String of bits (e.g., '1010').

        Returns:
            dict: Dictionary mapping index to bit value.
        """
        return {i:int(bit) for i,bit in enumerate(list(input_string))}
