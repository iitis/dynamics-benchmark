from cyclicycles.runner import Runner
from tqdm import tqdm
num_cycles = 30
instance_type = 'dynamics'
instances = [str(i) for i in range(3,4)]
import time

# Experiment with and without ancilla
for use_ancilla in [False]:
    print(f"\n\n========== Running with use_ancilla={use_ancilla} ==========\n")
    for timepoints in range(3,4):

        for instance in instances:
            for i in tqdm(range(1)):
                print("_________ instance __________")
                # Run cyclic annealing
                success =False
                while not success:
                    runner = Runner(sampler="1.10")
                    """
                    response, results, cycle_energies = runner.execute_cyclic_annealing(
                        num_reads=1000,
                        num_cycles=num_cycles,
                        use_forward_init=True,
                        instance_type='dynamics',
                        instance_id=instance,
                        num_timepoints=timepoints,
                        use_ancilla_transformation=use_ancilla
                    )
                    print(f"Energy progression: {cycle_energies}")
                    print(f"Best energy found: {results['best_energy']}")
                    """
                    # Run forward annealing
                    for _ in range(7):

                        response, results = runner.execute_instance(
                            instance_type="dynamics",
                            instance_id=instance,
                            num_reads=1000,
                            num_timepoints=timepoints,
                            use_ancilla_transformation=False
                        )
                        print(f"Best energy found (forward annealing): {results['energies'][0]}")
                    success=True
                    """
                       except Exception as e:
                        print("Error")
                        print(e)
                    """

                        
