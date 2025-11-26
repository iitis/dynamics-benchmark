from benchmarker.core.instance import BenchmarkInstance

from memory_profiler import profile 

@profile
def create_instance(tp):
    test_cases=[
        BenchmarkInstance(
            instance_id=system,
            number_time_points=tp,
        ) 
        for system in range(1,2) 
    ]


## Change here 
num_timepoints = [2000]
###


for i in num_timepoints:
    create_instance(i)

print("Done")