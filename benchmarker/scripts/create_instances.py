from benchmarker.core.instance import BenchmarkInstance
test_cases=[
    BenchmarkInstance(
        instance_id=system,
        number_time_points=timepoints,
    ) 
    for system in range(1,9) 
    for timepoints in [1000,10000]
]