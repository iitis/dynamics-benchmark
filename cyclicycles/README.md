# CyclicCycles

A quantum annealing framework for benchmarking **cyclic annealing** strategies on D-Wave quantum annealers.

## Quick Start

```python
from cyclicycles.runner import Runner

runner = Runner(sampler="1.10")
response, results, cycle_energies = runner.execute_cyclic_annealing(
    num_reads=1000,
    num_cycles=5,
    instance_type='dynamics',
    instance_id='1',
    num_timepoints=10
)
```

## Installation

```bash
poetry install
```

## Requirements

- Python ≥ 3.11
- D-Wave Ocean SDK ≥ 9.0.0
- NumPy, Pandas, Matplotlib


