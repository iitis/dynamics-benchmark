# Quantum Dynamics Benchmarking Framework

**Authors:** Philipp Hanussek, Jakub Pawłowski, Zakaria Mzaouali, Bartłomiej Gardas

This repository accompanies the paper *Solving quantum-inspired dynamics on quantum and classical computers*.

A framework for benchmarking different approaches to solving quantum dynamics problems, with a focus on comparing quantum annealing and classical methods.



## Overview

This project provides tools for:
- Benchmarking quantum dynamics solvers (DWave, Velox, etc.)
- Collecting and analyzing performance metrics
- Visualizing results and dynamics
- Comparing different solution approaches

## Installation

### Requirements
- Python 3.11+
- Poetry (recommended) or pip

It is recommended to use a python virtual environment.

### Solver access
To perform calculations, solver API access is required. See [D-Wave help pages](https://support.dwavesys.com/hc/en-us/articles/360003682634-How-Do-I-Get-an-API-Token).
### Using Poetry (recommended)
```bash
# Clone the repository
git clone https://github.com/atg205/dynamics-benchmark.git

# Install poetry if you haven't already
pip install poetry

# Install dependencies
cd dynamics-benchmark/benchmarker
poetry install

# For development (editable install)
pip install -e .
```

## Project Structure
```
dynamics-benchmark/
├── benchmarker/           # Main package
│   ├── core/             # Core functionality
│   │   ├── case.py       # Test case definitions
│   │   ├── instance.py   # Problem instances
│   │   ├── plotter.py    # Plotting utilities
│   │   ├── results.py    # Results handling
│   │   ├── runner.py     # Benchmark execution
│   │   └── save_utils.py # Data persistence
│   ├── scripts/          # Analysis & plotting scripts
│   │   ├── plot_dynamics.py        # System dynamics visualization
│   │   ├── plot_success_prob.py    # Success probability analysis
│   │   ├── plot_tta_comparison.py  # Time-to-answer comparison
│   │   ├── print_success_ratios.py # Success ratio statistics
│   │   ├── run_benchmark_tests.py  # Run benchmark test suite
│   │   └── run_velox.py           # Velox solver execution
│   └── main.py           # Entry point
├── data/                 # Benchmark data
└── plots/               # Generated visualizations
```

## Usage

### Running Benchmarks
```bash
# Run benchmark test suite
python -m benchmarker.scripts.run_benchmark_tests \
    --systems 1 2 3 \
    --samplers 1.6 \
    --annealing-times 100 200 \
    --num-reps 1000

# Run benchmarks with Velox solver
python -m benchmarker.scripts.run_velox --system 1 --timepoints 3

# Generate dynamics plots
python -m benchmarker.scripts.plot_dynamics --system 1

# Analyze success probabilities
python -m benchmarker.scripts.plot_success_prob_by_ta --system 1

# Compare time-to-answer metrics
python -m benchmarker.scripts.plot_tta_comparison

# Print success ratios
python -m benchmarker.scripts.print_success_ratios
```

Each script supports various command-line arguments. Use `--help` with any script to see available options.

## Data Storage

Results and data are organized as follows:
- `data/instances/`: Raw problem instances
- `data/results/<system_id>/<solver>/`: Benchmark results per system and solver
- `data/xubo/`: XUBO problem representations
- `plots/`: Generated visualizations and analysis results

## Documentation

For detailed API documentation and advanced usage, please refer to the docstrings in the source code. Each module and script is documented with examples and parameter descriptions.

## Related Projects

This project builds on [dwdynamics](dwdynamics/README.md), which provides the core functionality for quantum dynamics simulation. While dwdynamics focuses on the implementation of solving methods, this framework adds benchmarking capabilities and comprehensive result analysis.
