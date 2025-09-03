from pathlib import Path
import json
from typing import Dict, Any
from .results import BenchmarkResult
from ..config import HESSIAN_RESULTS_DIR, ensure_dir
import re

def get_last_index(files: list[str]) -> int:
    """
    Returns the highest index found in a list of filenames ending with .json.

    Args:
        files (list of str): List of filenames.

    Returns:
        int: Highest index found, or 0 if list is empty.
    """
    if not files:
        return 0
    return max([int(re.findall(r'\d+(?=\.json)', file)[0]) for file in files])



def save_benchmark_result(system: int, solver: str, precision: int, timepoints: int, result: BenchmarkResult) -> None:
    """
    Save benchmark result to the configured results directory.
    
    Args:
        system: system or instance id/name
        solver: solver id (e.g., '6.4', '1.6', 'velox')
        precision: precision value (int)
        timepoints: number of timepoints (int)
        result: BenchmarkResult instance to save
    """
    # Construct path using pathlib
    result_path = HESSIAN_RESULTS_DIR / str(system) / str(solver)
    ensure_dir(result_path)
    files = [str(file) for file in result_path.iterdir()]
    idx = get_last_index(files) +1   
    
    file_path = result_path / f"precision_{precision}_timepoints_{timepoints}_{idx}.json"
    
    # Save result as JSON
    with file_path.open("w") as f:
        json.dump(result.result.to_serializable(), f, indent=2)

    
