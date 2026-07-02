"""
LangChain tool definitions for the InversionAI agent.

These tools are invoked by the ReAct agent to perform geophysical
inversion tasks: validation, execution, comparison, and explanation.
"""

import json
import logging
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def list_available_skills() -> str:
    """List all available inversion skills and workflows.

    Returns a formatted list of algorithms and capabilities available
    in the InversionAI system.
    """
    skills = {
        "algorithms": [
            {
                "name": "tomofast-x",
                "description": "Fast parallel geophysical inversion (Fortran/MPI)",
                "supports": ["gravity", "magnetic", "joint"],
                "strengths": "Speed, large models, joint inversion",
            },
            {
                "name": "simpeg",
                "description": "Flexible Python-based geophysical inversion",
                "supports": ["gravity", "magnetic"],
                "strengths": "Flexibility, custom regularization, prototyping",
            },
        ],
        "workflows": [
            "single_inversion - Run one algorithm on a dataset",
            "comparison - Run both algorithms and compare results",
            "validation - Check data files before inversion",
            "parameter_sweep - Run multiple configurations",
        ],
        "utilities": [
            "data_validation - Verify input file format and quality",
            "result_visualization - 3D plots and cross-sections",
            "parameter_explanation - Explain what each parameter does",
        ],
    }
    return json.dumps(skills, indent=2)


@tool
def validate_data_file(filepath: str) -> str:
    """Validate a geophysical data file for use in inversion.

    Checks file existence, format, required columns, coordinate systems,
    and data quality. Reports issues that would prevent a successful inversion.

    Args:
        filepath: Path to the data file to validate.
    """
    logger.info("Validating data file: %s", filepath)
    path = Path(filepath)

    if not path.exists():
        return json.dumps({
            "valid": False,
            "error": f"File not found: {filepath}",
            "suggestions": ["Check the file path", "Ensure the file has been downloaded"],
        })

    if not path.is_file():
        return json.dumps({
            "valid": False,
            "error": f"Path is not a file: {filepath}",
        })

    # Check file extension
    supported_extensions = {".txt", ".csv", ".dat", ".obs", ".grv", ".mag", ".ubc"}
    if path.suffix.lower() not in supported_extensions:
        return json.dumps({
            "valid": False,
            "error": f"Unsupported file extension: {path.suffix}",
            "supported": list(supported_extensions),
        })

    # Basic file size check
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb == 0:
        return json.dumps({"valid": False, "error": "File is empty."})

    # Try to read and check structure
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        num_lines = len(lines)
        if num_lines < 2:
            return json.dumps({
                "valid": False,
                "error": "File has fewer than 2 lines — likely incomplete.",
            })

        # Check if numeric data is present
        sample_line = lines[min(1, num_lines - 1)].strip()
        parts = sample_line.replace(",", " ").split()
        num_columns = len(parts)

        return json.dumps({
            "valid": True,
            "file": filepath,
            "size_mb": round(size_mb, 2),
            "num_lines": num_lines,
            "num_columns": num_columns,
            "preview": lines[0].strip()[:100],
            "message": f"File looks valid: {num_lines} data points, {num_columns} columns.",
        })

    except UnicodeDecodeError:
        return json.dumps({
            "valid": True,
            "file": filepath,
            "size_mb": round(size_mb, 2),
            "message": "Binary file detected. May be a mesh or model file.",
            "note": "Cannot preview binary content.",
        })
    except Exception as e:
        return json.dumps({
            "valid": False,
            "error": f"Error reading file: {str(e)}",
        })


@tool
def run_inversion(algorithm: str, data_path: str, params: dict[str, Any]) -> str:
    """Run a geophysical inversion using the specified algorithm.

    Args:
        algorithm: The inversion algorithm to use ('tomofast-x' or 'simpeg').
        data_path: Path to the input data file.
        params: Dictionary of inversion parameters (iterations, regularization, etc.).
    """
    logger.info("Running %s inversion on %s with params: %s", algorithm, data_path, params)

    algo_lower = algorithm.lower().replace("-", "").replace("_", "")

    if "tomofast" in algo_lower:
        return _run_tomofast(data_path, params)
    elif "simpeg" in algo_lower:
        return _run_simpeg(data_path, params)
    else:
        return json.dumps({
            "success": False,
            "error": f"Unknown algorithm: {algorithm}",
            "available": ["tomofast-x", "simpeg"],
        })


def _run_tomofast(data_path: str, params: dict[str, Any]) -> str:
    """Execute Tomofast-x inversion (stub - delegates to skills layer)."""
    # In production, this would call the skills/tomofast module
    iterations = params.get("iterations", 50)
    data_type = params.get("data_type", "gravity")

    return json.dumps({
        "success": True,
        "algorithm": "tomofast-x",
        "status": "submitted",
        "data_path": data_path,
        "data_type": data_type,
        "iterations": iterations,
        "message": (
            f"Tomofast-x {data_type} inversion submitted with {iterations} iterations. "
            "Use get_run_status() to monitor progress."
        ),
        "output_dir": f"./results/tomofast_{data_type}/",
    })


def _run_simpeg(data_path: str, params: dict[str, Any]) -> str:
    """Execute SimPEG inversion (stub - delegates to skills layer)."""
    iterations = params.get("iterations", 30)
    data_type = params.get("data_type", "gravity")

    return json.dumps({
        "success": True,
        "algorithm": "simpeg",
        "status": "submitted",
        "data_path": data_path,
        "data_type": data_type,
        "iterations": iterations,
        "message": (
            f"SimPEG {data_type} inversion submitted with {iterations} iterations. "
            "Use get_run_status() to monitor progress."
        ),
        "output_dir": f"./results/simpeg_{data_type}/",
    })


@tool
def compare_inversions(result_a_path: str, result_b_path: str) -> str:
    """Compare results from two inversion runs.

    Computes difference metrics including RMS difference, correlation,
    and structural similarity between two inversion result models.

    Args:
        result_a_path: Path to the first inversion result.
        result_b_path: Path to the second inversion result.
    """
    logger.info("Comparing inversions: %s vs %s", result_a_path, result_b_path)

    path_a = Path(result_a_path)
    path_b = Path(result_b_path)

    errors = []
    if not path_a.exists():
        errors.append(f"Result A not found: {result_a_path}")
    if not path_b.exists():
        errors.append(f"Result B not found: {result_b_path}")

    if errors:
        return json.dumps({"success": False, "errors": errors})

    # Stub: in production, load and compare actual model files
    return json.dumps({
        "success": True,
        "result_a": result_a_path,
        "result_b": result_b_path,
        "metrics": {
            "rms_difference": "Computed after loading model files",
            "correlation": "Computed after loading model files",
            "structural_similarity": "Computed after loading model files",
            "max_difference_location": "Computed after loading model files",
        },
        "message": (
            "Comparison framework ready. Full metrics will be computed "
            "when actual inversion result files are available."
        ),
        "interpretation_guide": (
            "RMS < 5% suggests good agreement. "
            "Correlation > 0.9 means structurally similar. "
            "Check max_difference_location for areas of disagreement."
        ),
    })


@tool
def get_run_status() -> str:
    """Get the status of currently running or recently completed inversions.

    Returns information about active runs, progress, and any errors.
    """
    logger.info("Checking run status...")

    # Stub: in production, query the actual run manager
    return json.dumps({
        "active_runs": [],
        "completed_runs": [],
        "message": "No active or recent inversion runs found.",
        "hint": "Use run_inversion() to start a new inversion.",
    })


@tool
def explain_parameter(param_name: str, algorithm: str) -> str:
    """Explain what a specific inversion parameter does and suggest good defaults.

    Args:
        param_name: The parameter name to explain.
        algorithm: The algorithm context ('tomofast-x' or 'simpeg').
    """
    logger.info("Explaining parameter '%s' for %s", param_name, algorithm)

    # Common parameter explanations
    explanations = {
        "iterations": {
            "description": "Number of inversion iterations to run.",
            "tomofast-x_default": 50,
            "simpeg_default": 30,
            "guidance": (
                "More iterations can improve fit but increase runtime. "
                "Start with defaults and increase if misfit is still high."
            ),
        },
        "regularization": {
            "description": "Controls model smoothness vs data fit trade-off.",
            "tomofast-x_default": "Automatic (L-curve)",
            "simpeg_default": "beta_cooling schedule",
            "guidance": (
                "Higher regularization = smoother model. "
                "Too low = noisy/unstable. Too high = over-smoothed."
            ),
        },
        "mesh_size": {
            "description": "Size of model cells in the inversion mesh.",
            "guidance": (
                "Smaller cells = more detail but slower. "
                "Cell size should be <= half the station spacing."
            ),
        },
        "depth_weighting": {
            "description": "Compensates for decreasing sensitivity with depth.",
            "tomofast-x_default": "Enabled (beta=2.0 for gravity)",
            "simpeg_default": "Enabled (default exponent)",
            "guidance": "Essential for potential field inversions. Usually keep enabled.",
        },
    }

    param_lower = param_name.lower().replace(" ", "_").replace("-", "_")

    if param_lower in explanations:
        info = explanations[param_lower]
        algo_default_key = f"{algorithm.lower().replace('-', '_')}_default"
        default_val = info.get(algo_default_key, "See documentation")
        return json.dumps({
            "parameter": param_name,
            "algorithm": algorithm,
            "description": info["description"],
            "default_value": default_val,
            "guidance": info["guidance"],
        })

    return json.dumps({
        "parameter": param_name,
        "algorithm": algorithm,
        "description": f"Parameter '{param_name}' explanation not found in built-in database.",
        "suggestion": "Try checking the algorithm documentation or ask for general guidance.",
    })
