"""
Tomofast-x output parser.

Parses stdout from Tomofast-x execution to extract iteration progress,
and parses output model files (VTK, text) into structured data.
"""

import logging
import re
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Regex patterns for Tomofast-x stdout parsing
# Typical output lines:
#   "Iteration    10, data misfit = 1.234567E+02"
#   "iter=   5  misfit=  0.12345E+03"
ITERATION_PATTERNS = [
    re.compile(r"[Ii]teration\s+(\d+).*misfit\s*=\s*([0-9.]+[Ee][+-]?\d+)"),
    re.compile(r"[Ii]teration\s+(\d+).*misfit\s*=\s*([0-9.]+)"),
    re.compile(r"iter\s*=?\s*(\d+).*misfit\s*=?\s*([0-9.]+[Ee][+-]?\d+)"),
    re.compile(r"iter\s*=?\s*(\d+).*misfit\s*=?\s*([0-9.]+)"),
    re.compile(r"(\d+)\s+([0-9.]+[Ee][+-]?\d+)\s+.*[Cc]onverg"),
]

# Pattern for completion
COMPLETION_PATTERN = re.compile(
    r"[Ii]nversion\s+completed|[Ff]inished|[Dd]one|SUCCESS"
)

# Pattern for errors
ERROR_PATTERN = re.compile(
    r"ERROR|FATAL|[Ff]ailed|[Aa]bort"
)


def parse_stdout_progress(line: str) -> Optional[tuple[int, float]]:
    """Parse a single stdout line for iteration progress.

    Args:
        line: A single line from Tomofast-x stdout.

    Returns:
        Tuple of (iteration_number, misfit_value) if found, None otherwise.
    """
    for pattern in ITERATION_PATTERNS:
        match = pattern.search(line)
        if match:
            try:
                iteration = int(match.group(1))
                misfit = float(match.group(2))
                return (iteration, misfit)
            except (ValueError, IndexError):
                continue
    return None


def parse_stdout_full(stdout: str) -> dict[str, Any]:
    """Parse complete stdout from a Tomofast-x run.

    Args:
        stdout: Full stdout text from the process.

    Returns:
        Dictionary with keys:
            - 'iterations': list of (iteration, misfit) tuples
            - 'completed': bool
            - 'errors': list of error lines
            - 'final_misfit': float or None
    """
    result: dict[str, Any] = {
        "iterations": [],
        "completed": False,
        "errors": [],
        "final_misfit": None,
    }

    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue

        # Check for progress
        progress = parse_stdout_progress(line)
        if progress:
            result["iterations"].append(progress)

        # Check for completion
        if COMPLETION_PATTERN.search(line):
            result["completed"] = True

        # Check for errors
        if ERROR_PATTERN.search(line):
            result["errors"].append(line)

    # Extract final misfit
    if result["iterations"]:
        result["final_misfit"] = result["iterations"][-1][1]

    return result


def parse_output_model(file_path: str) -> dict[str, Any]:
    """Parse a Tomofast-x output model file.

    Supports text-based model files and misfit history files.
    For VTK files, returns file metadata only (use VTK library for full parsing).

    Args:
        file_path: Path to the output file.

    Returns:
        Dictionary with parsed data. Keys depend on file type:
            - For misfit files: {'misfit_history': list[float]}
            - For model files: {'values': list[float], 'n_cells': int}
            - For VTK files: {'format': 'vtk', 'path': str, 'size_bytes': int}
    """
    path = Path(file_path)
    if not path.exists():
        logger.error(f"Output file not found: {file_path}")
        return {}

    suffix = path.suffix.lower()

    if suffix == ".vtk":
        return _parse_vtk_metadata(path)
    elif "misfit" in path.name.lower():
        return _parse_misfit_file(path)
    else:
        return _parse_model_text(path)


def _parse_vtk_metadata(path: Path) -> dict[str, Any]:
    """Extract metadata from a VTK file without full parsing.

    Args:
        path: Path to VTK file.

    Returns:
        Dictionary with VTK file metadata.
    """
    result: dict[str, Any] = {
        "format": "vtk",
        "path": str(path),
        "size_bytes": path.stat().st_size,
    }

    try:
        with open(path, "r", errors="replace") as f:
            # Read first few lines for header info
            header_lines = []
            for i, line in enumerate(f):
                if i >= 20:
                    break
                header_lines.append(line.strip())

        # Extract dataset type
        for line in header_lines:
            if line.startswith("DATASET"):
                result["dataset_type"] = line.split()[-1] if len(line.split()) > 1 else "UNKNOWN"
            elif line.startswith("DIMENSIONS"):
                parts = line.split()
                if len(parts) == 4:
                    result["dimensions"] = [int(x) for x in parts[1:4]]
            elif line.startswith("POINTS"):
                parts = line.split()
                if len(parts) >= 2:
                    result["n_points"] = int(parts[1])
            elif line.startswith("CELLS") or line.startswith("CELL_DATA"):
                parts = line.split()
                if len(parts) >= 2:
                    result["n_cells"] = int(parts[1])

    except IOError as e:
        logger.error(f"Error reading VTK file: {e}")

    return result


def _parse_misfit_file(path: Path) -> dict[str, Any]:
    """Parse a misfit history file.

    Expected format: one misfit value per line, or iteration-misfit pairs.

    Args:
        path: Path to misfit file.

    Returns:
        Dictionary with 'misfit_history' list.
    """
    misfit_history: list[float] = []

    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(("#", "!", "%")):
                    continue

                parts = line.split()
                try:
                    if len(parts) == 1:
                        misfit_history.append(float(parts[0]))
                    elif len(parts) >= 2:
                        # Assume second column is misfit
                        misfit_history.append(float(parts[1]))
                except ValueError:
                    continue

    except IOError as e:
        logger.error(f"Error reading misfit file: {e}")

    return {"misfit_history": misfit_history}


def _parse_model_text(path: Path) -> dict[str, Any]:
    """Parse a text-based model file.

    Expected format: one value per line (cell values), or multi-column
    with coordinates and values.

    Args:
        path: Path to model text file.

    Returns:
        Dictionary with model values and cell count.
    """
    values: list[float] = []

    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(("#", "!", "%")):
                    continue

                parts = line.split()
                try:
                    if len(parts) == 1:
                        values.append(float(parts[0]))
                    elif len(parts) >= 4:
                        # Assume last column is model value (x, y, z, value)
                        values.append(float(parts[-1]))
                    else:
                        values.append(float(parts[-1]))
                except ValueError:
                    continue

    except IOError as e:
        logger.error(f"Error reading model file: {e}")

    return {
        "values": values,
        "n_cells": len(values),
        "min_value": min(values) if values else 0.0,
        "max_value": max(values) if values else 0.0,
        "mean_value": sum(values) / len(values) if values else 0.0,
    }
