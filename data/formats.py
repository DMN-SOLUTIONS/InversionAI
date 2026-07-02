"""
Supported geophysical data format definitions and auto-detection.

Provides format specifications, enumerations, and detection logic for common
geophysical data formats used in potential field inversion workflows.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional


class DataFormat(Enum):
    """Enumeration of supported geophysical data formats."""

    CSV = "csv"
    UBC_MESH = "ubc_mesh"
    UBC_MODEL = "ubc_model"
    UBC_OBSERVATION = "ubc_observation"
    GIF_GRAVITY = "gif_gravity"
    GIF_MAGNETIC = "gif_magnetic"
    GEOSOFT_XYZ = "geosoft_xyz"
    TOMOFAST_DATA = "tomofast_data"
    TOMOFAST_MESH = "tomofast_mesh"
    TOMOFAST_MODEL = "tomofast_model"
    UNKNOWN = "unknown"


@dataclass
class FormatSpec:
    """Specification for a geophysical data format.

    Attributes:
        name: Human-readable format name.
        extension: Common file extension(s).
        required_columns: Columns that must be present.
        optional_columns: Columns that may be present.
        description: Brief description of the format.
    """

    name: str
    extension: List[str]
    required_columns: List[str]
    optional_columns: List[str] = field(default_factory=list)
    description: str = ""


# Format specifications registry
FORMAT_SPECS: dict[DataFormat, FormatSpec] = {
    DataFormat.CSV: FormatSpec(
        name="CSV (x, y, z, value)",
        extension=[".csv", ".txt", ".dat"],
        required_columns=["x", "y", "z"],
        optional_columns=["grav_anomaly_mGal", "mag_anomaly_nT", "uncertainty", "value"],
        description="Generic comma/tab-separated file with station coordinates and observations.",
    ),
    DataFormat.UBC_MESH: FormatSpec(
        name="UBC Mesh",
        extension=[".msh", ".txt"],
        required_columns=[],
        optional_columns=[],
        description="UBC-GIF 3D tensor mesh format. Line 1: NE NE NE, Line 2: E0 N0 Z0, "
        "Lines 3-5: cell sizes in each direction.",
    ),
    DataFormat.UBC_MODEL: FormatSpec(
        name="UBC Model",
        extension=[".mod", ".txt"],
        required_columns=[],
        optional_columns=[],
        description="UBC-GIF model file. One value per cell, ordered X-fastest.",
    ),
    DataFormat.UBC_OBSERVATION: FormatSpec(
        name="UBC Observation",
        extension=[".obs", ".loc", ".txt"],
        required_columns=[],
        optional_columns=[],
        description="UBC-GIF observation file with station locations and data.",
    ),
    DataFormat.GIF_GRAVITY: FormatSpec(
        name="GIF Gravity",
        extension=[".grv", ".obs"],
        required_columns=["x", "y", "z", "grav"],
        optional_columns=["std"],
        description="UBC-GIF gravity observation format: N_stations header, then x y z data std.",
    ),
    DataFormat.GIF_MAGNETIC: FormatSpec(
        name="GIF Magnetic",
        extension=[".mag", ".obs"],
        required_columns=["x", "y", "z", "mag"],
        optional_columns=["std"],
        description="UBC-GIF magnetic observation format with inducing field parameters.",
    ),
    DataFormat.GEOSOFT_XYZ: FormatSpec(
        name="Geosoft XYZ",
        extension=[".xyz"],
        required_columns=["x", "y"],
        optional_columns=["z", "value", "line"],
        description="Geosoft XYZ format with / prefixed headers and line-based data.",
    ),
    DataFormat.TOMOFAST_DATA: FormatSpec(
        name="Tomofast-x Data",
        extension=[".txt", ".dat"],
        required_columns=["x", "y", "z", "value", "uncertainty"],
        optional_columns=[],
        description="Tomofast-x native data format: x y z value uncertainty (space-separated).",
    ),
    DataFormat.TOMOFAST_MESH: FormatSpec(
        name="Tomofast-x Mesh",
        extension=[".txt"],
        required_columns=[],
        optional_columns=[],
        description="Tomofast-x mesh format: ncells header then x1 y1 z1 x2 y2 z2 per cell.",
    ),
    DataFormat.TOMOFAST_MODEL: FormatSpec(
        name="Tomofast-x Model",
        extension=[".txt"],
        required_columns=[],
        optional_columns=[],
        description="Tomofast-x model file: one value per cell, matching mesh ordering.",
    ),
}


def list_supported_formats() -> List[FormatSpec]:
    """Return a list of all supported format specifications.

    Returns:
        List of FormatSpec objects describing each supported format.
    """
    return list(FORMAT_SPECS.values())


def detect_format(filepath: str | Path) -> DataFormat:
    """Auto-detect the geophysical data format of a file.

    Uses file extension, content patterns, and header analysis to determine
    the most likely format.

    Args:
        filepath: Path to the data file.

    Returns:
        DataFormat enum indicating the detected format.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    ext = filepath.suffix.lower()
    name_lower = filepath.name.lower()

    # Read first few lines for content analysis
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            lines = [f.readline() for _ in range(20)]
            lines = [ln.strip() for ln in lines if ln.strip()]
    except Exception:
        return DataFormat.UNKNOWN

    if not lines:
        return DataFormat.UNKNOWN

    # --- Extension-based quick checks ---
    if ext == ".xyz":
        return DataFormat.GEOSOFT_XYZ

    if ext == ".grv":
        return DataFormat.GIF_GRAVITY

    if ext == ".mag":
        return DataFormat.GIF_MAGNETIC

    if ext == ".msh":
        return DataFormat.UBC_MESH

    if ext == ".mod":
        return DataFormat.UBC_MODEL

    # --- Content-based detection ---
    first_line = lines[0]

    # Check for Geosoft XYZ header (starts with /)
    if first_line.startswith("/"):
        return DataFormat.GEOSOFT_XYZ

    # Check for CSV with header
    if ext == ".csv" or "," in first_line:
        header_lower = first_line.lower()
        if any(col in header_lower for col in ["x", "y", "z", "grav", "mag", "value"]):
            return DataFormat.CSV

    # Check for UBC mesh format: first line is 3 integers (NE NN NZ)
    parts = first_line.split()
    if len(parts) == 3 and all(_is_int(p) for p in parts):
        # Could be UBC mesh (3 ints for cell counts) or data
        if len(lines) >= 2:
            second_parts = lines[1].split()
            # UBC mesh: second line is origin (3 floats)
            if len(second_parts) == 3 and all(_is_float(p) for p in second_parts):
                # Check third line for cell sizes
                if len(lines) >= 3:
                    third_parts = lines[2].split()
                    if all(_is_float(p) for p in third_parts):
                        return DataFormat.UBC_MESH

    # Check for GIF gravity format: first line is N (number of stations)
    if len(parts) == 1 and _is_int(parts[0]):
        n_stations = int(parts[0])
        if len(lines) >= 2:
            second_parts = lines[1].split()
            # GIF gravity: x y z data [std]
            if len(second_parts) in [4, 5] and all(_is_float(p) for p in second_parts):
                if n_stations > 0 and n_stations < 100000:
                    return DataFormat.GIF_GRAVITY

    # Check for GIF magnetic: first line has inducing field params
    if len(parts) == 3 and all(_is_float(p) for p in parts):
        # Could be magnetic: strength inc dec on first line
        if len(lines) >= 2:
            second_parts = lines[1].split()
            if len(second_parts) == 3 and all(_is_float(p) for p in second_parts):
                # Third line might be N_stations
                if len(lines) >= 3:
                    third_parts = lines[2].split()
                    if len(third_parts) == 1 and _is_int(third_parts[0]):
                        return DataFormat.GIF_MAGNETIC

    # Check for Tomofast-x mesh: first line is single integer (ncells)
    if len(parts) == 1 and _is_int(parts[0]):
        if len(lines) >= 2:
            second_parts = lines[1].split()
            # Tomofast mesh: 6 floats per line (x1 y1 z1 x2 y2 z2)
            if len(second_parts) == 6 and all(_is_float(p) for p in second_parts):
                return DataFormat.TOMOFAST_MESH

    # Check for Tomofast-x data: 5 columns (x y z value uncertainty)
    if len(parts) == 5 and all(_is_float(p) for p in parts):
        # Check if most lines have 5 columns
        consistent = sum(1 for ln in lines[1:6] if len(ln.split()) == 5)
        if consistent >= 3:
            return DataFormat.TOMOFAST_DATA

    # Check for UBC model: single column of floats
    if len(parts) == 1 and _is_float(parts[0]):
        consistent = sum(1 for ln in lines[:10] if len(ln.split()) == 1 and _is_float(ln))
        if consistent >= 5:
            return DataFormat.UBC_MODEL

    # Fallback to CSV if it has comma separation
    if "," in first_line:
        return DataFormat.CSV

    # Generic text with columns
    if ext in [".txt", ".dat"] and len(parts) >= 3:
        if all(_is_float(p) for p in parts):
            return DataFormat.TOMOFAST_DATA

    return DataFormat.UNKNOWN


def _is_int(s: str) -> bool:
    """Check if a string represents an integer."""
    try:
        int(s)
        return True
    except ValueError:
        return False


def _is_float(s: str) -> bool:
    """Check if a string represents a float."""
    try:
        float(s)
        return True
    except ValueError:
        return False
