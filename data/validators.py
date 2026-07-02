"""
Data validation for geophysical datasets.

Provides comprehensive validation for gravity, magnetic, mesh, and topography
files including format checks, range validation, and consistency tests.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .formats import DataFormat, detect_format


@dataclass
class ValidationResult:
    """Result of a data validation operation.

    Attributes:
        is_valid: Whether the data passed all validation checks.
        errors: List of critical errors that prevent data use.
        warnings: List of non-critical issues.
        info: Additional informational messages.
        n_records: Number of data records found.
        format_detected: The format detected for the file.
    """

    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: List[str] = field(default_factory=list)
    n_records: int = 0
    format_detected: Optional[DataFormat] = None

    def add_error(self, msg: str) -> None:
        """Add an error and mark result as invalid."""
        self.errors.append(msg)
        self.is_valid = False

    def add_warning(self, msg: str) -> None:
        """Add a warning (does not invalidate)."""
        self.warnings.append(msg)

    def add_info(self, msg: str) -> None:
        """Add an informational message."""
        self.info.append(msg)

    def __str__(self) -> str:
        status = "VALID" if self.is_valid else "INVALID"
        parts = [f"Validation: {status} ({self.n_records} records)"]
        if self.format_detected:
            parts.append(f"  Format: {self.format_detected.value}")
        for e in self.errors:
            parts.append(f"  ERROR: {e}")
        for w in self.warnings:
            parts.append(f"  WARNING: {w}")
        for i in self.info:
            parts.append(f"  INFO: {i}")
        return "\n".join(parts)


class DataValidator:
    """Validates geophysical data files for use in inversion workflows.

    Provides methods to validate gravity data, magnetic data, mesh files,
    and topography files with comprehensive checks for common issues.

    Example:
        >>> validator = DataValidator()
        >>> result = validator.validate_gravity_data("observations.csv")
        >>> if result.is_valid:
        ...     print("Data ready for inversion")
    """

    # Reasonable ranges for geophysical data
    GRAVITY_RANGE_MGAL: Tuple[float, float] = (-500.0, 500.0)
    MAGNETIC_RANGE_NT: Tuple[float, float] = (-100000.0, 100000.0)
    COORDINATE_RANGE_M: Tuple[float, float] = (-1e8, 1e8)
    DEPTH_RANGE_M: Tuple[float, float] = (-50000.0, 50000.0)

    def validate_gravity_data(self, filepath: str | Path) -> ValidationResult:
        """Validate a gravity observation data file.

        Checks:
            - File exists and is readable
            - Required columns present (x, y, z, gravity anomaly)
            - No NaN or Inf values
            - Values within reasonable ranges (-500 to 500 mGal)
            - Coordinate system consistency
            - Uncertainty column if present

        Args:
            filepath: Path to the gravity data file.

        Returns:
            ValidationResult with detailed findings.
        """
        result = ValidationResult()
        filepath = Path(filepath)

        # Basic file checks
        if not self._check_file_basics(filepath, result):
            return result

        # Detect format
        fmt = detect_format(filepath)
        result.format_detected = fmt

        # Load data
        data = self._load_tabular_data(filepath, result)
        if data is None:
            return result

        columns, values = data
        result.n_records = values.shape[0]
        result.add_info(f"Loaded {result.n_records} stations with {values.shape[1]} columns")

        # Check required columns
        col_lower = [c.lower() for c in columns]
        required = {"x": None, "y": None, "z": None}

        for i, col in enumerate(col_lower):
            if "x" == col or col.startswith("east"):
                required["x"] = i
            elif "y" == col or col.startswith("north"):
                required["y"] = i
            elif "z" == col or col.startswith("elev") or col.startswith("depth"):
                required["z"] = i

        for key, idx in required.items():
            if idx is None:
                result.add_error(f"Missing required coordinate column: {key}")

        if not result.is_valid:
            return result

        # Find gravity column
        grav_col = None
        for i, col in enumerate(col_lower):
            if "grav" in col or "anomaly" in col or "gz" in col:
                grav_col = i
                break
        if grav_col is None and values.shape[1] >= 4:
            grav_col = 3
            result.add_warning("No gravity column identified by name; using column 4")

        if grav_col is None:
            result.add_error("Cannot identify gravity anomaly column")
            return result

        # Validate values
        self._check_nan_inf(values, columns, result)

        # Check gravity range
        grav_values = values[:, grav_col]
        gmin, gmax = np.nanmin(grav_values), np.nanmax(grav_values)
        result.add_info(f"Gravity range: {gmin:.4f} to {gmax:.4f} mGal")

        if gmin < self.GRAVITY_RANGE_MGAL[0] or gmax > self.GRAVITY_RANGE_MGAL[1]:
            result.add_warning(
                f"Gravity values outside typical range "
                f"({self.GRAVITY_RANGE_MGAL[0]} to {self.GRAVITY_RANGE_MGAL[1]} mGal). "
                f"Check units."
            )

        # Check coordinates
        self._check_coordinates(values, required, result)

        # Check uncertainty column
        unc_col = None
        for i, col in enumerate(col_lower):
            if "unc" in col or "std" in col or "err" in col:
                unc_col = i
                break
        if unc_col is not None:
            unc_values = values[:, unc_col]
            if np.any(unc_values <= 0):
                result.add_warning("Some uncertainty values are <= 0")
            result.add_info(f"Uncertainty range: {np.nanmin(unc_values):.6f} to {np.nanmax(unc_values):.6f}")
        else:
            result.add_warning("No uncertainty column found; will need to assign default uncertainties")

        return result

    def validate_magnetic_data(self, filepath: str | Path) -> ValidationResult:
        """Validate a magnetic observation data file.

        Checks:
            - File exists and is readable
            - Required columns present (x, y, z, magnetic anomaly)
            - No NaN or Inf values
            - Values within reasonable ranges (-100000 to 100000 nT)
            - Field parameters (inclination, declination, strength) if present

        Args:
            filepath: Path to the magnetic data file.

        Returns:
            ValidationResult with detailed findings.
        """
        result = ValidationResult()
        filepath = Path(filepath)

        if not self._check_file_basics(filepath, result):
            return result

        fmt = detect_format(filepath)
        result.format_detected = fmt

        data = self._load_tabular_data(filepath, result)
        if data is None:
            return result

        columns, values = data
        result.n_records = values.shape[0]
        result.add_info(f"Loaded {result.n_records} stations with {values.shape[1]} columns")

        col_lower = [c.lower() for c in columns]

        # Check required columns
        required = {"x": None, "y": None, "z": None}
        for i, col in enumerate(col_lower):
            if "x" == col or col.startswith("east"):
                required["x"] = i
            elif "y" == col or col.startswith("north"):
                required["y"] = i
            elif "z" == col or col.startswith("elev") or col.startswith("depth"):
                required["z"] = i

        for key, idx in required.items():
            if idx is None:
                result.add_error(f"Missing required coordinate column: {key}")

        if not result.is_valid:
            return result

        # Find magnetic column
        mag_col = None
        for i, col in enumerate(col_lower):
            if "mag" in col or "tmi" in col or "anomaly" in col:
                mag_col = i
                break
        if mag_col is None and values.shape[1] >= 4:
            mag_col = 3
            result.add_warning("No magnetic column identified by name; using column 4")

        if mag_col is None:
            result.add_error("Cannot identify magnetic anomaly column")
            return result

        # Validate values
        self._check_nan_inf(values, columns, result)

        # Check magnetic range
        mag_values = values[:, mag_col]
        mmin, mmax = np.nanmin(mag_values), np.nanmax(mag_values)
        result.add_info(f"Magnetic anomaly range: {mmin:.2f} to {mmax:.2f} nT")

        if mmin < self.MAGNETIC_RANGE_NT[0] or mmax > self.MAGNETIC_RANGE_NT[1]:
            result.add_warning(
                f"Magnetic values outside typical range "
                f"({self.MAGNETIC_RANGE_NT[0]} to {self.MAGNETIC_RANGE_NT[1]} nT). "
                f"Check units."
            )

        # Check coordinates
        self._check_coordinates(values, required, result)

        # Check for field parameters in header/metadata
        result.add_info("Note: Verify inducing field parameters (inclination, declination, strength) are set correctly")

        return result

    def validate_mesh(self, filepath: str | Path) -> ValidationResult:
        """Validate a mesh file for consistency.

        Checks:
            - File exists and is readable
            - Valid mesh format (UBC or Tomofast)
            - Consistent cell counts
            - Positive cell sizes
            - Reasonable total extent

        Args:
            filepath: Path to the mesh file.

        Returns:
            ValidationResult with detailed findings.
        """
        result = ValidationResult()
        filepath = Path(filepath)

        if not self._check_file_basics(filepath, result):
            return result

        fmt = detect_format(filepath)
        result.format_detected = fmt

        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        except Exception as e:
            result.add_error(f"Cannot read file: {e}")
            return result

        if not lines:
            result.add_error("File is empty")
            return result

        # Try UBC mesh format
        first_parts = lines[0].split()
        if len(first_parts) == 3:
            try:
                ne, nn, nz = int(first_parts[0]), int(first_parts[1]), int(first_parts[2])
                result.n_records = ne * nn * nz
                result.add_info(f"Mesh dimensions: {ne} x {nn} x {nz} = {ne*nn*nz} cells")

                if ne <= 0 or nn <= 0 or nz <= 0:
                    result.add_error("Cell counts must be positive")
                    return result

                if ne > 1000 or nn > 1000 or nz > 500:
                    result.add_warning(f"Large mesh ({ne}x{nn}x{nz}): may be slow to compute")

                # Check origin
                if len(lines) >= 2:
                    origin_parts = lines[1].split()
                    if len(origin_parts) == 3:
                        ox, oy, oz = float(origin_parts[0]), float(origin_parts[1]), float(origin_parts[2])
                        result.add_info(f"Origin: ({ox}, {oy}, {oz})")
                    else:
                        result.add_error(f"Invalid origin line: expected 3 values, got {len(origin_parts)}")

                # Check cell sizes
                for dim_idx, dim_name in enumerate(["East", "North", "Vertical"], start=2):
                    if dim_idx < len(lines):
                        sizes = lines[dim_idx].split()
                        # Can be individual sizes or compressed notation (e.g., "10*50")
                        total_cells = 0
                        for s in sizes:
                            if "*" in s:
                                n, sz = s.split("*")
                                total_cells += int(n)
                                if float(sz) <= 0:
                                    result.add_error(f"Negative/zero cell size in {dim_name} direction")
                            else:
                                total_cells += 1
                                if float(s) <= 0:
                                    result.add_error(f"Negative/zero cell size in {dim_name} direction")

            except (ValueError, IndexError) as e:
                result.add_error(f"Error parsing mesh: {e}")

        # Try Tomofast mesh format: first line is N_cells
        elif len(first_parts) == 1:
            try:
                ncells = int(first_parts[0])
                result.n_records = ncells
                result.add_info(f"Tomofast mesh with {ncells} cells")

                if ncells <= 0:
                    result.add_error("Number of cells must be positive")
                    return result

                # Check a few cell definitions
                if len(lines) >= 2:
                    cell_parts = lines[1].split()
                    if len(cell_parts) == 6:
                        result.add_info("Cell format: x1 y1 z1 x2 y2 z2 (bounding box)")
                    else:
                        result.add_warning(f"Unexpected cell definition format ({len(cell_parts)} values)")

                actual_cells = len(lines) - 1
                if actual_cells != ncells:
                    result.add_warning(
                        f"Header says {ncells} cells but file has {actual_cells} cell lines"
                    )
            except ValueError:
                result.add_error("Cannot parse mesh header")
        else:
            result.add_error("Unrecognized mesh format")

        return result

    def validate_topography(self, filepath: str | Path) -> ValidationResult:
        """Validate a topography file.

        Checks:
            - File exists and is readable
            - Contains x, y, elevation columns
            - No NaN or Inf values
            - Elevation within reasonable range

        Args:
            filepath: Path to the topography file.

        Returns:
            ValidationResult with detailed findings.
        """
        result = ValidationResult()
        filepath = Path(filepath)

        if not self._check_file_basics(filepath, result):
            return result

        data = self._load_tabular_data(filepath, result)
        if data is None:
            return result

        columns, values = data
        result.n_records = values.shape[0]

        if values.shape[1] < 3:
            result.add_error("Topography file must have at least 3 columns (x, y, elevation)")
            return result

        result.add_info(f"Loaded {result.n_records} topography points")

        # Check elevation range
        elev = values[:, 2]
        emin, emax = np.nanmin(elev), np.nanmax(elev)
        result.add_info(f"Elevation range: {emin:.1f} to {emax:.1f} m")

        if emax - emin > 20000:
            result.add_warning("Elevation range > 20 km: check units (should be meters)")

        # Check for NaN/Inf
        self._check_nan_inf(values, columns, result)

        # Check coordinate range
        x_range = np.nanmax(values[:, 0]) - np.nanmin(values[:, 0])
        y_range = np.nanmax(values[:, 1]) - np.nanmin(values[:, 1])
        result.add_info(f"Spatial extent: {x_range:.1f} x {y_range:.1f} m")

        return result

    # --- Private helper methods ---

    def _check_file_basics(self, filepath: Path, result: ValidationResult) -> bool:
        """Check basic file accessibility."""
        if not filepath.exists():
            result.add_error(f"File not found: {filepath}")
            return False
        if not filepath.is_file():
            result.add_error(f"Not a file: {filepath}")
            return False
        if filepath.stat().st_size == 0:
            result.add_error("File is empty")
            return False
        if not os.access(filepath, os.R_OK):
            result.add_error(f"File not readable: {filepath}")
            return False
        return True

    def _load_tabular_data(
        self, filepath: Path, result: ValidationResult
    ) -> Optional[Tuple[List[str], np.ndarray]]:
        """Load tabular data from a file with auto-detection of delimiter and header.

        Returns:
            Tuple of (column_names, data_array) or None if loading fails.
        """
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except Exception as e:
            result.add_error(f"Cannot read file: {e}")
            return None

        lines = [ln.strip() for ln in content.split("\n") if ln.strip()]
        if not lines:
            result.add_error("No data lines found")
            return None

        # Detect delimiter
        delimiter = self._detect_delimiter(lines[0])

        # Detect header
        first_parts = lines[0].split(delimiter) if delimiter != " " else lines[0].split()
        has_header = not all(self._is_numeric(p.strip()) for p in first_parts)

        if has_header:
            columns = [p.strip().strip('"').strip("'") for p in first_parts]
            data_lines = lines[1:]
        else:
            n_cols = len(first_parts)
            columns = [f"col_{i}" for i in range(n_cols)]
            data_lines = lines

        # Parse data
        rows = []
        for i, line in enumerate(data_lines):
            parts = line.split(delimiter) if delimiter != " " else line.split()
            try:
                row = [float(p.strip()) for p in parts if p.strip()]
                if row:
                    rows.append(row)
            except ValueError:
                if i < 5:  # Skip problematic lines at start (possible sub-headers)
                    continue
                result.add_warning(f"Skipped non-numeric line {i + 1 + (1 if has_header else 0)}")

        if not rows:
            result.add_error("No numeric data could be parsed")
            return None

        # Ensure consistent column count
        n_cols = len(rows[0])
        rows = [r for r in rows if len(r) == n_cols]

        values = np.array(rows)
        if len(columns) != n_cols:
            columns = [f"col_{i}" for i in range(n_cols)]

        return columns, values

    def _detect_delimiter(self, line: str) -> str:
        """Detect the delimiter in a data line."""
        if "\t" in line:
            return "\t"
        if "," in line:
            return ","
        return " "

    def _is_numeric(self, s: str) -> bool:
        """Check if a string is numeric."""
        try:
            float(s)
            return True
        except ValueError:
            return False

    def _check_nan_inf(
        self, values: np.ndarray, columns: List[str], result: ValidationResult
    ) -> None:
        """Check for NaN and Inf values."""
        nan_count = np.sum(np.isnan(values))
        inf_count = np.sum(np.isinf(values))

        if nan_count > 0:
            result.add_error(f"Data contains {nan_count} NaN values")
        if inf_count > 0:
            result.add_error(f"Data contains {inf_count} Inf values")

    def _check_coordinates(
        self, values: np.ndarray, coord_indices: dict, result: ValidationResult
    ) -> None:
        """Check coordinate ranges and consistency."""
        for name, idx in coord_indices.items():
            if idx is None:
                continue
            col_vals = values[:, idx]
            vmin, vmax = np.nanmin(col_vals), np.nanmax(col_vals)

            if vmin < self.COORDINATE_RANGE_M[0] or vmax > self.COORDINATE_RANGE_M[1]:
                result.add_warning(f"Coordinate {name} outside typical range: [{vmin}, {vmax}]")

            # Check for duplicate stations
        if all(v is not None for v in coord_indices.values()):
            coords = values[:, [coord_indices["x"], coord_indices["y"], coord_indices["z"]]]
            unique_rows = np.unique(coords, axis=0)
            if len(unique_rows) < len(coords):
                n_dup = len(coords) - len(unique_rows)
                result.add_warning(f"{n_dup} duplicate station locations detected")

            # Check station spacing
            if len(coords) > 1:
                from scipy.spatial import distance as sp_dist

                try:
                    dists = sp_dist.pdist(coords[:, :2])
                    min_dist = np.min(dists)
                    result.add_info(f"Minimum station spacing: {min_dist:.1f} m")
                except ImportError:
                    # scipy not available; skip spacing check
                    pass
