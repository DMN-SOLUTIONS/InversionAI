"""
Geophysical data loader with format auto-detection.

Provides a unified interface for loading geophysical observation data from
various formats into a common GeoDataset structure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .formats import DataFormat, detect_format


@dataclass
class GeoDataset:
    """Container for geophysical observation data.

    Attributes:
        stations: Station coordinates as (n, 3) array [x, y, z].
        observations: Observed data values as (n,) array.
        uncertainties: Data uncertainties as (n,) array.
        metadata: Additional metadata (format, units, field params, etc.).
    """

    stations: np.ndarray  # (n, 3)
    observations: np.ndarray  # (n,)
    uncertainties: np.ndarray  # (n,)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_stations(self) -> int:
        """Number of observation stations."""
        return self.stations.shape[0]

    @property
    def extent(self) -> Dict[str, Tuple[float, float]]:
        """Spatial extent of the data.

        Returns:
            Dictionary with 'x', 'y', 'z' keys mapping to (min, max) tuples.
        """
        return {
            "x": (float(np.min(self.stations[:, 0])), float(np.max(self.stations[:, 0]))),
            "y": (float(np.min(self.stations[:, 1])), float(np.max(self.stations[:, 1]))),
            "z": (float(np.min(self.stations[:, 2])), float(np.max(self.stations[:, 2]))),
        }

    @property
    def station_spacing(self) -> float:
        """Estimate median station spacing from nearest-neighbor distances.

        Returns:
            Estimated median spacing in same units as coordinates.
        """
        if self.n_stations < 2:
            return 0.0
        # Use a simple approach without scipy dependency
        coords_2d = self.stations[:, :2]
        # Sample up to 200 points for efficiency
        n_sample = min(200, self.n_stations)
        indices = np.linspace(0, self.n_stations - 1, n_sample, dtype=int)
        sample = coords_2d[indices]

        min_dists = []
        for i in range(len(sample)):
            dists = np.sqrt(np.sum((sample - sample[i]) ** 2, axis=1))
            dists[i] = np.inf
            min_dists.append(np.min(dists))

        return float(np.median(min_dists))

    def __repr__(self) -> str:
        ext = self.extent
        return (
            f"GeoDataset(n_stations={self.n_stations}, "
            f"x=[{ext['x'][0]:.1f}, {ext['x'][1]:.1f}], "
            f"y=[{ext['y'][0]:.1f}, {ext['y'][1]:.1f}], "
            f"obs_range=[{np.min(self.observations):.4f}, {np.max(self.observations):.4f}])"
        )


class DataLoader:
    """Load geophysical data from various formats into GeoDataset.

    Handles format detection, delimiter parsing, header recognition,
    and unit conversion for common geophysical file formats.

    Example:
        >>> loader = DataLoader()
        >>> dataset = loader.load("observations.csv")
        >>> print(dataset.n_stations)
        100
    """

    def load(self, filepath: str | Path, format: Optional[DataFormat] = None) -> GeoDataset:
        """Load geophysical data from a file.

        Args:
            filepath: Path to the data file.
            format: Explicit format specification. If None, auto-detects.

        Returns:
            GeoDataset containing the loaded data.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file cannot be parsed.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        if format is None:
            format = detect_format(filepath)

        loaders = {
            DataFormat.CSV: self._load_csv,
            DataFormat.GIF_GRAVITY: self._load_gif_gravity,
            DataFormat.GIF_MAGNETIC: self._load_gif_magnetic,
            DataFormat.GEOSOFT_XYZ: self._load_geosoft_xyz,
            DataFormat.TOMOFAST_DATA: self._load_tomofast_data,
            DataFormat.UBC_OBSERVATION: self._load_ubc_observation,
        }

        loader_func = loaders.get(format, self._load_generic)
        dataset = loader_func(filepath)
        dataset.metadata["source_file"] = str(filepath)
        dataset.metadata["format"] = format.value

        return dataset

    def _load_csv(self, filepath: Path) -> GeoDataset:
        """Load CSV format data."""
        delimiter = self._detect_delimiter(filepath)

        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]

        if not lines:
            raise ValueError(f"Empty file: {filepath}")

        # Detect header
        first_parts = self._split_line(lines[0], delimiter)
        has_header = not all(self._is_numeric(p) for p in first_parts)

        if has_header:
            columns = [p.strip().strip('"').strip("'").lower() for p in first_parts]
            data_lines = lines[1:]
        else:
            columns = []
            data_lines = lines

        # Parse numeric data
        rows = []
        for line in data_lines:
            parts = self._split_line(line, delimiter)
            try:
                row = [float(p) for p in parts if p]
                if row:
                    rows.append(row)
            except ValueError:
                continue

        if not rows:
            raise ValueError(f"No numeric data found in {filepath}")

        data = np.array(rows)

        # Identify columns
        x_col, y_col, z_col, obs_col, unc_col = self._identify_columns(columns, data.shape[1])

        stations = np.column_stack([data[:, x_col], data[:, y_col], data[:, z_col]])
        observations = data[:, obs_col]

        if unc_col is not None:
            uncertainties = data[:, unc_col]
        else:
            # Default: 5% of data range
            data_range = np.max(np.abs(observations))
            uncertainties = np.full(len(observations), 0.05 * data_range)

        metadata = {"columns": columns, "delimiter": delimiter}

        return GeoDataset(
            stations=stations,
            observations=observations,
            uncertainties=uncertainties,
            metadata=metadata,
        )

    def _load_gif_gravity(self, filepath: Path) -> GeoDataset:
        """Load UBC-GIF gravity observation format."""
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]

        if not lines:
            raise ValueError(f"Empty file: {filepath}")

        # First line: number of observations
        n_obs = int(lines[0])
        data_lines = lines[1 : n_obs + 1]

        rows = []
        for line in data_lines:
            parts = line.split()
            rows.append([float(p) for p in parts])

        data = np.array(rows)

        stations = data[:, :3]
        observations = data[:, 3]
        uncertainties = data[:, 4] if data.shape[1] >= 5 else np.full(n_obs, 0.05 * np.max(np.abs(observations)))

        return GeoDataset(
            stations=stations,
            observations=observations,
            uncertainties=uncertainties,
            metadata={"format_type": "gif_gravity"},
        )

    def _load_gif_magnetic(self, filepath: Path) -> GeoDataset:
        """Load UBC-GIF magnetic observation format."""
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]

        if not lines:
            raise ValueError(f"Empty file: {filepath}")

        # First line: inducing field (strength inclination declination)
        field_parts = lines[0].split()
        field_strength = float(field_parts[0])
        field_inc = float(field_parts[1])
        field_dec = float(field_parts[2])

        # Second line: measurement type (inducing field direction repeated or specific)
        # Third line: number of observations
        idx = 1
        # Sometimes second line is also field direction for measurements
        if len(lines[idx].split()) == 3:
            idx += 1

        n_obs = int(lines[idx])
        idx += 1
        data_lines = lines[idx : idx + n_obs]

        rows = []
        for line in data_lines:
            parts = line.split()
            rows.append([float(p) for p in parts])

        data = np.array(rows)

        stations = data[:, :3]
        observations = data[:, 3]
        uncertainties = data[:, 4] if data.shape[1] >= 5 else np.full(n_obs, 0.05 * np.max(np.abs(observations)))

        return GeoDataset(
            stations=stations,
            observations=observations,
            uncertainties=uncertainties,
            metadata={
                "format_type": "gif_magnetic",
                "field_strength_nT": field_strength,
                "field_inclination_deg": field_inc,
                "field_declination_deg": field_dec,
            },
        )

    def _load_geosoft_xyz(self, filepath: Path) -> GeoDataset:
        """Load Geosoft XYZ format."""
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        # Parse header lines (start with /)
        columns: List[str] = []
        data_lines: List[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("/"):
                # Header line - may contain column names
                if "=" not in stripped:
                    parts = stripped[1:].split()
                    if parts:
                        columns = [p.lower() for p in parts]
            elif stripped:
                data_lines.append(stripped)

        rows = []
        for line in data_lines:
            parts = line.split()
            try:
                row = [float(p) if p != "*" else np.nan for p in parts]
                rows.append(row)
            except ValueError:
                continue

        if not rows:
            raise ValueError(f"No data found in {filepath}")

        data = np.array(rows)
        x_col, y_col, z_col, obs_col, unc_col = self._identify_columns(columns, data.shape[1])

        stations = np.column_stack([data[:, x_col], data[:, y_col], data[:, z_col]])
        observations = data[:, obs_col]
        uncertainties = (
            data[:, unc_col]
            if unc_col is not None
            else np.full(len(observations), 0.05 * np.max(np.abs(observations)))
        )

        return GeoDataset(
            stations=stations,
            observations=observations,
            uncertainties=uncertainties,
            metadata={"format_type": "geosoft_xyz", "columns": columns},
        )

    def _load_tomofast_data(self, filepath: Path) -> GeoDataset:
        """Load Tomofast-x native data format (x y z value uncertainty)."""
        data = np.loadtxt(filepath)

        if data.ndim == 1:
            data = data.reshape(1, -1)

        if data.shape[1] < 4:
            raise ValueError(f"Tomofast data requires at least 4 columns, got {data.shape[1]}")

        stations = data[:, :3]
        observations = data[:, 3]
        uncertainties = data[:, 4] if data.shape[1] >= 5 else np.full(len(observations), 0.05 * np.max(np.abs(observations)))

        return GeoDataset(
            stations=stations,
            observations=observations,
            uncertainties=uncertainties,
            metadata={"format_type": "tomofast_data"},
        )

    def _load_ubc_observation(self, filepath: Path) -> GeoDataset:
        """Load UBC observation/location file."""
        # Same structure as GIF gravity for basic case
        return self._load_gif_gravity(filepath)

    def _load_generic(self, filepath: Path) -> GeoDataset:
        """Attempt generic loading for unrecognized formats."""
        # Try numpy loadtxt first
        try:
            data = np.loadtxt(filepath)
            if data.ndim == 1:
                data = data.reshape(1, -1)

            if data.shape[1] >= 4:
                stations = data[:, :3]
                observations = data[:, 3]
                uncertainties = (
                    data[:, 4]
                    if data.shape[1] >= 5
                    else np.full(len(observations), 0.05 * np.max(np.abs(observations)))
                )
                return GeoDataset(
                    stations=stations,
                    observations=observations,
                    uncertainties=uncertainties,
                    metadata={"format_type": "generic"},
                )
        except Exception:
            pass

        # Fall back to CSV loader
        return self._load_csv(filepath)

    # --- Helper methods ---

    def _detect_delimiter(self, filepath: Path) -> str:
        """Detect file delimiter from first few lines."""
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            sample = f.read(4096)

        # Count delimiters
        n_comma = sample.count(",")
        n_tab = sample.count("\t")
        n_lines = sample.count("\n") or 1

        if n_comma / n_lines > 2:
            return ","
        if n_tab / n_lines > 2:
            return "\t"
        return " "

    def _split_line(self, line: str, delimiter: str) -> List[str]:
        """Split a line by delimiter, handling whitespace."""
        if delimiter == " ":
            return line.split()
        return [p.strip() for p in line.split(delimiter)]

    def _is_numeric(self, s: str) -> bool:
        """Check if string is numeric."""
        try:
            float(s.strip())
            return True
        except (ValueError, AttributeError):
            return False

    def _identify_columns(
        self, columns: List[str], n_cols: int
    ) -> Tuple[int, int, int, int, Optional[int]]:
        """Identify x, y, z, observation, and uncertainty column indices.

        Args:
            columns: List of column names (may be empty).
            n_cols: Total number of columns.

        Returns:
            Tuple of (x_idx, y_idx, z_idx, obs_idx, unc_idx or None).
        """
        x_col, y_col, z_col, obs_col, unc_col = 0, 1, 2, 3, None

        if columns:
            for i, col in enumerate(columns):
                col_l = col.lower().strip()
                if col_l in ("x", "easting", "east", "lon", "longitude"):
                    x_col = i
                elif col_l in ("y", "northing", "north", "lat", "latitude"):
                    y_col = i
                elif col_l in ("z", "elevation", "elev", "depth", "altitude"):
                    z_col = i
                elif any(kw in col_l for kw in ("grav", "mag", "anomaly", "value", "tmi", "gz")):
                    obs_col = i
                elif any(kw in col_l for kw in ("unc", "std", "err", "sigma")):
                    unc_col = i

        # Ensure obs_col is different from coordinate columns
        if obs_col in (x_col, y_col, z_col):
            obs_col = min(3, n_cols - 1)

        # Look for uncertainty if not found by name
        if unc_col is None and n_cols >= 5:
            candidates = set(range(n_cols)) - {x_col, y_col, z_col, obs_col}
            if candidates:
                unc_col = min(candidates)

        return x_col, y_col, z_col, obs_col, unc_col
