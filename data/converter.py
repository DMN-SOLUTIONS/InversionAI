"""
Format converter for geophysical datasets.

Converts GeoDataset objects between Tomofast-x, SimPEG, UBC-GIF,
and CSV formats for interoperability between inversion codes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .loader import GeoDataset


class FormatConverter:
    """Convert geophysical datasets between supported formats.

    Provides methods to export GeoDataset objects to various file formats
    used by different inversion software packages.

    Example:
        >>> converter = FormatConverter()
        >>> converter.to_tomofast(dataset, "output/gravity_data.txt")
        >>> arrays = converter.to_simpeg(dataset)
    """

    def to_tomofast(
        self,
        dataset: GeoDataset,
        output_path: str | Path,
        data_type: str = "gravity",
    ) -> Path:
        """Write dataset in Tomofast-x native format.

        Format: space-separated columns: x y z value uncertainty
        No header line.

        Args:
            dataset: GeoDataset to export.
            output_path: Output file path.
            data_type: Type of data ('gravity' or 'magnetic').

        Returns:
            Path to the written file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = np.column_stack([
            dataset.stations,
            dataset.observations,
            dataset.uncertainties,
        ])

        header_comment = f"# Tomofast-x {data_type} data: x y z value uncertainty\n"
        header_comment += f"# N_stations: {dataset.n_stations}\n"

        with open(output_path, "w") as f:
            f.write(header_comment)
            for row in data:
                f.write(f"{row[0]:.6f} {row[1]:.6f} {row[2]:.6f} {row[3]:.10e} {row[4]:.10e}\n")

        return output_path

    def to_simpeg(self, dataset: GeoDataset) -> Dict[str, Any]:
        """Convert dataset to SimPEG-compatible arrays.

        Returns arrays that can be directly used with SimPEG's
        gravity and magnetic survey classes.

        Args:
            dataset: GeoDataset to convert.

        Returns:
            Dictionary with keys:
                - 'receiver_locations': (n, 3) array of station positions
                - 'data': (n,) array of observations
                - 'standard_deviation': (n,) array of uncertainties
                - 'metadata': dict with additional parameters
        """
        result = {
            "receiver_locations": dataset.stations.copy(),
            "data": dataset.observations.copy(),
            "standard_deviation": dataset.uncertainties.copy(),
            "metadata": dataset.metadata.copy(),
        }

        # Add field parameters for magnetic data
        if "field_strength_nT" in dataset.metadata:
            result["inducing_field"] = (
                dataset.metadata["field_strength_nT"],
                dataset.metadata.get("field_inclination_deg", 0.0),
                dataset.metadata.get("field_declination_deg", 0.0),
            )

        return result

    def to_ubc(
        self,
        dataset: GeoDataset,
        output_path: str | Path,
        data_type: str = "gravity",
    ) -> Path:
        """Write dataset in UBC-GIF observation format.

        Gravity format:
            Line 1: N_observations
            Lines 2+: x y z data uncertainty

        Magnetic format:
            Line 1: field_strength inclination declination
            Line 2: field_strength inclination declination (measurement direction)
            Line 3: N_observations
            Lines 4+: x y z data uncertainty

        Args:
            dataset: GeoDataset to export.
            output_path: Output file path.
            data_type: Type of data ('gravity' or 'magnetic').

        Returns:
            Path to the written file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            if data_type == "magnetic":
                # Write inducing field parameters
                strength = dataset.metadata.get("field_strength_nT", 55000.0)
                inc = dataset.metadata.get("field_inclination_deg", -60.0)
                dec = dataset.metadata.get("field_declination_deg", 0.0)
                f.write(f"{strength:.1f} {inc:.1f} {dec:.1f}\n")
                f.write(f"{strength:.1f} {inc:.1f} {dec:.1f}\n")

            f.write(f"{dataset.n_stations}\n")
            for i in range(dataset.n_stations):
                x, y, z = dataset.stations[i]
                obs = dataset.observations[i]
                unc = dataset.uncertainties[i]
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {obs:.10e} {unc:.10e}\n")

        return output_path

    def to_csv(
        self,
        dataset: GeoDataset,
        output_path: str | Path,
        data_type: str = "gravity",
        include_header: bool = True,
    ) -> Path:
        """Write dataset as a standard CSV file.

        Args:
            dataset: GeoDataset to export.
            output_path: Output file path.
            data_type: Type of data for column naming ('gravity' or 'magnetic').
            include_header: Whether to include a header row.

        Returns:
            Path to the written file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        obs_col = "grav_anomaly_mGal" if data_type == "gravity" else "mag_anomaly_nT"
        unc_unit = "mGal" if data_type == "gravity" else "nT"

        with open(output_path, "w") as f:
            if include_header:
                f.write(f"x,y,z,{obs_col},uncertainty_{unc_unit}\n")
            for i in range(dataset.n_stations):
                x, y, z = dataset.stations[i]
                obs = dataset.observations[i]
                unc = dataset.uncertainties[i]
                f.write(f"{x:.1f},{y:.1f},{z:.1f},{obs:.6f},{unc:.6f}\n")

        return output_path

    def to_tomofast_mesh(
        self,
        origin: Tuple[float, float, float],
        cell_sizes: Tuple[np.ndarray, np.ndarray, np.ndarray],
        output_path: str | Path,
    ) -> Path:
        """Write a mesh in Tomofast-x format.

        Tomofast mesh format:
            Line 1: N_cells
            Lines 2+: x1 y1 z1 x2 y2 z2 (cell bounding box corners)

        Args:
            origin: Mesh origin (x0, y0, z0) - top-southwest corner.
            cell_sizes: Tuple of (dx_array, dy_array, dz_array).
            output_path: Output file path.

        Returns:
            Path to the written file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        dx, dy, dz = cell_sizes
        nx, ny, nz = len(dx), len(dy), len(dz)
        n_cells = nx * ny * nz

        # Build cell boundaries
        x_edges = np.cumsum(np.concatenate([[origin[0]], dx]))
        y_edges = np.cumsum(np.concatenate([[origin[1]], dy]))
        z_edges = np.cumsum(np.concatenate([[origin[2]], -dz]))  # z goes down

        with open(output_path, "w") as f:
            f.write(f"{n_cells}\n")
            for iz in range(nz):
                for iy in range(ny):
                    for ix in range(nx):
                        x1, x2 = x_edges[ix], x_edges[ix + 1]
                        y1, y2 = y_edges[iy], y_edges[iy + 1]
                        z1, z2 = z_edges[iz], z_edges[iz + 1]
                        f.write(f"{x1:.6f} {y1:.6f} {z1:.6f} {x2:.6f} {y2:.6f} {z2:.6f}\n")

        return output_path

    def to_ubc_mesh(
        self,
        origin: Tuple[float, float, float],
        cell_sizes: Tuple[np.ndarray, np.ndarray, np.ndarray],
        output_path: str | Path,
    ) -> Path:
        """Write a mesh in UBC-GIF format.

        UBC mesh format:
            Line 1: NE NN NZ
            Line 2: E0 N0 Z0
            Line 3: cell sizes in easting direction
            Line 4: cell sizes in northing direction
            Line 5: cell sizes in vertical direction

        Args:
            origin: Mesh origin (E0, N0, Z0).
            cell_sizes: Tuple of (dx_array, dy_array, dz_array).
            output_path: Output file path.

        Returns:
            Path to the written file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        dx, dy, dz = cell_sizes
        nx, ny, nz = len(dx), len(dy), len(dz)

        with open(output_path, "w") as f:
            f.write(f"{nx} {ny} {nz}\n")
            f.write(f"{origin[0]:.6f} {origin[1]:.6f} {origin[2]:.6f}\n")
            f.write(self._compress_cell_sizes(dx) + "\n")
            f.write(self._compress_cell_sizes(dy) + "\n")
            f.write(self._compress_cell_sizes(dz) + "\n")

        return output_path

    def _compress_cell_sizes(self, sizes: np.ndarray) -> str:
        """Compress uniform cell sizes to UBC notation (e.g., '20*50.0').

        Args:
            sizes: Array of cell sizes.

        Returns:
            String representation with run-length encoding where possible.
        """
        if len(sizes) == 0:
            return ""

        parts = []
        current = sizes[0]
        count = 1

        for s in sizes[1:]:
            if np.isclose(s, current):
                count += 1
            else:
                if count > 1:
                    parts.append(f"{count}*{current:.2f}")
                else:
                    parts.append(f"{current:.2f}")
                current = s
                count = 1

        if count > 1:
            parts.append(f"{count}*{current:.2f}")
        else:
            parts.append(f"{current:.2f}")

        return " ".join(parts)
