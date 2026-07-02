"""
Automatic mesh generation for geophysical inversions.

Generates tensor meshes from data extent with intelligent defaults for
cell sizes, padding, and depth extent based on station spacing and survey geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .loader import GeoDataset


@dataclass
class MeshConfig:
    """Configuration for a 3D tensor mesh.

    Attributes:
        origin: Mesh origin (x0, y0, z0) - top-southwest corner.
        cell_sizes: Tuple of (dx, dy, dz) arrays for cell sizes in each direction.
        n_cells: Tuple of (nx, ny, nz) cell counts.
        padding_cells: Tuple of (pad_x, pad_y, pad_z) padding cell counts on each side.
        total_extent: Tuple of (x_extent, y_extent, z_extent) total mesh dimensions.
    """

    origin: Tuple[float, float, float]
    cell_sizes: Tuple[np.ndarray, np.ndarray, np.ndarray]
    n_cells: Tuple[int, int, int]
    padding_cells: Tuple[int, int, int]
    total_extent: Tuple[float, float, float] = field(init=False)

    def __post_init__(self) -> None:
        dx, dy, dz = self.cell_sizes
        self.total_extent = (float(np.sum(dx)), float(np.sum(dy)), float(np.sum(dz)))

    @property
    def total_cells(self) -> int:
        """Total number of cells in the mesh."""
        return self.n_cells[0] * self.n_cells[1] * self.n_cells[2]

    @property
    def core_extent(self) -> Tuple[float, float, float]:
        """Extent of the core (non-padded) region."""
        dx, dy, dz = self.cell_sizes
        px, py, pz = self.padding_cells
        nx, ny, nz = self.n_cells
        core_x = float(np.sum(dx[px : nx - px]))
        core_y = float(np.sum(dy[py : ny - py]))
        core_z = float(np.sum(dz[: nz - pz]))  # z padding is at bottom
        return (core_x, core_y, core_z)

    def summary(self) -> str:
        """Return a human-readable summary of the mesh configuration."""
        nx, ny, nz = self.n_cells
        px, py, pz = self.padding_cells
        return (
            f"Mesh Configuration:\n"
            f"  Dimensions: {nx} x {ny} x {nz} = {self.total_cells:,} cells\n"
            f"  Origin: ({self.origin[0]:.1f}, {self.origin[1]:.1f}, {self.origin[2]:.1f})\n"
            f"  Total extent: {self.total_extent[0]:.1f} x {self.total_extent[1]:.1f} x {self.total_extent[2]:.1f} m\n"
            f"  Core extent: {self.core_extent[0]:.1f} x {self.core_extent[1]:.1f} x {self.core_extent[2]:.1f} m\n"
            f"  Padding: {px} x {py} x {pz} cells per side\n"
            f"  Core cell size: {self.cell_sizes[0][px]:.1f} x {self.cell_sizes[1][py]:.1f} x {self.cell_sizes[2][0]:.1f} m"
        )

    def __repr__(self) -> str:
        nx, ny, nz = self.n_cells
        return f"MeshConfig({nx}x{ny}x{nz}, origin={self.origin}, total_cells={self.total_cells:,})"


def generate_mesh_from_data(
    dataset: GeoDataset,
    n_cells: Optional[Tuple[int, int, int]] = None,
    depth_extent: Optional[float] = None,
    padding: Optional[int] = None,
    expansion_factor: float = 1.5,
    max_cells: int = 500000,
) -> MeshConfig:
    """Generate an inversion mesh from data extent and station spacing.

    Creates a tensor mesh with:
    - Core region covering the data extent
    - Padding cells with geometrically expanding sizes
    - Depth extent based on survey dimensions
    - Cell sizes based on station spacing

    Args:
        dataset: GeoDataset containing station locations.
        n_cells: Explicit (nx, ny, nz) cell counts for core region.
            If None, determined from station spacing.
        depth_extent: Maximum depth of the mesh in meters.
            If None, defaults to 1.5x the largest horizontal extent.
        padding: Number of padding cells on each side.
            If None, defaults to 5.
        expansion_factor: Factor for geometric expansion of padding cells.
        max_cells: Maximum total cells allowed (warning issued if exceeded).

    Returns:
        MeshConfig with the generated mesh parameters.

    Raises:
        ValueError: If dataset has fewer than 2 stations.
    """
    if dataset.n_stations < 2:
        raise ValueError("Need at least 2 stations to generate a mesh")

    # Get data extent
    extent = dataset.extent
    x_range = extent["x"][1] - extent["x"][0]
    y_range = extent["y"][1] - extent["y"][0]
    max_range = max(x_range, y_range)

    # Estimate station spacing
    spacing = dataset.station_spacing
    if spacing <= 0:
        spacing = max_range / 10.0

    # Determine core cell sizes (approximately station spacing / 2 for adequate resolution)
    core_cell_size = spacing / 2.0

    # Determine core cell counts
    if n_cells is not None:
        nx_core, ny_core, nz_core = n_cells
    else:
        nx_core = max(4, int(np.ceil(x_range / core_cell_size)))
        ny_core = max(4, int(np.ceil(y_range / core_cell_size)))

        # Depth: use depth extent
        if depth_extent is None:
            depth_extent = max_range * 1.5

        nz_core = max(4, int(np.ceil(depth_extent / core_cell_size)))

        # Cap to reasonable numbers
        nx_core = min(nx_core, 100)
        ny_core = min(ny_core, 100)
        nz_core = min(nz_core, 50)

    # Recalculate cell sizes to match extent exactly
    dx_core = x_range / nx_core if nx_core > 0 else core_cell_size
    dy_core = y_range / ny_core if ny_core > 0 else core_cell_size

    if depth_extent is None:
        depth_extent = max_range * 1.5
    dz_core = depth_extent / nz_core if nz_core > 0 else core_cell_size

    # Padding
    if padding is None:
        padding = 5

    # Generate padding cell sizes (geometric expansion)
    pad_x = _generate_padding(dx_core, padding, expansion_factor)
    pad_y = _generate_padding(dy_core, padding, expansion_factor)
    pad_z = _generate_padding(dz_core, padding, expansion_factor)

    # Assemble full cell size arrays
    dx = np.concatenate([pad_x[::-1], np.full(nx_core, dx_core), pad_x])
    dy = np.concatenate([pad_y[::-1], np.full(ny_core, dy_core), pad_y])
    dz = np.concatenate([np.full(nz_core, dz_core), pad_z])  # padding only at bottom

    # Calculate origin (accounting for padding)
    x0 = extent["x"][0] - np.sum(pad_x)
    y0 = extent["y"][0] - np.sum(pad_y)
    z0 = float(np.max(dataset.stations[:, 2]))  # top of mesh at station elevation

    # Total cell counts
    nx_total = len(dx)
    ny_total = len(dy)
    nz_total = len(dz)
    total = nx_total * ny_total * nz_total

    if total > max_cells:
        import warnings
        warnings.warn(
            f"Mesh has {total:,} cells (> {max_cells:,}). "
            f"Consider reducing n_cells or increasing cell size.",
            UserWarning,
            stacklevel=2,
        )

    return MeshConfig(
        origin=(x0, y0, z0),
        cell_sizes=(dx, dy, dz),
        n_cells=(nx_total, ny_total, nz_total),
        padding_cells=(padding, padding, padding),
    )


def _generate_padding(core_size: float, n_pad: int, factor: float) -> np.ndarray:
    """Generate geometrically expanding padding cell sizes.

    Args:
        core_size: Size of the innermost padding cell (same as core).
        n_pad: Number of padding cells.
        factor: Expansion factor between successive cells.

    Returns:
        Array of padding cell sizes (innermost to outermost).
    """
    if n_pad <= 0:
        return np.array([])

    sizes = np.array([core_size * (factor ** i) for i in range(n_pad)])
    return sizes
