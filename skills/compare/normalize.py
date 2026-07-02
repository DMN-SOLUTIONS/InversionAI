"""
Model normalization and interpolation module.

Provides functions to interpolate inversion models from different meshes
onto a common regular grid for fair comparison.
"""

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


def interpolate_to_common_grid(
    model_path_a: str,
    model_path_b: str,
    mesh_path_a: str = "",
    mesh_path_b: str = "",
    target_resolution: Optional[float] = None,
) -> tuple["np.ndarray", "np.ndarray", dict[str, Any]]:
    """Interpolate two models onto a common regular grid.

    Handles different mesh types (regular grid, unstructured) by loading
    model values with their spatial coordinates and interpolating onto
    a shared regular grid.

    Args:
        model_path_a: Path to first model file.
        model_path_b: Path to second model file.
        mesh_path_a: Path to first mesh file (optional, for UBC format).
        mesh_path_b: Path to second mesh file (optional, for UBC format).
        target_resolution: Cell size for common grid. If None, auto-calculated.

    Returns:
        Tuple of (model_a_resampled, model_b_resampled, common_grid_info).
        common_grid_info contains: nx, ny, nz, origin, cell_size, shape.

    Raises:
        ValueError: If models cannot be loaded or interpolated.
    """
    logger.info("Loading models for interpolation...")

    # Load model A with coordinates
    coords_a, values_a = _load_model_with_coords(model_path_a, mesh_path_a)
    coords_b, values_b = _load_model_with_coords(model_path_b, mesh_path_b)

    logger.info(f"Model A: {len(values_a)} cells, Model B: {len(values_b)} cells")

    # Determine common grid bounds (intersection of both model domains)
    common_bounds = _compute_common_bounds(coords_a, coords_b)

    # Determine resolution
    if target_resolution is None:
        target_resolution = _estimate_resolution(coords_a, coords_b)

    logger.info(f"Common grid resolution: {target_resolution:.2f}")

    # Build common grid
    common_grid = _build_common_grid(common_bounds, target_resolution)

    # Interpolate both models onto common grid
    logger.info("Interpolating Model A onto common grid...")
    model_a_interp = _interpolate_model(coords_a, values_a, common_grid)

    logger.info("Interpolating Model B onto common grid...")
    model_b_interp = _interpolate_model(coords_b, values_b, common_grid)

    logger.info(
        f"Interpolation complete. Common grid: {common_grid['shape']} = "
        f"{np.prod(common_grid['shape'])} cells"
    )

    return model_a_interp, model_b_interp, common_grid


def _load_model_with_coords(
    model_path: str,
    mesh_path: str = "",
) -> tuple["np.ndarray", "np.ndarray"]:
    """Load model values with their spatial coordinates.

    Supports:
    - UBC mesh + model format
    - VTK files (basic parsing)
    - Simple text files (x, y, z, value columns)
    - Single-column files (requires mesh for coordinates)

    Args:
        model_path: Path to model file.
        mesh_path: Path to mesh file (for UBC format).

    Returns:
        Tuple of (coordinates Nx3, values N).
    """
    path = Path(model_path)

    if not path.exists():
        raise ValueError(f"Model file not found: {model_path}")

    suffix = path.suffix.lower()

    if suffix == ".vtk":
        return _load_vtk_model(model_path)
    elif mesh_path and Path(mesh_path).exists():
        return _load_ubc_model(model_path, mesh_path)
    else:
        return _load_text_model(model_path)


def _load_text_model(model_path: str) -> tuple["np.ndarray", "np.ndarray"]:
    """Load a text model file.

    Handles two formats:
    1. Multi-column (x, y, z, value): coordinates included
    2. Single-column (value only): generates synthetic coordinates

    Args:
        model_path: Path to text file.

    Returns:
        (coordinates, values) tuple.
    """
    try:
        data = np.loadtxt(model_path, comments=("#", "!", "%"))
    except ValueError:
        # Try skipping header
        data = np.loadtxt(model_path, skiprows=1, comments=("#", "!", "%"))

    if data.ndim == 1:
        # Single column: generate linear coordinates
        n = len(data)
        # Assume cubic arrangement
        n_side = int(np.ceil(n ** (1.0 / 3.0)))
        coords = _generate_grid_coords(n_side, n_side, n_side, 1.0)[:n]
        return coords, data

    if data.shape[1] >= 4:
        # Multi-column: x, y, z, value
        coords = data[:, :3]
        values = data[:, 3]
        return coords, values
    elif data.shape[1] == 1:
        n = data.shape[0]
        n_side = int(np.ceil(n ** (1.0 / 3.0)))
        coords = _generate_grid_coords(n_side, n_side, n_side, 1.0)[:n]
        return coords, data[:, 0]
    else:
        raise ValueError(
            f"Cannot parse model file with {data.shape[1]} columns: {model_path}"
        )


def _load_ubc_model(
    model_path: str,
    mesh_path: str,
) -> tuple["np.ndarray", "np.ndarray"]:
    """Load UBC-format mesh and model files.

    UBC mesh format:
    Line 1: nx ny nz
    Line 2: x0 y0 z0 (origin, top SW corner)
    Lines 3-5: cell widths in x, y, z

    Model file: single column of values (one per cell).

    Args:
        model_path: Path to model file.
        mesh_path: Path to UBC mesh file.

    Returns:
        (cell_centers, values) tuple.
    """
    # Parse mesh file
    with open(mesh_path, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    # Line 1: dimensions
    dims = lines[0].split()
    nx, ny, nz = int(dims[0]), int(dims[1]), int(dims[2])

    # Line 2: origin
    origin_parts = lines[1].split()
    x0, y0, z0 = float(origin_parts[0]), float(origin_parts[1]), float(origin_parts[2])

    # Lines 3-5: cell widths
    hx = np.array([float(v) for v in lines[2].split()])
    hy = np.array([float(v) for v in lines[3].split()])
    hz = np.array([float(v) for v in lines[4].split()])

    # If dimensions don't match, use uniform widths
    if len(hx) == 1:
        hx = np.ones(nx) * hx[0]
    if len(hy) == 1:
        hy = np.ones(ny) * hy[0]
    if len(hz) == 1:
        hz = np.ones(nz) * hz[0]

    # Compute cell centers
    x_centers = x0 + np.cumsum(hx) - hx / 2
    y_centers = y0 + np.cumsum(hy) - hy / 2
    z_centers = z0 - np.cumsum(hz) + hz / 2  # UBC: z positive up, mesh goes down

    # Create coordinate grid (UBC order: x fastest, then y, then z from top)
    coords = []
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                coords.append([x_centers[ix], y_centers[iy], z_centers[iz]])

    coords = np.array(coords)

    # Load model values
    values = np.loadtxt(model_path, comments=("#", "!", "%"))
    if values.ndim > 1:
        values = values.flatten()

    # Trim if sizes don't match
    n_cells = nx * ny * nz
    if len(values) > n_cells:
        values = values[:n_cells]
    elif len(values) < n_cells:
        coords = coords[: len(values)]

    return coords, values


def _load_vtk_model(model_path: str) -> tuple["np.ndarray", "np.ndarray"]:
    """Load model from VTK file (basic structured grid support).

    Args:
        model_path: Path to VTK file.

    Returns:
        (coordinates, values) tuple.
    """
    coords_list = []
    values_list = []
    reading_points = False
    reading_scalars = False
    n_points = 0

    with open(model_path, "r", errors="replace") as f:
        for line in f:
            line = line.strip()

            if line.startswith("POINTS"):
                parts = line.split()
                n_points = int(parts[1])
                reading_points = True
                reading_scalars = False
                continue

            if line.startswith("SCALARS") or line.startswith("CELL_DATA"):
                reading_points = False
                reading_scalars = True
                continue

            if line.startswith("LOOKUP_TABLE"):
                continue

            if reading_points and line:
                parts = line.split()
                for i in range(0, len(parts), 3):
                    if i + 2 < len(parts):
                        coords_list.append([
                            float(parts[i]),
                            float(parts[i + 1]),
                            float(parts[i + 2]),
                        ])

            if reading_scalars and line:
                parts = line.split()
                for p in parts:
                    try:
                        values_list.append(float(p))
                    except ValueError:
                        reading_scalars = False
                        break

    if not coords_list:
        # If no point coordinates found, generate synthetic
        n = len(values_list)
        n_side = int(np.ceil(n ** (1.0 / 3.0)))
        coords = _generate_grid_coords(n_side, n_side, n_side, 1.0)[:n]
    else:
        coords = np.array(coords_list)

    values = np.array(values_list)

    # Match sizes
    min_len = min(len(coords), len(values))
    return coords[:min_len], values[:min_len]


def _generate_grid_coords(
    nx: int,
    ny: int,
    nz: int,
    cell_size: float,
) -> "np.ndarray":
    """Generate regular grid cell center coordinates.

    Args:
        nx, ny, nz: Number of cells in each direction.
        cell_size: Uniform cell size.

    Returns:
        Array of shape (nx*ny*nz, 3) with cell center coordinates.
    """
    x = np.arange(nx) * cell_size + cell_size / 2
    y = np.arange(ny) * cell_size + cell_size / 2
    z = np.arange(nz) * cell_size + cell_size / 2

    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    coords = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    return coords


def _compute_common_bounds(
    coords_a: "np.ndarray",
    coords_b: "np.ndarray",
) -> dict[str, float]:
    """Compute the intersection bounding box of two coordinate sets.

    Args:
        coords_a: First set of coordinates (Nx3).
        coords_b: Second set of coordinates (Nx3).

    Returns:
        Dictionary with x_min, x_max, y_min, y_max, z_min, z_max.
    """
    bounds = {
        "x_min": max(coords_a[:, 0].min(), coords_b[:, 0].min()),
        "x_max": min(coords_a[:, 0].max(), coords_b[:, 0].max()),
        "y_min": max(coords_a[:, 1].min(), coords_b[:, 1].min()),
        "y_max": min(coords_a[:, 1].max(), coords_b[:, 1].max()),
        "z_min": max(coords_a[:, 2].min(), coords_b[:, 2].min()),
        "z_max": min(coords_a[:, 2].max(), coords_b[:, 2].max()),
    }

    # Validate bounds
    for dim in ["x", "y", "z"]:
        if bounds[f"{dim}_max"] <= bounds[f"{dim}_min"]:
            logger.warning(
                f"No overlap in {dim}-direction. Using union instead of intersection."
            )
            bounds[f"{dim}_min"] = min(
                coords_a[:, "xyz".index(dim)].min(),
                coords_b[:, "xyz".index(dim)].min(),
            )
            bounds[f"{dim}_max"] = max(
                coords_a[:, "xyz".index(dim)].max(),
                coords_b[:, "xyz".index(dim)].max(),
            )

    return bounds


def _estimate_resolution(
    coords_a: "np.ndarray",
    coords_b: "np.ndarray",
) -> float:
    """Estimate appropriate resolution for common grid.

    Uses the coarser of the two model resolutions to avoid over-interpolation.

    Args:
        coords_a: First coordinate set.
        coords_b: Second coordinate set.

    Returns:
        Estimated cell size for common grid.
    """
    res_a = _estimate_model_resolution(coords_a)
    res_b = _estimate_model_resolution(coords_b)

    # Use the coarser resolution (conservative approach)
    resolution = max(res_a, res_b)
    logger.info(f"Resolution estimates: A={res_a:.2f}, B={res_b:.2f}, using {resolution:.2f}")
    return resolution


def _estimate_model_resolution(coords: "np.ndarray") -> float:
    """Estimate cell size from model coordinates.

    Args:
        coords: Nx3 array of cell centers.

    Returns:
        Estimated cell size.
    """
    if len(coords) < 2:
        return 1.0

    # Look at sorted unique x-values to estimate spacing
    x_unique = np.unique(coords[:, 0])
    if len(x_unique) > 1:
        x_diffs = np.diff(x_unique)
        return float(np.median(x_diffs))

    return 1.0


def _build_common_grid(
    bounds: dict[str, float],
    cell_size: float,
) -> dict[str, Any]:
    """Build common grid specification.

    Args:
        bounds: Bounding box dictionary.
        cell_size: Cell size for the grid.

    Returns:
        Grid specification dictionary with coordinates and shape.
    """
    x = np.arange(bounds["x_min"], bounds["x_max"], cell_size)
    y = np.arange(bounds["y_min"], bounds["y_max"], cell_size)
    z = np.arange(bounds["z_min"], bounds["z_max"], cell_size)

    # Ensure at least one cell in each direction
    if len(x) == 0:
        x = np.array([bounds["x_min"]])
    if len(y) == 0:
        y = np.array([bounds["y_min"]])
    if len(z) == 0:
        z = np.array([bounds["z_min"]])

    return {
        "x": x,
        "y": y,
        "z": z,
        "shape": (len(x), len(y), len(z)),
        "cell_size": cell_size,
        "origin": (float(x[0]), float(y[0]), float(z[0])),
        "n_cells": len(x) * len(y) * len(z),
    }


def _interpolate_model(
    coords: "np.ndarray",
    values: "np.ndarray",
    common_grid: dict[str, Any],
) -> "np.ndarray":
    """Interpolate model values onto common grid using nearest-neighbor.

    Falls back to nearest-neighbor if linear interpolation fails
    (common with irregular meshes).

    Args:
        coords: Original cell center coordinates (Nx3).
        values: Model values (N,).
        common_grid: Target grid specification.

    Returns:
        Interpolated values on common grid (flattened).
    """
    from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator

    # Build target points
    xx, yy, zz = np.meshgrid(
        common_grid["x"],
        common_grid["y"],
        common_grid["z"],
        indexing="ij",
    )
    target_points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    # Try linear interpolation first
    try:
        interp = LinearNDInterpolator(coords, values)
        result = interp(target_points)

        # Fill NaN values with nearest neighbor
        nan_mask = np.isnan(result)
        if nan_mask.any():
            nn_interp = NearestNDInterpolator(coords, values)
            result[nan_mask] = nn_interp(target_points[nan_mask])

        return result

    except Exception as e:
        logger.warning(f"Linear interpolation failed ({e}), using nearest-neighbor")
        interp = NearestNDInterpolator(coords, values)
        return interp(target_points)
