"""
SimPEG mesh builder module.

Constructs discretize meshes (TensorMesh or TreeMesh) from data extents,
automatically calculating cell sizes, padding, and depth extent.
"""

import logging
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class MeshBuilder:
    """Builds discretize meshes from observation data locations.

    Automatically determines mesh parameters based on data spacing,
    extent, and desired resolution. Supports TensorMesh (regular grid)
    and TreeMesh (octree with adaptive refinement).
    """

    def build_mesh(
        self,
        locations: "np.ndarray",
        mesh_type: str = "TensorMesh",
        cell_size: Optional[float] = None,
        padding_factor: float = 1.5,
        depth_factor: float = 2.0,
        n_pad_cells: int = 5,
    ) -> Any:
        """Build a discretize mesh from data locations.

        Args:
            locations: Nx3 array of observation locations (x, y, z).
            mesh_type: 'TensorMesh' or 'TreeMesh'.
            cell_size: Base cell size. If None, auto-calculated from data spacing.
            padding_factor: How far padding extends beyond data (as multiple of extent).
            depth_factor: Depth extent as multiple of horizontal extent.
            n_pad_cells: Number of padding cells on each side.

        Returns:
            discretize mesh object (TensorMesh or TreeMesh).

        Raises:
            ImportError: If discretize is not installed.
            ValueError: If locations are invalid.
        """
        import discretize

        if locations.ndim != 2 or locations.shape[1] < 3:
            raise ValueError(
                f"Locations must be Nx3 array, got shape {locations.shape}"
            )

        # Calculate data extents
        x_min, y_min, z_min = locations.min(axis=0)[:3]
        x_max, y_max, z_max = locations.max(axis=0)[:3]

        x_extent = x_max - x_min
        y_extent = y_max - y_min

        # Auto-calculate cell size from data spacing if not provided
        if cell_size is None:
            cell_size = self._estimate_cell_size(locations)
            logger.info(f"Auto-calculated cell size: {cell_size:.2f}")

        # Calculate mesh dimensions
        max_extent = max(x_extent, y_extent)
        depth_extent = max_extent * depth_factor

        if mesh_type == "TensorMesh":
            mesh = self._build_tensor_mesh(
                x_min=x_min, x_max=x_max,
                y_min=y_min, y_max=y_max,
                z_min=z_min,
                cell_size=cell_size,
                depth_extent=depth_extent,
                padding_factor=padding_factor,
                n_pad_cells=n_pad_cells,
            )
        elif mesh_type == "TreeMesh":
            mesh = self._build_tree_mesh(
                locations=locations,
                cell_size=cell_size,
                depth_extent=depth_extent,
                padding_factor=padding_factor,
                x_extent=x_extent,
                y_extent=y_extent,
            )
        else:
            raise ValueError(f"Unknown mesh type: {mesh_type}. Use 'TensorMesh' or 'TreeMesh'.")

        logger.info(
            f"Built {mesh_type} with {mesh.nC} cells, "
            f"cell size={cell_size:.1f}m"
        )
        return mesh

    def _estimate_cell_size(self, locations: "np.ndarray") -> float:
        """Estimate appropriate cell size from data point spacing.

        Uses the median nearest-neighbor distance as a guide,
        then rounds to a convenient value.

        Args:
            locations: Nx3 array of observation points.

        Returns:
            Estimated cell size in data units.
        """
        from scipy.spatial import cKDTree

        n_points = len(locations)

        if n_points < 2:
            return 100.0  # Default fallback

        # Use 2D (horizontal) distances for spacing estimate
        xy = locations[:, :2]

        # Subsample if too many points for KDTree
        if n_points > 10000:
            indices = np.random.choice(n_points, 10000, replace=False)
            xy_sample = xy[indices]
        else:
            xy_sample = xy

        tree = cKDTree(xy_sample)
        distances, _ = tree.query(xy_sample, k=2)  # k=2: self + nearest
        nearest_distances = distances[:, 1]  # Skip self-distance

        # Use median spacing divided by 2 for good resolution
        median_spacing = np.median(nearest_distances)
        cell_size = median_spacing / 2.0

        # Round to a "nice" number
        cell_size = self._round_to_nice(cell_size)

        # Ensure minimum cell size
        cell_size = max(cell_size, 1.0)

        return cell_size

    def _round_to_nice(self, value: float) -> float:
        """Round a value to a 'nice' number (1, 2, 5 x power of 10).

        Args:
            value: Value to round.

        Returns:
            Nearest nice number.
        """
        if value <= 0:
            return 1.0

        magnitude = 10 ** np.floor(np.log10(value))
        normalized = value / magnitude

        if normalized < 1.5:
            nice = 1.0
        elif normalized < 3.5:
            nice = 2.0
        elif normalized < 7.5:
            nice = 5.0
        else:
            nice = 10.0

        return nice * magnitude

    def _build_tensor_mesh(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        z_min: float,
        cell_size: float,
        depth_extent: float,
        padding_factor: float,
        n_pad_cells: int,
    ) -> Any:
        """Build a TensorMesh with padding.

        Args:
            x_min, x_max: Data x-extent.
            y_min, y_max: Data y-extent.
            z_min: Top of model (typically observation elevation).
            cell_size: Core cell size.
            depth_extent: Total depth of model.
            padding_factor: Padding expansion factor.
            n_pad_cells: Number of padding cells.

        Returns:
            discretize.TensorMesh
        """
        import discretize

        x_extent = x_max - x_min
        y_extent = y_max - y_min

        # Core cells
        nx_core = int(np.ceil(x_extent / cell_size))
        ny_core = int(np.ceil(y_extent / cell_size))
        nz_core = int(np.ceil(depth_extent / cell_size))

        # Limit total cells to reasonable number
        max_core = 100
        if nx_core > max_core:
            cell_size_x = x_extent / max_core
            nx_core = max_core
        else:
            cell_size_x = cell_size

        if ny_core > max_core:
            cell_size_y = y_extent / max_core
            ny_core = max_core
        else:
            cell_size_y = cell_size

        if nz_core > max_core:
            cell_size_z = depth_extent / max_core
            nz_core = max_core
        else:
            cell_size_z = cell_size

        # Build cell widths with padding
        hx = self._build_cell_widths(nx_core, cell_size_x, n_pad_cells, padding_factor)
        hy = self._build_cell_widths(ny_core, cell_size_y, n_pad_cells, padding_factor)
        hz = self._build_cell_widths_depth(nz_core, cell_size_z, n_pad_cells, padding_factor)

        # Origin (bottom SW corner)
        x_origin = x_min - sum(hx[:n_pad_cells])
        y_origin = y_min - sum(hy[:n_pad_cells])
        z_origin = z_min - depth_extent - sum(hz[:n_pad_cells])

        mesh = discretize.TensorMesh([hx, hy, hz], origin=[x_origin, y_origin, z_origin])

        return mesh

    def _build_cell_widths(
        self,
        n_core: int,
        cell_size: float,
        n_pad: int,
        expansion_factor: float,
    ) -> "np.ndarray":
        """Build cell widths with symmetric padding.

        Args:
            n_core: Number of core cells.
            cell_size: Core cell size.
            n_pad: Number of padding cells on each side.
            expansion_factor: Geometric expansion factor for padding.

        Returns:
            Array of cell widths.
        """
        # Padding cells expand geometrically
        pad_widths = cell_size * expansion_factor ** np.arange(1, n_pad + 1)

        # Combine: padding (reversed) + core + padding
        core_widths = np.ones(n_core) * cell_size
        widths = np.concatenate([pad_widths[::-1], core_widths, pad_widths])

        return widths

    def _build_cell_widths_depth(
        self,
        n_core: int,
        cell_size: float,
        n_pad: int,
        expansion_factor: float,
    ) -> "np.ndarray":
        """Build cell widths for depth direction (padding only at bottom).

        Args:
            n_core: Number of core cells.
            cell_size: Core cell size.
            n_pad: Number of padding cells at bottom.
            expansion_factor: Geometric expansion factor.

        Returns:
            Array of cell widths (bottom to top).
        """
        pad_widths = cell_size * expansion_factor ** np.arange(1, n_pad + 1)
        core_widths = np.ones(n_core) * cell_size

        # Bottom padding + core (z increases upward)
        widths = np.concatenate([pad_widths[::-1], core_widths])

        return widths

    def _build_tree_mesh(
        self,
        locations: "np.ndarray",
        cell_size: float,
        depth_extent: float,
        padding_factor: float,
        x_extent: float,
        y_extent: float,
    ) -> Any:
        """Build a TreeMesh with octree refinement near data.

        Args:
            locations: Observation locations for refinement.
            cell_size: Minimum cell size (at finest level).
            depth_extent: Model depth.
            padding_factor: Padding extent factor.
            x_extent: Data x-extent.
            y_extent: Data y-extent.

        Returns:
            discretize.TreeMesh
        """
        import discretize

        # TreeMesh requires power-of-2 dimensions
        max_extent = max(x_extent, y_extent, depth_extent) * (1 + padding_factor)
        n_base = int(2 ** np.ceil(np.log2(max_extent / cell_size)))

        # Limit to reasonable size
        n_base = min(n_base, 512)

        h = [np.ones(n_base) * cell_size] * 3
        mesh = discretize.TreeMesh(h, origin="CCC")

        # Refine around data locations
        mesh.refine_points(locations, padding_cells_by_level=[4, 4, 2], finalize=False)

        # Refine surface
        mesh.refine_surface(
            locations,
            padding_cells_by_level=[4, 4, 2],
            finalize=False,
        )

        mesh.finalize()

        return mesh

    @staticmethod
    def mesh_summary(mesh: Any) -> dict[str, Any]:
        """Get a summary of mesh properties.

        Args:
            mesh: discretize mesh object.

        Returns:
            Dictionary with mesh statistics.
        """
        summary = {
            "n_cells": mesh.nC,
            "n_nodes": mesh.nN if hasattr(mesh, "nN") else None,
            "mesh_type": type(mesh).__name__,
        }

        if hasattr(mesh, "h"):
            summary["min_cell_size"] = min(h.min() for h in mesh.h)
            summary["max_cell_size"] = max(h.max() for h in mesh.h)

        if hasattr(mesh, "origin"):
            summary["origin"] = list(mesh.origin)

        return summary
