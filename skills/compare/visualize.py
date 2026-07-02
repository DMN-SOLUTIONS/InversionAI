"""
Visualization module for model comparison.

Generates side-by-side comparison plots using Plotly including:
- Cross-sections (XY, XZ, YZ planes)
- Depth slices
- Misfit convergence curves
- Histograms of model values
- Difference maps
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def generate_comparison_plots(
    model_a: "np.ndarray",
    model_b: "np.ndarray",
    common_grid: dict[str, Any],
    misfit_a: list[float],
    misfit_b: list[float],
    label_a: str = "Model A",
    label_b: str = "Model B",
    depth_slices: list[float] | None = None,
    output_dir: str = "./plots",
    plot_format: str = "html",
) -> list[str]:
    """Generate all comparison visualization plots.

    Args:
        model_a: First model values on common grid.
        model_b: Second model values on common grid.
        common_grid: Grid specification with coordinates and shape.
        misfit_a: Misfit history for first model.
        misfit_b: Misfit history for second model.
        label_a: Label for first model.
        label_b: Label for second model.
        depth_slices: Z-values for depth slice plots.
        output_dir: Directory to save plots.
        plot_format: Output format ('html' or 'png').

    Returns:
        List of paths to generated plot files.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plot_paths: list[str] = []

    # Reshape models to 3D
    shape = common_grid["shape"]
    nx, ny, nz = shape

    n_expected = nx * ny * nz
    if len(model_a) != n_expected or len(model_b) != n_expected:
        logger.warning(
            f"Model size mismatch with grid. Expected {n_expected}, "
            f"got A={len(model_a)}, B={len(model_b)}. Truncating/padding."
        )
        model_a = _resize_model(model_a, n_expected)
        model_b = _resize_model(model_b, n_expected)

    model_a_3d = model_a.reshape(nx, ny, nz)
    model_b_3d = model_b.reshape(nx, ny, nz)
    diff_3d = model_a_3d - model_b_3d

    x = common_grid["x"]
    y = common_grid["y"]
    z = common_grid["z"]

    # 1. Misfit convergence plot
    path = _plot_convergence(misfit_a, misfit_b, label_a, label_b, output_path, plot_format)
    if path:
        plot_paths.append(path)

    # 2. Model value histograms
    path = _plot_histograms(model_a, model_b, label_a, label_b, output_path, plot_format)
    if path:
        plot_paths.append(path)

    # 3. Cross-sections
    # XY cross-section (horizontal slice at mid-depth)
    mid_z = nz // 2
    path = _plot_cross_section_xy(
        model_a_3d, model_b_3d, diff_3d, x, y, mid_z, z[mid_z] if mid_z < len(z) else 0,
        label_a, label_b, output_path, plot_format,
    )
    if path:
        plot_paths.append(path)

    # XZ cross-section (vertical slice at mid-y)
    mid_y = ny // 2
    path = _plot_cross_section_xz(
        model_a_3d, model_b_3d, diff_3d, x, z, mid_y, y[mid_y] if mid_y < len(y) else 0,
        label_a, label_b, output_path, plot_format,
    )
    if path:
        plot_paths.append(path)

    # YZ cross-section (vertical slice at mid-x)
    mid_x = nx // 2
    path = _plot_cross_section_yz(
        model_a_3d, model_b_3d, diff_3d, y, z, mid_x, x[mid_x] if mid_x < len(x) else 0,
        label_a, label_b, output_path, plot_format,
    )
    if path:
        plot_paths.append(path)

    # 4. Depth slices
    if depth_slices:
        for depth in depth_slices:
            # Find closest z-index
            z_idx = _find_nearest_index(z, depth)
            if z_idx is not None:
                path = _plot_depth_slice(
                    model_a_3d, model_b_3d, diff_3d, x, y, z_idx, depth,
                    label_a, label_b, output_path, plot_format,
                )
                if path:
                    plot_paths.append(path)

    # 5. Difference map
    path = _plot_difference_map(diff_3d, x, y, z, label_a, label_b, output_path, plot_format)
    if path:
        plot_paths.append(path)

    logger.info(f"Generated {len(plot_paths)} comparison plots in {output_dir}")
    return plot_paths


def _resize_model(model: "np.ndarray", target_size: int) -> "np.ndarray":
    """Resize model array to target size (pad with zeros or truncate)."""
    if len(model) >= target_size:
        return model[:target_size]
    else:
        padded = np.zeros(target_size)
        padded[: len(model)] = model
        return padded


def _find_nearest_index(array: "np.ndarray", value: float) -> int | None:
    """Find index of nearest value in array."""
    if len(array) == 0:
        return None
    idx = int(np.argmin(np.abs(array - value)))
    return idx


def _save_figure(fig: Any, output_path: Path, name: str, plot_format: str) -> str:
    """Save a plotly figure to file.

    Args:
        fig: Plotly figure object.
        output_path: Output directory.
        name: Base filename (without extension).
        plot_format: 'html' or 'png'.

    Returns:
        Path to saved file.
    """
    if plot_format == "html":
        filepath = str(output_path / f"{name}.html")
        fig.write_html(filepath)
    elif plot_format == "png":
        filepath = str(output_path / f"{name}.png")
        try:
            fig.write_image(filepath)
        except Exception as e:
            logger.warning(f"PNG export failed ({e}), falling back to HTML")
            filepath = str(output_path / f"{name}.html")
            fig.write_html(filepath)
    else:
        filepath = str(output_path / f"{name}.html")
        fig.write_html(filepath)

    logger.debug(f"Saved plot: {filepath}")
    return filepath


def _plot_convergence(
    misfit_a: list[float],
    misfit_b: list[float],
    label_a: str,
    label_b: str,
    output_path: Path,
    plot_format: str,
) -> str | None:
    """Plot misfit convergence curves."""
    import plotly.graph_objects as go

    if not misfit_a and not misfit_b:
        return None

    fig = go.Figure()

    if misfit_a:
        fig.add_trace(go.Scatter(
            x=list(range(1, len(misfit_a) + 1)),
            y=misfit_a,
            mode="lines+markers",
            name=label_a,
            line=dict(color="blue", width=2),
            marker=dict(size=4),
        ))

    if misfit_b:
        fig.add_trace(go.Scatter(
            x=list(range(1, len(misfit_b) + 1)),
            y=misfit_b,
            mode="lines+markers",
            name=label_b,
            line=dict(color="red", width=2),
            marker=dict(size=4),
        ))

    fig.update_layout(
        title="Misfit Convergence Comparison",
        xaxis_title="Iteration",
        yaxis_title="Data Misfit",
        yaxis_type="log",
        template="plotly_white",
        legend=dict(x=0.7, y=0.95),
    )

    return _save_figure(fig, output_path, "convergence", plot_format)


def _plot_histograms(
    model_a: "np.ndarray",
    model_b: "np.ndarray",
    label_a: str,
    label_b: str,
    output_path: Path,
    plot_format: str,
) -> str:
    """Plot model value histograms."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=2, subplot_titles=[label_a, label_b])

    fig.add_trace(
        go.Histogram(x=model_a, nbinsx=50, name=label_a, marker_color="blue", opacity=0.7),
        row=1, col=1,
    )
    fig.add_trace(
        go.Histogram(x=model_b, nbinsx=50, name=label_b, marker_color="red", opacity=0.7),
        row=1, col=2,
    )

    fig.update_layout(
        title="Model Value Distribution",
        template="plotly_white",
        showlegend=False,
    )
    fig.update_xaxes(title_text="Model Value", row=1, col=1)
    fig.update_xaxes(title_text="Model Value", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=1)

    return _save_figure(fig, output_path, "histograms", plot_format)


def _plot_cross_section_xy(
    model_a_3d: "np.ndarray",
    model_b_3d: "np.ndarray",
    diff_3d: "np.ndarray",
    x: "np.ndarray",
    y: "np.ndarray",
    z_idx: int,
    z_value: float,
    label_a: str,
    label_b: str,
    output_path: Path,
    plot_format: str,
) -> str:
    """Plot XY cross-section (horizontal slice)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[label_a, label_b, "Difference"],
    )

    # Common color scale
    vmin = min(model_a_3d[:, :, z_idx].min(), model_b_3d[:, :, z_idx].min())
    vmax = max(model_a_3d[:, :, z_idx].max(), model_b_3d[:, :, z_idx].max())

    fig.add_trace(
        go.Heatmap(z=model_a_3d[:, :, z_idx].T, x=x, y=y, zmin=vmin, zmax=vmax,
                   colorscale="Viridis", showscale=False),
        row=1, col=1,
    )
    fig.add_trace(
        go.Heatmap(z=model_b_3d[:, :, z_idx].T, x=x, y=y, zmin=vmin, zmax=vmax,
                   colorscale="Viridis", showscale=True),
        row=1, col=2,
    )
    fig.add_trace(
        go.Heatmap(z=diff_3d[:, :, z_idx].T, x=x, y=y,
                   colorscale="RdBu_r", zmid=0, showscale=True),
        row=1, col=3,
    )

    fig.update_layout(
        title=f"XY Cross-Section at z={z_value:.1f}m",
        template="plotly_white",
    )

    return _save_figure(fig, output_path, f"cross_section_xy_z{z_value:.0f}", plot_format)


def _plot_cross_section_xz(
    model_a_3d: "np.ndarray",
    model_b_3d: "np.ndarray",
    diff_3d: "np.ndarray",
    x: "np.ndarray",
    z: "np.ndarray",
    y_idx: int,
    y_value: float,
    label_a: str,
    label_b: str,
    output_path: Path,
    plot_format: str,
) -> str:
    """Plot XZ cross-section (vertical slice along x)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[label_a, label_b, "Difference"],
    )

    slice_a = model_a_3d[:, y_idx, :]
    slice_b = model_b_3d[:, y_idx, :]
    slice_diff = diff_3d[:, y_idx, :]

    vmin = min(slice_a.min(), slice_b.min())
    vmax = max(slice_a.max(), slice_b.max())

    fig.add_trace(
        go.Heatmap(z=slice_a.T, x=x, y=z, zmin=vmin, zmax=vmax,
                   colorscale="Viridis", showscale=False),
        row=1, col=1,
    )
    fig.add_trace(
        go.Heatmap(z=slice_b.T, x=x, y=z, zmin=vmin, zmax=vmax,
                   colorscale="Viridis", showscale=True),
        row=1, col=2,
    )
    fig.add_trace(
        go.Heatmap(z=slice_diff.T, x=x, y=z,
                   colorscale="RdBu_r", zmid=0, showscale=True),
        row=1, col=3,
    )

    fig.update_layout(
        title=f"XZ Cross-Section at y={y_value:.1f}m",
        template="plotly_white",
    )

    return _save_figure(fig, output_path, f"cross_section_xz_y{y_value:.0f}", plot_format)


def _plot_cross_section_yz(
    model_a_3d: "np.ndarray",
    model_b_3d: "np.ndarray",
    diff_3d: "np.ndarray",
    y: "np.ndarray",
    z: "np.ndarray",
    x_idx: int,
    x_value: float,
    label_a: str,
    label_b: str,
    output_path: Path,
    plot_format: str,
) -> str:
    """Plot YZ cross-section (vertical slice along y)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[label_a, label_b, "Difference"],
    )

    slice_a = model_a_3d[x_idx, :, :]
    slice_b = model_b_3d[x_idx, :, :]
    slice_diff = diff_3d[x_idx, :, :]

    vmin = min(slice_a.min(), slice_b.min())
    vmax = max(slice_a.max(), slice_b.max())

    fig.add_trace(
        go.Heatmap(z=slice_a.T, x=y, y=z, zmin=vmin, zmax=vmax,
                   colorscale="Viridis", showscale=False),
        row=1, col=1,
    )
    fig.add_trace(
        go.Heatmap(z=slice_b.T, x=y, y=z, zmin=vmin, zmax=vmax,
                   colorscale="Viridis", showscale=True),
        row=1, col=2,
    )
    fig.add_trace(
        go.Heatmap(z=slice_diff.T, x=y, y=z,
                   colorscale="RdBu_r", zmid=0, showscale=True),
        row=1, col=3,
    )

    fig.update_layout(
        title=f"YZ Cross-Section at x={x_value:.1f}m",
        template="plotly_white",
    )

    return _save_figure(fig, output_path, f"cross_section_yz_x{x_value:.0f}", plot_format)


def _plot_depth_slice(
    model_a_3d: "np.ndarray",
    model_b_3d: "np.ndarray",
    diff_3d: "np.ndarray",
    x: "np.ndarray",
    y: "np.ndarray",
    z_idx: int,
    depth: float,
    label_a: str,
    label_b: str,
    output_path: Path,
    plot_format: str,
) -> str:
    """Plot a depth slice comparison."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[label_a, label_b, "Difference"],
    )

    slice_a = model_a_3d[:, :, z_idx]
    slice_b = model_b_3d[:, :, z_idx]
    slice_diff = diff_3d[:, :, z_idx]

    vmin = min(slice_a.min(), slice_b.min())
    vmax = max(slice_a.max(), slice_b.max())

    fig.add_trace(
        go.Heatmap(z=slice_a.T, x=x, y=y, zmin=vmin, zmax=vmax,
                   colorscale="Viridis", showscale=False),
        row=1, col=1,
    )
    fig.add_trace(
        go.Heatmap(z=slice_b.T, x=x, y=y, zmin=vmin, zmax=vmax,
                   colorscale="Viridis", showscale=True),
        row=1, col=2,
    )
    fig.add_trace(
        go.Heatmap(z=slice_diff.T, x=x, y=y,
                   colorscale="RdBu_r", zmid=0, showscale=True),
        row=1, col=3,
    )

    fig.update_layout(
        title=f"Depth Slice at {depth:.0f}m",
        template="plotly_white",
    )

    return _save_figure(fig, output_path, f"depth_slice_{depth:.0f}m", plot_format)


def _plot_difference_map(
    diff_3d: "np.ndarray",
    x: "np.ndarray",
    y: "np.ndarray",
    z: "np.ndarray",
    label_a: str,
    label_b: str,
    output_path: Path,
    plot_format: str,
) -> str:
    """Plot maximum absolute difference projected along each axis."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Max |diff| along Z", "Max |diff| along Y", "Max |diff| along X"],
    )

    abs_diff = np.abs(diff_3d)

    # Project along Z (plan view)
    proj_z = abs_diff.max(axis=2)
    fig.add_trace(
        go.Heatmap(z=proj_z.T, x=x, y=y, colorscale="Hot", showscale=False),
        row=1, col=1,
    )

    # Project along Y (XZ view)
    proj_y = abs_diff.max(axis=1)
    fig.add_trace(
        go.Heatmap(z=proj_y.T, x=x, y=z, colorscale="Hot", showscale=False),
        row=1, col=2,
    )

    # Project along X (YZ view)
    proj_x = abs_diff.max(axis=0)
    fig.add_trace(
        go.Heatmap(z=proj_x.T, x=y, y=z, colorscale="Hot", showscale=True),
        row=1, col=3,
    )

    fig.update_layout(
        title=f"Maximum Absolute Difference ({label_a} - {label_b})",
        template="plotly_white",
    )

    return _save_figure(fig, output_path, "difference_map", plot_format)
