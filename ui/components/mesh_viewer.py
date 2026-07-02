"""
InversionAI - Mesh Viewer Component
Reusable 3D mesh wireframe and model volume rendering functions.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from typing import Any, Optional


def render_mesh_3d(mesh_config: Any):
    """Render a 3D wireframe of the mesh extent.

    Args:
        mesh_config: Dictionary or object with keys:
            - origin (tuple/list): (x0, y0, z0)
            - extent (tuple/list): (dx, dy, dz) total size
            - n_cells (tuple/list): (nx, ny, nz) cell counts
            Or shorthand: n_cells_x, n_cells_y, n_cells_z, cell_size_x, etc.
    """
    if mesh_config is None:
        st.info("No mesh configuration available.")
        return

    # Extract mesh parameters
    if isinstance(mesh_config, dict):
        cfg = mesh_config
    elif hasattr(mesh_config, "__dict__"):
        cfg = mesh_config.__dict__
    else:
        st.warning("Unsupported mesh_config format.")
        return

    # Determine mesh bounds
    if "origin" in cfg and "extent" in cfg:
        x0, y0, z0 = cfg["origin"]
        dx, dy, dz = cfg["extent"]
    else:
        nx = cfg.get("n_cells_x", 50)
        ny = cfg.get("n_cells_y", 50)
        nz = cfg.get("n_cells_z", 25)
        cs_x = cfg.get("cell_size_x", 100)
        cs_y = cfg.get("cell_size_y", 100)
        cs_z = cfg.get("cell_size_z", 100)
        x0 = cfg.get("origin_x", 0)
        y0 = cfg.get("origin_y", 0)
        z0 = cfg.get("origin_z", 0)
        dx = nx * cs_x
        dy = ny * cs_y
        dz = nz * cs_z

    x1, y1, z1 = x0 + dx, y0 + dy, z0 - dz  # Z goes down for depth

    # Build wireframe box edges
    # 8 vertices of a box
    vertices_x = [x0, x1, x1, x0, x0, x1, x1, x0]
    vertices_y = [y0, y0, y1, y1, y0, y0, y1, y1]
    vertices_z = [z0, z0, z0, z0, z1, z1, z1, z1]

    # Edges as line segments (pairs of vertex indices)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # top face
        (4, 5), (5, 6), (6, 7), (7, 4),  # bottom face
        (0, 4), (1, 5), (2, 6), (3, 7),  # vertical edges
    ]

    edge_x, edge_y, edge_z = [], [], []
    for i, j in edges:
        edge_x.extend([vertices_x[i], vertices_x[j], None])
        edge_y.extend([vertices_y[i], vertices_y[j], None])
        edge_z.extend([vertices_z[i], vertices_z[j], None])

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode="lines",
        line=dict(color="red", width=4),
        name="Mesh Extent",
    ))

    # Add corner markers
    fig.add_trace(go.Scatter3d(
        x=vertices_x, y=vertices_y, z=vertices_z,
        mode="markers",
        marker=dict(size=4, color="red"),
        name="Corners",
        showlegend=False,
    ))

    fig.update_layout(
        title="Mesh Extent (3D Wireframe)",
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode="data",
        ),
        height=450,
        template="plotly_white",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Show mesh info
    info_col1, info_col2, info_col3 = st.columns(3)
    with info_col1:
        st.caption(f"X: {x0:.0f} → {x0+dx:.0f} m ({dx:.0f} m)")
    with info_col2:
        st.caption(f"Y: {y0:.0f} → {y0+dy:.0f} m ({dy:.0f} m)")
    with info_col3:
        st.caption(f"Z: {z0:.0f} → {z1:.0f} m ({dz:.0f} m depth)")


def render_model_3d(
    model: Optional[np.ndarray],
    mesh: Any = None,
    colormap: str = "RdBu_r",
):
    """Render a 3D volume rendering of the inversion model.

    Args:
        model: 3D numpy array of model values (nx, ny, nz).
        mesh: Optional mesh configuration for axis labels/scaling.
        colormap: Plotly colorscale name.
    """
    if model is None:
        st.info("No model data available for 3D rendering.")
        return

    if model.ndim != 3:
        st.warning(f"Expected 3D array, got {model.ndim}D.")
        return

    nx, ny, nz = model.shape

    # Create coordinate grids
    X, Y, Z = np.mgrid[0:nx, 0:ny, 0:nz]

    # Determine isosurface range
    vmin, vmax = np.percentile(model, [5, 95])
    if abs(vmax - vmin) < 1e-10:
        st.warning("Model values are nearly uniform — nothing to render.")
        return

    fig = go.Figure(data=go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=model.flatten(),
        isomin=vmin,
        isomax=vmax,
        opacity=0.2,
        surface_count=15,
        colorscale=colormap,
        caps=dict(x_show=True, y_show=True, z_show=True),
    ))

    fig.update_layout(
        title="3D Model Volume",
        scene=dict(
            xaxis_title="X cells",
            yaxis_title="Y cells",
            zaxis_title="Z cells",
            aspectmode="data",
        ),
        height=500,
        template="plotly_white",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Model statistics
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    with stat_col1:
        st.metric("Min", f"{model.min():.6f}")
    with stat_col2:
        st.metric("Max", f"{model.max():.6f}")
    with stat_col3:
        st.metric("Mean", f"{model.mean():.6f}")
    with stat_col4:
        st.metric("RMS", f"{np.sqrt(np.mean(model**2)):.6f}")
