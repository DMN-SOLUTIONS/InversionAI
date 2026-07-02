"""
InversionAI - Configuration Page
Guided parameter selection with tooltips, mesh config, and magnetic field inputs.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go


def get_default_config(df) -> dict:
    """Generate default configuration based on data extent."""
    config = {
        "n_iterations": 50,
        "regularization_alpha": 1.0,
        "depth_weighting_enabled": True,
        "depth_weighting_exponent": 2.0,
        "model_bounds_min": -1.0,
        "model_bounds_max": 1.0,
        "mesh_mode": "auto",
        "n_cells_x": 50,
        "n_cells_y": 50,
        "n_cells_z": 25,
        "padding_cells": 5,
        # Magnetic field parameters
        "inclination": -60.0,
        "declination": 0.0,
        "field_strength": 50000.0,
    }

    if df is not None and "X" in df.columns and "Y" in df.columns:
        x_range = df["X"].max() - df["X"].min()
        y_range = df["Y"].max() - df["Y"].min()
        # Auto-calculate cell sizes
        config["cell_size_x"] = x_range / config["n_cells_x"] if x_range > 0 else 100
        config["cell_size_y"] = y_range / config["n_cells_y"] if y_range > 0 else 100
        config["cell_size_z"] = min(config["cell_size_x"], config["cell_size_y"])

        if "Z" in df.columns:
            z_range = df["Z"].max() - df["Z"].min()
            config["depth_extent"] = max(z_range * 2, x_range * 0.5)
        else:
            config["depth_extent"] = x_range * 0.5

    return config


def render_mesh_preview(df, config: dict):
    """Render a 3D wireframe preview of mesh extent overlaid on data points."""
    if df is None or "X" not in df.columns or "Y" not in df.columns:
        st.warning("Cannot preview mesh: no data loaded.")
        return

    x_min, x_max = df["X"].min(), df["X"].max()
    y_min, y_max = df["Y"].min(), df["Y"].max()

    # Add padding
    pad_x = (x_max - x_min) * 0.1
    pad_y = (y_max - y_min) * 0.1
    mesh_x_min = x_min - pad_x
    mesh_x_max = x_max + pad_x
    mesh_y_min = y_min - pad_y
    mesh_y_max = y_max + pad_y

    z_top = df["Z"].max() if "Z" in df.columns else 0
    z_bot = z_top - config.get("depth_extent", (x_max - x_min) * 0.5)

    # Create wireframe box
    box_x = [mesh_x_min, mesh_x_max, mesh_x_max, mesh_x_min, mesh_x_min,
             mesh_x_min, mesh_x_max, mesh_x_max, mesh_x_min, mesh_x_min,
             None, mesh_x_max, mesh_x_max, None, mesh_x_max, mesh_x_max,
             None, mesh_x_min, mesh_x_min]
    box_y = [mesh_y_min, mesh_y_min, mesh_y_max, mesh_y_max, mesh_y_min,
             mesh_y_min, mesh_y_min, mesh_y_max, mesh_y_max, mesh_y_min,
             None, mesh_y_min, mesh_y_min, None, mesh_y_max, mesh_y_max,
             None, mesh_y_max, mesh_y_max]
    box_z = [z_top, z_top, z_top, z_top, z_top,
             z_bot, z_bot, z_bot, z_bot, z_bot,
             None, z_top, z_bot, None, z_top, z_bot,
             None, z_top, z_bot]

    fig = go.Figure()

    # Data points
    scatter_kwargs = dict(
        x=df["X"],
        y=df["Y"],
        mode="markers",
        marker=dict(size=3, color="blue", opacity=0.6),
        name="Stations",
    )
    if "Z" in df.columns:
        fig.add_trace(go.Scatter3d(z=df["Z"], **scatter_kwargs))
        fig.add_trace(go.Scatter3d(
            x=box_x, y=box_y, z=box_z,
            mode="lines",
            line=dict(color="red", width=3),
            name="Mesh Extent",
        ))
        fig.update_layout(scene=dict(
            xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)"
        ))
    else:
        fig.add_trace(go.Scatter3d(
            z=[z_top] * len(df), **scatter_kwargs
        ))
        fig.add_trace(go.Scatter3d(
            x=box_x, y=box_y, z=box_z,
            mode="lines",
            line=dict(color="red", width=3),
            name="Mesh Extent",
        ))
        fig.update_layout(scene=dict(
            xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)"
        ))

    fig.update_layout(
        title="Mesh Extent Preview",
        template="plotly_white",
        height=450,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_configure_page():
    """Render the Configuration page."""
    st.header("🔧 Inversion Configuration")

    # Check prerequisites
    if not st.session_state.get("data_validated", False):
        st.warning("⚠️ Please upload and validate data first.")
        if st.button("← Go to Setup"):
            st.session_state.current_page = "setup"
            st.rerun()
        return

    df = st.session_state.get("dataset")
    defaults = get_default_config(df)

    st.info(
        "💡 Configure your inversion parameters below. Hover over labels for "
        "explanations. Use **'Use Defaults'** for a quick start with recommended values."
    )

    # Use defaults button
    col_def1, col_def2, col_def3 = st.columns([1, 1, 3])
    with col_def1:
        if st.button("🎯 Use Defaults", type="secondary", use_container_width=True):
            st.session_state.config = defaults
            st.session_state.config_ready = True
            st.success("Default configuration applied!")

    st.markdown("---")

    # Initialize config from session or defaults
    config = st.session_state.get("config", {})
    if not config:
        config = defaults

    # === Inversion Parameters ===
    st.subheader("📐 Inversion Parameters")

    param_col1, param_col2 = st.columns(2)

    with param_col1:
        config["n_iterations"] = st.slider(
            "Number of Iterations",
            min_value=10,
            max_value=200,
            value=config.get("n_iterations", defaults["n_iterations"]),
            step=5,
            help="More iterations may improve fit but increase computation time. "
                 "50-100 is typical for most problems.",
        )

        config["regularization_alpha"] = st.slider(
            "Regularization (α)",
            min_value=0.01,
            max_value=10.0,
            value=float(config.get("regularization_alpha", defaults["regularization_alpha"])),
            step=0.01,
            format="%.2f",
            help="Controls smoothness vs data fit tradeoff. Higher = smoother model. "
                 "Start with 1.0 and adjust based on results.",
        )

    with param_col2:
        config["depth_weighting_enabled"] = st.toggle(
            "Depth Weighting",
            value=config.get("depth_weighting_enabled", defaults["depth_weighting_enabled"]),
            help="Compensates for loss of sensitivity with depth. "
                 "Recommended ON for gravity/magnetic inversions.",
        )

        if config["depth_weighting_enabled"]:
            config["depth_weighting_exponent"] = st.slider(
                "Depth Weighting Exponent (β)",
                min_value=0.5,
                max_value=4.0,
                value=float(config.get("depth_weighting_exponent", defaults["depth_weighting_exponent"])),
                step=0.1,
                format="%.1f",
                help="Exponent for depth weighting function. "
                     "β=2 for gravity, β=3 for magnetics is typical.",
            )

        st.markdown("**Model Bounds**")
        bound_col1, bound_col2 = st.columns(2)
        with bound_col1:
            config["model_bounds_min"] = st.number_input(
                "Min",
                value=float(config.get("model_bounds_min", defaults["model_bounds_min"])),
                format="%.4f",
                help="Minimum allowed model value (e.g., density contrast in g/cc)",
            )
        with bound_col2:
            config["model_bounds_max"] = st.number_input(
                "Max",
                value=float(config.get("model_bounds_max", defaults["model_bounds_max"])),
                format="%.4f",
                help="Maximum allowed model value (e.g., density contrast in g/cc)",
            )

    st.markdown("---")

    # === Mesh Configuration ===
    st.subheader("🧊 Mesh Configuration")

    config["mesh_mode"] = st.radio(
        "Mesh Generation Mode",
        options=["auto", "manual"],
        index=0 if config.get("mesh_mode", "auto") == "auto" else 1,
        horizontal=True,
        help="'Auto' generates mesh from data extent. 'Manual' lets you specify cell counts.",
    )

    if config["mesh_mode"] == "manual":
        mesh_col1, mesh_col2, mesh_col3 = st.columns(3)
        with mesh_col1:
            config["n_cells_x"] = st.number_input(
                "N Cells X",
                min_value=5,
                max_value=200,
                value=int(config.get("n_cells_x", defaults["n_cells_x"])),
                help="Number of cells in X (Easting) direction",
            )
        with mesh_col2:
            config["n_cells_y"] = st.number_input(
                "N Cells Y",
                min_value=5,
                max_value=200,
                value=int(config.get("n_cells_y", defaults["n_cells_y"])),
                help="Number of cells in Y (Northing) direction",
            )
        with mesh_col3:
            config["n_cells_z"] = st.number_input(
                "N Cells Z",
                min_value=5,
                max_value=100,
                value=int(config.get("n_cells_z", defaults["n_cells_z"])),
                help="Number of cells in Z (Depth) direction",
            )

        total_cells = config["n_cells_x"] * config["n_cells_y"] * config["n_cells_z"]
        st.caption(f"Total mesh cells: **{total_cells:,}**")
        if total_cells > 500_000:
            st.warning("⚠️ Large mesh (>500k cells) may be slow. Consider reducing cell count.")
    else:
        st.caption(
            "Mesh will be auto-generated from data extent with recommended cell sizes."
        )
        if df is not None and "X" in df.columns:
            x_range = df["X"].max() - df["X"].min()
            y_range = df["Y"].max() - df["Y"].min()
            st.caption(
                f"Data extent: X={x_range:.0f}m, Y={y_range:.0f}m → "
                f"Estimated cells: {defaults['n_cells_x']}×{defaults['n_cells_y']}×{defaults['n_cells_z']}"
            )

    st.markdown("---")

    # === Magnetic Field Parameters ===
    data_format = st.session_state.get("data_format", "")
    is_magnetic = "mag" in data_format.lower() if data_format else False

    st.subheader("🧲 Magnetic Field Parameters")
    if not is_magnetic:
        st.info("ℹ️ These parameters are primarily for magnetic inversions. Skip if running gravity only.")

    mag_col1, mag_col2, mag_col3 = st.columns(3)
    with mag_col1:
        config["inclination"] = st.number_input(
            "Inclination (°)",
            min_value=-90.0,
            max_value=90.0,
            value=float(config.get("inclination", defaults["inclination"])),
            format="%.1f",
            help="Magnetic field inclination. Negative in Southern Hemisphere.",
        )
    with mag_col2:
        config["declination"] = st.number_input(
            "Declination (°)",
            min_value=-180.0,
            max_value=180.0,
            value=float(config.get("declination", defaults["declination"])),
            format="%.1f",
            help="Magnetic field declination (deviation from geographic north).",
        )
    with mag_col3:
        config["field_strength"] = st.number_input(
            "Field Strength (nT)",
            min_value=10000.0,
            max_value=70000.0,
            value=float(config.get("field_strength", defaults["field_strength"])),
            format="%.0f",
            help="Total magnetic field intensity in nanotesla.",
        )

    st.markdown("---")

    # === Mesh Preview ===
    st.subheader("👁️ Mesh Extent Preview")
    render_mesh_preview(df, config)

    # Save configuration
    st.session_state.config = config

    st.markdown("---")

    # Action buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("← Back to Setup", use_container_width=True):
            st.session_state.current_page = "setup"
            st.rerun()

    with col3:
        if st.button(
            "▶️ Start Inversion",
            type="primary",
            use_container_width=True,
        ):
            st.session_state.config = config
            st.session_state.config_ready = True
            st.session_state.total_iterations = config["n_iterations"]
            st.session_state.current_page = "run"
            st.rerun()
