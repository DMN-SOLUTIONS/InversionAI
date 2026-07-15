"""
InversionAI - Streamlit UI Main Entry Point
A visual interface for non-experts to run geophysical inversions.

Layout:
  [Sidebar] | [Center: Instructions & Controls] | [Right: Visualizations / Logs]
"""

import sys
from pathlib import Path

# Add project root to path so 'ui', 'skills', 'data', etc. are importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="InversionAI",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_session_state():
    """Initialize session state variables for workflow progress."""
    defaults = {
        # Navigation
        "current_page": "setup",
        # Data state
        "uploaded_file": None,
        "dataset": None,
        "data_validated": False,
        "data_format": None,
        # Algorithm selection
        "use_tomofast": False,
        "use_simpeg": False,
        # Configuration
        "config": {},
        "config_ready": False,
        # Execution state
        "run_status": "idle",  # idle, running, completed, failed, cancelled
        "progress_tomofast": 0.0,
        "progress_simpeg": 0.0,
        "misfit_history_tomofast": [],
        "misfit_history_simpeg": [],
        "current_iteration_tomofast": 0,
        "current_iteration_simpeg": 0,
        "total_iterations": 50,
        "logs": [],
        "start_time": None,
        # Results
        "results_tomofast": None,
        "results_simpeg": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_sidebar():
    """Render the sidebar with algorithm selection, data status, and run controls."""
    with st.sidebar:
        st.image("https://img.shields.io/badge/InversionAI-v1.0-blue", width=150)
        st.title("🌍 InversionAI")
        st.markdown("---")

        # Workflow progress
        st.subheader("📋 Workflow Progress")
        steps = {
            "setup": ("1. Data Upload", st.session_state.get("data_validated", False)),
            "configure": ("2. Configuration", st.session_state.get("config_ready", False)),
            "run": ("3. Execution", st.session_state.get("run_status") == "completed"),
            "compare": ("4. Results", st.session_state.get("results_tomofast") is not None or st.session_state.get("results_simpeg") is not None),
        }
        for _key, (label, done) in steps.items():
            icon = "✅" if done else "⬜"
            st.markdown(f"{icon} {label}")

        st.markdown("---")

        # Data upload status
        st.subheader("📂 Data Status")
        if st.session_state.get("dataset") is not None:
            st.success("Data loaded successfully")
            fmt = st.session_state.get("data_format", "Unknown")
            st.caption(f"Format: {fmt}")
        else:
            st.info("No data uploaded yet")

        st.markdown("---")

        # Algorithm selection summary
        st.subheader("⚙️ Algorithms")
        use_tomo = st.session_state.get("use_tomofast", False)
        use_simpeg = st.session_state.get("use_simpeg", False)
        if use_tomo:
            st.markdown("✅ Tomofast-x")
        if use_simpeg:
            st.markdown("✅ SimPEG")
        if not use_tomo and not use_simpeg:
            st.caption("No algorithm selected")

        st.markdown("---")

        # Navigation
        st.subheader("🧭 Navigation")
        pages = ["setup", "configure", "run", "compare"]
        page_labels = {
            "setup": "📤 Setup & Upload",
            "configure": "🔧 Configure",
            "run": "▶️ Run Inversion",
            "compare": "📊 Compare Results",
        }
        for page in pages:
            if st.button(page_labels[page], key=f"nav_{page}"):
                st.session_state.current_page = page
                st.rerun()


def render_center_panel():
    """Render the center panel with instructions and feature controls."""
    page = st.session_state.get("current_page", "setup")

    if page == "setup":
        from ui.pages.setup import render_setup_page
        render_setup_page()
    elif page == "configure":
        from ui.pages.configure import render_configure_page
        render_configure_page()
    elif page == "run":
        from ui.pages.run import render_run_page
        render_run_page()
    elif page == "compare":
        from ui.pages.compare import render_compare_page
        render_compare_page()
    else:
        st.error(f"Unknown page: {page}")


def render_visualization_panel():
    """Render the visualization panel (top-right).

    Shows stacked visualizations:
    1. Station map (always, when data loaded)
    2. Mesh extent preview (when on configure page or data loaded)
    3. Inversion results (convergence + model slice, after run)
    """
    page = st.session_state.get("current_page", "setup")
    dataset = st.session_state.get("dataset")
    results = st.session_state.get("results_tomofast")

    st.markdown("#### 📊 Visualization")

    if dataset is None and results is None:
        st.info("Load data or run an inversion to see visualizations here.")
        return

    # 1. Station map (always show when data is loaded)
    if dataset is not None:
        _render_data_viz(dataset)

    # 2. Mesh extent preview (show when data is loaded)
    if dataset is not None and "X" in dataset.columns:
        st.markdown("---")
        _render_mesh_extent_viz(dataset)

    # 3. Results visualization (show after inversion completes)
    if results and st.session_state.get("run_status") == "completed":
        st.markdown("---")
        _render_results_viz(results)

        # 4. 2D Model Cross-Sections
        st.markdown("---")
        _render_model_cross_sections(results)


def _render_data_viz(dataset):
    """Render data visualization: station map and data summary."""
    import plotly.express as px
    import pandas as pd

    if "X" in dataset.columns and "Y" in dataset.columns:
        color_col = "Value" if "Value" in dataset.columns else None
        fig = px.scatter(
            dataset,
            x="X",
            y="Y",
            color=color_col,
            color_continuous_scale="Viridis",
            title="Station Locations",
            labels={"X": "Easting (m)", "Y": "Northing (m)"},
        )
        fig.update_layout(
            height=350,
            margin=dict(l=30, r=10, t=40, b=30),
            template="plotly_white",
            yaxis_scaleanchor="x",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Quick stats
        n_pts = len(dataset)
        cols = st.columns(3)
        with cols[0]:
            st.metric("Stations", f"{n_pts:,}")
        with cols[1]:
            if "Value" in dataset.columns:
                st.metric("Min Value", f"{dataset['Value'].min():.4f}")
        with cols[2]:
            if "Value" in dataset.columns:
                st.metric("Max Value", f"{dataset['Value'].max():.4f}")
    else:
        st.dataframe(dataset.head(10), use_container_width=True)


def _render_mesh_extent_viz(dataset):
    """Render mesh extent preview: 3D wireframe box over station locations."""
    import plotly.graph_objects as go
    import numpy as np

    if "X" not in dataset.columns or "Y" not in dataset.columns:
        _render_data_viz(dataset)
        return

    x_min, x_max = dataset["X"].min(), dataset["X"].max()
    y_min, y_max = dataset["Y"].min(), dataset["Y"].max()

    # Mesh extent with padding
    pad_x = (x_max - x_min) * 0.1
    pad_y = (y_max - y_min) * 0.1
    mesh_x_min = x_min - pad_x
    mesh_x_max = x_max + pad_x
    mesh_y_min = y_min - pad_y
    mesh_y_max = y_max + pad_y

    z_top = dataset["Z"].max() if "Z" in dataset.columns else 0
    depth_extent = max(x_max - x_min, y_max - y_min) * 0.5
    z_bot = z_top - depth_extent

    # Wireframe box
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

    # Station points
    z_vals = dataset["Z"].values if "Z" in dataset.columns else np.full(len(dataset), z_top)
    fig.add_trace(go.Scatter3d(
        x=dataset["X"], y=dataset["Y"], z=z_vals,
        mode="markers",
        marker=dict(size=3, color="blue", opacity=0.6),
        name="Stations",
    ))

    # Mesh wireframe
    fig.add_trace(go.Scatter3d(
        x=box_x, y=box_y, z=box_z,
        mode="lines",
        line=dict(color="red", width=3),
        name="Mesh Extent",
    ))

    fig.update_layout(
        title="Mesh Extent Preview",
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
        ),
        template="plotly_white",
        height=400,
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=True,
        legend=dict(x=0.01, y=0.99),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Mesh info
    cols = st.columns(3)
    with cols[0]:
        st.metric("X Extent", f"{mesh_x_max - mesh_x_min:.0f} m")
    with cols[1]:
        st.metric("Y Extent", f"{mesh_y_max - mesh_y_min:.0f} m")
    with cols[2]:
        st.metric("Depth", f"{depth_extent:.0f} m")


def _render_results_viz(results):
    """Render inversion results: convergence + model slice."""
    import plotly.graph_objects as go
    import numpy as np
    import os
    import re

    # Convergence plot
    misfit_history = results.get("misfit_history", [])
    if misfit_history:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=misfit_history,
            mode="lines+markers",
            name="Data Cost",
            line=dict(color="#2563eb", width=2),
            marker=dict(size=4),
        ))
        fig.update_layout(
            title="Convergence",
            xaxis_title="Iteration",
            yaxis_title="Data Cost",
            height=250,
            margin=dict(l=40, r=10, t=40, b=30),
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Summary metrics
    parsed = results.get("parsed", {})
    cols = st.columns(3)
    with cols[0]:
        st.metric("Iterations", parsed.get("iterations_completed", "N/A"))
    with cols[1]:
        rmse = parsed.get("final_rmse")
        st.metric("RMSE", f"{rmse:.2e}" if rmse else "N/A")
    with cols[2]:
        if parsed.get("model_min") is not None:
            st.metric("Model Range", f"{parsed['model_min']:.1f} – {parsed['model_max']:.1f}")


def _render_model_cross_sections(results):
    """Render 2D model cross-sections with interactive slice sliders."""
    import plotly.graph_objects as go
    import numpy as np
    import os
    import re

    output_dir = results.get("output_dir", "")
    if not output_dir or not os.path.exists(output_dir):
        return

    # Load model
    model_path = os.path.join(output_dir, "model", "grav_final_model_full.txt")
    if not os.path.exists(model_path):
        model_dir = os.path.join(output_dir, "model")
        if os.path.exists(model_dir):
            candidates = [f for f in os.listdir(model_dir) if "model" in f and f.endswith(".txt")]
            if candidates:
                model_path = os.path.join(model_dir, candidates[0])

    if not os.path.exists(model_path):
        return

    # Read model values
    values = []
    with open(model_path, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                val = float(line)
                if i == 0 and val == int(val) and val > 100:
                    continue
                values.append(val)
            except ValueError:
                pass

    if not values:
        return

    model_values = np.array(values)

    # Get grid dimensions
    nx, ny, nz = 0, 0, 0
    parfile_copy = os.path.join(output_dir, "Parfile_copy.txt")
    if os.path.exists(parfile_copy):
        with open(parfile_copy, "r") as f:
            content = f.read()
        match = re.search(r"modelGrid\.size\s*=\s*(\d+)\s+(\d+)\s+(\d+)", content)
        if match:
            nx, ny, nz = int(match.group(1)), int(match.group(2)), int(match.group(3))

    if nx * ny * nz != len(model_values):
        # Try known sizes
        n = len(model_values)
        known = {57057: (13, 133, 33), 4000: (20, 20, 10), 62500: (50, 50, 25)}
        if n in known:
            nx, ny, nz = known[n]
        else:
            for z in range(int(n ** 0.33) + 5, 1, -1):
                if n % z == 0:
                    rem = n // z
                    for y in range(int(rem ** 0.5) + 5, 1, -1):
                        if rem % y == 0:
                            nx, ny, nz = rem // y, y, z
                            break
                    if nx > 0:
                        break

    if nx * ny * nz != len(model_values):
        return

    model_3d = model_values.reshape((nx, ny, nz), order='F')

    st.markdown("**🔍 2D Model Cross-Sections**")
    st.caption(f"Grid: {nx} × {ny} × {nz} cells")

    # Slice sliders
    sl_col1, sl_col2, sl_col3 = st.columns(3)
    with sl_col1:
        slice_z = st.slider("Z (depth)", 0, nz - 1, nz // 2, key="right_slice_z")
    with sl_col2:
        slice_x = st.slider("X", 0, nx - 1, nx // 2, key="right_slice_x")
    with sl_col3:
        slice_y = st.slider("Y", 0, ny - 1, ny // 2, key="right_slice_y")

    # Depth slice (XY)
    slice_data = model_3d[:, :, slice_z]
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=slice_data.T,
        colorscale="RdBu_r",
        colorbar=dict(title="Density", thickness=12),
    ))
    fig.update_layout(
        title=f"Depth Slice Z={slice_z}",
        xaxis_title="X", yaxis_title="Y",
        height=280,
        margin=dict(l=40, r=10, t=35, b=30),
        yaxis=dict(scaleanchor="x"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # X-section (YZ)
    slice_data = model_3d[slice_x, :, :]
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=slice_data.T,
        colorscale="RdBu_r",
        colorbar=dict(title="Density", thickness=12),
    ))
    fig.update_layout(
        title=f"X-Section X={slice_x}",
        xaxis_title="Y", yaxis_title="Z",
        height=250,
        margin=dict(l=40, r=10, t=35, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Y-section (XZ)
    slice_data = model_3d[:, slice_y, :]
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=slice_data.T,
        colorscale="RdBu_r",
        colorbar=dict(title="Density", thickness=12),
    ))
    fig.update_layout(
        title=f"Y-Section Y={slice_y}",
        xaxis_title="X", yaxis_title="Z",
        height=250,
        margin=dict(l=40, r=10, t=35, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_log_panel():
    """Render the log panel (bottom-right).

    Shows execution events with filtering, stats, and full detail view.
    """
    st.markdown("#### 📝 Execution Log")

    try:
        from core import event_logger, LogSource, LogLevel

        # Filter controls
        filter_col1, filter_col2, filter_col3 = st.columns([2, 2, 1])
        with filter_col1:
            source_filter = st.multiselect(
                "Filter by source",
                options=["All", "Agent", "Skill", "Tool", "Workflow", "Data", "System"],
                default=["All"],
                key="right_log_source_filter",
            )
        with filter_col2:
            level_filter = st.multiselect(
                "Filter by level",
                options=["All", "Info", "Success", "Warning", "Error", "Debug"],
                default=["All"],
                key="right_log_level_filter",
            )
        with filter_col3:
            show_details = st.checkbox("Show details", value=True, key="right_log_show_details")

        # Get and filter events
        all_events = event_logger.get_all()

        if "All" not in source_filter:
            source_map = {
                "Agent": LogSource.AGENT, "Skill": LogSource.SKILL, "Tool": LogSource.TOOL,
                "Workflow": LogSource.WORKFLOW, "Data": LogSource.DATA, "System": LogSource.SYSTEM,
            }
            selected_sources = [source_map[s] for s in source_filter if s in source_map]
            all_events = [e for e in all_events if e.source in selected_sources]

        if "All" not in level_filter:
            level_map = {
                "Info": LogLevel.INFO, "Success": LogLevel.SUCCESS, "Warning": LogLevel.WARNING,
                "Error": LogLevel.ERROR, "Debug": LogLevel.DEBUG,
            }
            selected_levels = [level_map[l] for l in level_filter if l in level_map]
            all_events = [e for e in all_events if e.level in selected_levels]

        # Display stats
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        with stat_col1:
            st.metric("Total Events", event_logger.count())
        with stat_col2:
            st.metric("Filtered", len(all_events))
        with stat_col3:
            errors = len([e for e in all_events if e.level == LogLevel.ERROR])
            st.metric("Errors", errors)
        with stat_col4:
            warnings = len([e for e in all_events if e.level == LogLevel.WARNING])
            st.metric("Warnings", warnings)

        # Render log entries
        with st.expander(f"Full Log ({len(all_events)} events)", expanded=True):
            if all_events:
                if show_details:
                    log_text = "\n".join(e.format_detailed() for e in all_events)
                else:
                    log_text = "\n".join(e.format_short() for e in all_events)
                st.code(log_text, language="text")
            else:
                st.caption("No log entries. Run an inversion to see execution details.")

    except ImportError:
        st.caption("Log system not available.")


def main():
    """Main application entry point."""
    init_session_state()
    render_sidebar()

    # Two-column layout: Center (controls) | Right (viz + logs)
    col_center, col_right = st.columns([3, 2])

    with col_center:
        render_center_panel()

    with col_right:
        # Top: Visualizations
        render_visualization_panel()

        st.markdown("---")

        # Bottom: Logs
        render_log_panel()


if __name__ == "__main__":
    main()
