"""
InversionAI - Results & Comparison Page
Shows single-method results or side-by-side comparison depending on what was run.
"""

import streamlit as st
import numpy as np
import pandas as pd
import os
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

from ui.components.convergence_plot import render_convergence_comparison


def _has_model_array(results: dict) -> bool:
    """Check if results contain a valid numpy model array."""
    if not results:
        return False
    model = results.get("model")
    return model is not None and hasattr(model, "shape") and len(model.shape) == 3


def _read_model_file(filepath: str) -> np.ndarray | None:
    """Read a Tomofast-x model text file into a flat numpy array.
    First line is the number of cells, remaining lines are values.
    """
    try:
        values = []
        with open(filepath, "r") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    val = float(line)
                    if i == 0 and val == int(val) and val > 100:
                        # First line is cell count, skip
                        continue
                    values.append(val)
                except ValueError:
                    pass
        if values:
            return np.array(values)
    except Exception:
        pass
    return None


def _load_model_3d(results: dict) -> np.ndarray | None:
    """Load the Tomofast-x model file and reshape to 3D grid."""
    import re

    output_dir = results.get("output_dir", "")
    if not output_dir:
        return None

    # Find model file
    model_path = os.path.join(output_dir, "model", "grav_final_model_full.txt")
    if not os.path.exists(model_path):
        model_dir = os.path.join(output_dir, "model")
        if os.path.exists(model_dir):
            candidates = [f for f in os.listdir(model_dir) if "model" in f and f.endswith(".txt")]
            if candidates:
                model_path = os.path.join(model_dir, candidates[0])

    if not os.path.exists(model_path):
        return None

    model_values = _read_model_file(model_path)
    if model_values is None or len(model_values) == 0:
        return None

    # Get grid dimensions from Parfile_copy.txt in output dir
    nx, ny, nz = 0, 0, 0
    parfile_copy = os.path.join(output_dir, "Parfile_copy.txt")
    if os.path.exists(parfile_copy):
        with open(parfile_copy, "r") as f:
            content = f.read()
        match = re.search(r"modelGrid\.size\s*=\s*(\d+)\s+(\d+)\s+(\d+)", content)
        if match:
            nx, ny, nz = int(match.group(1)), int(match.group(2)), int(match.group(3))

    # Fallback: try known sizes
    if nx * ny * nz != len(model_values):
        known = {57057: (13, 133, 33), 4000: (20, 20, 10), 62500: (50, 50, 25)}
        n = len(model_values)
        if n in known:
            nx, ny, nz = known[n]
        else:
            # Factorize
            import math
            for z in range(int(n ** 0.33) + 5, 1, -1):
                if n % z == 0:
                    rem = n // z
                    for y in range(int(rem ** 0.5) + 5, 1, -1):
                        if rem % y == 0:
                            nx, ny, nz = rem // y, y, z
                            break
                    if nx > 0:
                        break

    if nx * ny * nz == len(model_values):
        # Reshape: Tomofast stores x-fastest (Fortran order)
        model_3d = model_values.reshape((nx, ny, nz), order='F')
        return model_3d

    return None


def _read_data_file(filepath: str) -> np.ndarray | None:
    """Read a Tomofast-x data file (x, y, z, value format)."""
    try:
        data = np.loadtxt(filepath)
        return data
    except Exception:
        pass
    return None


def render_metrics_table(results_tomo, results_simpeg):
    """Render metrics table that handles both result formats."""
    metrics = []

    if results_tomo:
        parsed = results_tomo.get("parsed", {})
        final_misfit = results_tomo.get("final_misfit")
        if final_misfit is None:
            history = results_tomo.get("misfit_history", [])
            final_misfit = history[-1] if history else None

        row = {"Algorithm": "⚡ Tomofast-x"}
        if final_misfit is not None:
            row["Final Data Cost"] = f"{final_misfit:.6f}"
        if parsed.get("final_rmse"):
            row["RMSE"] = f"{parsed['final_rmse']:.2e}"
        row["Iterations"] = results_tomo.get("iterations") or parsed.get("iterations_completed", "N/A")
        if parsed.get("model_min") is not None:
            row["Model Range"] = f"[{parsed['model_min']:.1f}, {parsed['model_max']:.1f}]"
        if parsed.get("memory_gb"):
            row["Memory (GB)"] = f"{parsed['memory_gb']:.4f}"
        metrics.append(row)

    if results_simpeg:
        final_misfit = results_simpeg.get("final_misfit")
        if final_misfit is None:
            history = results_simpeg.get("misfit_history", [])
            final_misfit = history[-1] if history else None

        row = {"Algorithm": "🔬 SimPEG"}
        if final_misfit is not None:
            row["Final Data Cost"] = f"{final_misfit:.6f}"
        row["Iterations"] = results_simpeg.get("iterations", "N/A")
        if results_simpeg.get("runtime"):
            row["Runtime (s)"] = f"{results_simpeg['runtime']:.1f}"
        metrics.append(row)

    if metrics:
        df = pd.DataFrame(metrics)
        st.dataframe(df, use_container_width=True, hide_index=True)


def render_convergence_section(results_tomo, results_simpeg):
    """Render convergence curves."""
    histories = {}
    if results_tomo and results_tomo.get("misfit_history"):
        histories["Tomofast-x"] = results_tomo["misfit_history"]
    if results_simpeg and results_simpeg.get("misfit_history"):
        histories["SimPEG"] = results_simpeg["misfit_history"]

    if histories:
        render_convergence_comparison(histories)
    else:
        st.info("No convergence data available.")


def render_data_fit_plot(results_tomo):
    """Render observed vs predicted data plot from Tomofast-x output."""
    output_dir = results_tomo.get("output_dir", "")
    if not output_dir:
        return

    # Try to load observed and predicted data
    obs_path = os.path.join(output_dir, "data", "grav_observed.txt")
    pred_path = os.path.join(output_dir, "data", "grav_final.txt")
    misfit_path = os.path.join(output_dir, "data", "grav_misfit.txt")

    if os.path.exists(obs_path) and os.path.exists(pred_path):
        obs_data = _read_data_file(obs_path)
        pred_data = _read_data_file(pred_path)

        if obs_data is not None and pred_data is not None:
            # Data format: x, y, z, value (or similar)
            n_cols_obs = obs_data.shape[1] if len(obs_data.shape) > 1 else 1
            n_cols_pred = pred_data.shape[1] if len(pred_data.shape) > 1 else 1

            # Get the value column (last column)
            if n_cols_obs >= 4:
                obs_vals = obs_data[:, 3]
                obs_x = obs_data[:, 0]
                obs_y = obs_data[:, 1]
            else:
                obs_vals = obs_data.flatten()
                obs_x = np.arange(len(obs_vals))
                obs_y = None

            if n_cols_pred >= 4:
                pred_vals = pred_data[:, 3]
            else:
                pred_vals = pred_data.flatten()

            # Ensure same length
            min_len = min(len(obs_vals), len(pred_vals))
            obs_vals = obs_vals[:min_len]
            pred_vals = pred_vals[:min_len]

            # Scatter: observed vs predicted
            fig_scatter = go.Figure()
            fig_scatter.add_trace(go.Scatter(
                x=obs_vals, y=pred_vals,
                mode="markers", name="Data Points",
                marker=dict(size=5, opacity=0.6),
            ))
            # 1:1 line
            val_range = [min(obs_vals.min(), pred_vals.min()), max(obs_vals.max(), pred_vals.max())]
            fig_scatter.add_trace(go.Scatter(
                x=val_range, y=val_range,
                mode="lines", name="1:1 line",
                line=dict(dash="dash", color="red"),
            ))
            fig_scatter.update_layout(
                title="Observed vs Predicted Data",
                xaxis_title="Observed",
                yaxis_title="Predicted",
                template="plotly_white",
                height=400,
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

            # Map view of observed data
            if obs_y is not None and n_cols_obs >= 4:
                fig_map = px.scatter(
                    x=obs_x[:min_len], y=obs_y[:min_len],
                    color=obs_vals,
                    color_continuous_scale="Viridis",
                    labels={"x": "Easting", "y": "Northing", "color": "Gravity (mGal)"},
                    title="Observed Gravity Data (map view)",
                )
                fig_map.update_layout(
                    height=400, template="plotly_white",
                    yaxis_scaleanchor="x",
                )
                st.plotly_chart(fig_map, use_container_width=True)

    # Misfit map
    if os.path.exists(misfit_path):
        misfit_data = _read_data_file(misfit_path)
        if misfit_data is not None and len(misfit_data.shape) > 1 and misfit_data.shape[1] >= 4:
            fig_misfit = px.scatter(
                x=misfit_data[:, 0], y=misfit_data[:, 1],
                color=misfit_data[:, 3],
                color_continuous_scale="RdBu_r",
                labels={"x": "Easting", "y": "Northing", "color": "Misfit"},
                title="Data Misfit (map view)",
            )
            fig_misfit.update_layout(
                height=400, template="plotly_white",
                yaxis_scaleanchor="x",
            )
            st.plotly_chart(fig_misfit, use_container_width=True)


def render_model_histogram(results_tomo):
    """Render histogram of model values from the output model file."""
    output_dir = results_tomo.get("output_dir", "")
    if not output_dir:
        return

    model_path = os.path.join(output_dir, "model", "grav_final_model_full.txt")
    if not os.path.exists(model_path):
        # Try finding any model file
        model_dir = os.path.join(output_dir, "model")
        if os.path.exists(model_dir):
            model_files = [f for f in os.listdir(model_dir) if f.endswith(".txt")]
            if model_files:
                model_path = os.path.join(model_dir, model_files[0])

    if os.path.exists(model_path):
        model_values = _read_model_file(model_path)
        if model_values is not None and len(model_values) > 0:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=model_values,
                nbinsx=80,
                name="Density Model",
                marker_color="#2563eb",
            ))
            fig.update_layout(
                title="Model Value Distribution (Density Contrast)",
                xaxis_title="Density (kg/m³)",
                yaxis_title="Count",
                template="plotly_white",
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Stats
            st.caption(
                f"Model stats: min={model_values.min():.2f}, max={model_values.max():.2f}, "
                f"mean={model_values.mean():.4f}, std={model_values.std():.4f}, "
                f"n_cells={len(model_values):,}"
            )


def render_output_files(results_tomo):
    """List and allow download of output files."""
    output_dir = results_tomo.get("output_dir", "")
    if not output_dir or not os.path.exists(output_dir):
        st.info("No output directory found.")
        return

    vtk_files = sorted(Path(output_dir).rglob("*.vtk"))
    txt_files = sorted(Path(output_dir).rglob("*.txt"))

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**📦 VTK Files (ParaView):**")
        for f in vtk_files:
            rel = f.relative_to(output_dir)
            size_kb = f.stat().st_size / 1024
            st.caption(f"• `{rel}` ({size_kb:.0f} KB)")

    with col2:
        st.markdown("**📄 Text Files:**")
        for f in txt_files:
            rel = f.relative_to(output_dir)
            size_kb = f.stat().st_size / 1024
            st.caption(f"• `{rel}` ({size_kb:.0f} KB)")


def generate_summary(results_tomo, results_simpeg) -> str:
    """Generate a natural language summary."""
    parts = ["## Inversion Results Summary\n"]

    if results_tomo and results_simpeg:
        parts.append("**Comparison mode:** Both Tomofast-x and SimPEG were run on the same dataset.\n")
    elif results_tomo:
        parts.append("**Single method:** Tomofast-x inversion results.\n")
    elif results_simpeg:
        parts.append("**Single method:** SimPEG inversion results.\n")

    if results_tomo:
        parsed = results_tomo.get("parsed", {})
        tomo_iters = results_tomo.get("iterations") or parsed.get("iterations_completed", "N/A")
        final_misfit = results_tomo.get("final_misfit")
        if final_misfit is None:
            history = results_tomo.get("misfit_history", [])
            final_misfit = history[-1] if history else None

        parts.append(f"**Tomofast-x:** {tomo_iters} major iterations completed.")
        if parsed.get("final_rmse"):
            parts.append(f"  - Final RMSE: {parsed['final_rmse']:.2e}")
        if final_misfit:
            parts.append(f"  - Final data cost: {final_misfit:.6f}")
        if parsed.get("model_min") is not None:
            parts.append(f"  - Model range: [{parsed['model_min']:.1f}, {parsed['model_max']:.1f}]")

    if results_simpeg:
        sim_iters = results_simpeg.get("iterations", "N/A")
        final_misfit = results_simpeg.get("final_misfit")
        parts.append(f"\n**SimPEG:** {sim_iters} iterations completed.")
        if final_misfit:
            parts.append(f"  - Final misfit: {final_misfit:.6f}")

    return "\n".join(parts)


def render_compare_page():
    """Render the Results / Comparison page."""
    results_tomo = st.session_state.get("results_tomofast")
    results_simpeg = st.session_state.get("results_simpeg")

    if results_tomo is None and results_simpeg is None:
        st.header("📊 Results")
        st.warning("⚠️ No inversion results available. Please run an inversion first.")
        if st.button("← Go to Run"):
            st.session_state.current_page = "run"
            st.rerun()
        return

    # Determine mode
    is_comparison = results_tomo is not None and results_simpeg is not None

    if is_comparison:
        st.header("📊 Comparison: Tomofast-x vs SimPEG")
    elif results_tomo:
        st.header("📊 Results: Tomofast-x Inversion")
    else:
        st.header("📊 Results: SimPEG Inversion")

    # Summary
    summary = generate_summary(results_tomo, results_simpeg)
    st.markdown(summary)
    st.markdown("---")

    # Metrics table
    st.subheader("📋 Metrics")
    render_metrics_table(results_tomo, results_simpeg)
    st.markdown("---")

    # Convergence
    st.subheader("📉 Convergence")
    render_convergence_section(results_tomo, results_simpeg)
    st.markdown("---")

    # Data fit plots (single method or comparison)
    if results_tomo and results_tomo.get("output_dir"):
        st.subheader("📈 Data Fit")
        render_data_fit_plot(results_tomo)
        st.markdown("---")

    # Model histogram
    if results_tomo and results_tomo.get("output_dir"):
        st.subheader("📊 Model Distribution")
        render_model_histogram(results_tomo)
        st.markdown("---")

    # 3D model cross-sections
    # Load 3D model from file if available
    model_3d_tomo = None
    model_3d_simpeg = None

    if results_tomo and results_tomo.get("output_dir"):
        model_3d_tomo = _load_model_3d(results_tomo)
    elif _has_model_array(results_tomo):
        model_3d_tomo = results_tomo["model"]

    if _has_model_array(results_simpeg):
        model_3d_simpeg = results_simpeg["model"]

    if model_3d_tomo is not None or model_3d_simpeg is not None:
        st.subheader("🌐 3D Model Visualization")

        model_ref = model_3d_tomo if model_3d_tomo is not None else model_3d_simpeg
        name_ref = "Tomofast-x" if model_3d_tomo is not None else "SimPEG"
        nx, ny, nz = model_ref.shape

        # Full 3D volume with isosurface/volume rendering
        _render_3d_model(model_ref, name_ref)

        st.markdown("---")

        # 3D model with section planes
        st.subheader("🔪 3D Cross-Section Slices")
        _render_3d_sections(model_ref, name_ref)

        st.markdown("---")

        st.subheader("🔍 2D Model Cross-Sections")

        st.caption(f"Model grid: {nx} × {ny} × {nz} cells ({nx*ny*nz:,} total)")

        sl_col1, sl_col2, sl_col3 = st.columns(3)
        with sl_col1:
            slice_z = st.slider("Depth Slice (Z)", 0, nz - 1, nz // 2)
        with sl_col2:
            slice_x = st.slider("X Slice", 0, nx - 1, nx // 2)
        with sl_col3:
            slice_y = st.slider("Y Slice", 0, ny - 1, ny // 2)

        # Determine if comparison mode
        if model_3d_tomo is not None and model_3d_simpeg is not None:
            # Side-by-side comparison
            tab_z, tab_x, tab_y = st.tabs(["Depth (XY)", "X-section (YZ)", "Y-section (XZ)"])
            with tab_z:
                c1, c2 = st.columns(2)
                with c1:
                    _render_slice(model_3d_tomo, "Tomofast-x", slice_z, axis="z")
                with c2:
                    _render_slice(model_3d_simpeg, "SimPEG", slice_z, axis="z")
            with tab_x:
                c1, c2 = st.columns(2)
                with c1:
                    _render_slice(model_3d_tomo, "Tomofast-x", slice_x, axis="x")
                with c2:
                    _render_slice(model_3d_simpeg, "SimPEG", slice_x, axis="x")
            with tab_y:
                c1, c2 = st.columns(2)
                with c1:
                    _render_slice(model_3d_tomo, "Tomofast-x", slice_y, axis="y")
                with c2:
                    _render_slice(model_3d_simpeg, "SimPEG", slice_y, axis="y")

            # Difference map
            if model_3d_tomo.shape == model_3d_simpeg.shape:
                st.subheader("🔀 Difference Map (Tomofast-x − SimPEG)")
                diff = model_3d_tomo - model_3d_simpeg
                _render_slice(diff, "Difference", slice_z, axis="z")
        else:
            # Single method - show all 3 cross-sections
            model = model_3d_tomo if model_3d_tomo is not None else model_3d_simpeg
            name = "Tomofast-x" if model_3d_tomo is not None else "SimPEG"

            tab_z, tab_x, tab_y = st.tabs(["Depth Slice (XY)", "X-Section (YZ)", "Y-Section (XZ)"])
            with tab_z:
                _render_slice(model, f"{name} — Depth Z={slice_z}", slice_z, axis="z")
            with tab_x:
                _render_slice(model, f"{name} — X={slice_x}", slice_x, axis="x")
            with tab_y:
                _render_slice(model, f"{name} — Y={slice_y}", slice_y, axis="y")

        st.markdown("---")

    elif results_tomo and results_tomo.get("output_dir"):
        st.subheader("🔍 3D Model Cross-Sections")
        st.warning("Could not reshape model into 3D grid. Check model file and grid dimensions.")
        st.markdown("---")

    # AI Interpretation - Mineralization Leads
    if model_3d_tomo is not None or model_3d_simpeg is not None:
        model_interp = model_3d_tomo if model_3d_tomo is not None else model_3d_simpeg
        st.subheader("🧠 AI Geological Interpretation")
        _render_interpretation(model_interp)
        st.markdown("---")

    # Output files
    if results_tomo and results_tomo.get("output_dir"):
        st.subheader("📁 Output Files")
        render_output_files(results_tomo)
        st.markdown("---")

    # Raw stdout
    if results_tomo and results_tomo.get("stdout"):
        with st.expander("🖥️ Tomofast-x Raw Output"):
            st.code(results_tomo["stdout"], language="text")

    # Navigation
    if st.button("← Back to Run"):
        st.session_state.current_page = "run"
        st.rerun()


def _render_interpretation(model: np.ndarray):
    """Generate AI interpretation of density model highlighting potential mineralization leads."""
    nx, ny, nz = model.shape
    total_cells = nx * ny * nz

    # Statistical analysis
    model_flat = model.flatten()
    mean_val = float(model_flat.mean())
    std_val = float(model_flat.std())
    vmin, vmax = float(model_flat.min()), float(model_flat.max())

    # Define anomaly thresholds
    high_threshold = mean_val + 1.5 * std_val
    low_threshold = mean_val - 1.5 * std_val
    very_high_threshold = mean_val + 2.5 * std_val
    very_low_threshold = mean_val - 2.5 * std_val

    # Identify anomalous regions
    high_mask = model > high_threshold
    low_mask = model < low_threshold
    very_high_mask = model > very_high_threshold
    very_low_mask = model < very_low_threshold

    n_high = int(high_mask.sum())
    n_low = int(low_mask.sum())
    n_very_high = int(very_high_mask.sum())
    n_very_low = int(very_low_mask.sum())

    pct_high = 100 * n_high / total_cells
    pct_low = 100 * n_low / total_cells

    # Find centroids of anomalous clusters
    high_anomalies = _find_anomaly_clusters(model, high_threshold, "high")
    low_anomalies = _find_anomaly_clusters(model, low_threshold, "low")

    # Disclaimer
    st.warning(
        "⚠️ **DISCLAIMER:** This is an AI-generated interpretation for guidance purposes only. "
        "**DMN SOLUTIONS** does not take responsibility for this interpretation. "
        "All geological conclusions should be validated by qualified geoscientists "
        "before any exploration decisions are made."
    )

    # Summary metrics
    st.markdown("#### 📊 Model Statistics")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Mean Density", f"{mean_val:.4f} kg/m³")
    with m_col2:
        st.metric("Std Deviation", f"{std_val:.4f}")
    with m_col3:
        st.metric("High Anomaly Cells", f"{n_high:,} ({pct_high:.1f}%)")
    with m_col4:
        st.metric("Low Anomaly Cells", f"{n_low:,} ({pct_low:.1f}%)")

    # Interpretation text
    st.markdown("#### 🎯 Potential Mineralization Leads")

    if high_anomalies:
        st.markdown("**🔴 High-Density Anomalies** (potential targets: iron ore, massive sulfides, mafic/ultramafic intrusions)")
        for i, anomaly in enumerate(high_anomalies[:5], 1):
            depth_label = _depth_category(anomaly["z_center"], nz)
            st.markdown(
                f"- **Lead H{i}:** Centroid at cell ({anomaly['x_center']}, {anomaly['y_center']}, {anomaly['z_center']}) | "
                f"Peak density: **{anomaly['peak_value']:.4f}** kg/m³ | "
                f"Volume: ~{anomaly['volume_cells']:,} cells | "
                f"Depth: {depth_label}"
            )
    else:
        st.info("No significant high-density anomalies detected above threshold.")

    if low_anomalies:
        st.markdown("**🔵 Low-Density Anomalies** (potential targets: sedimentary basins, alteration zones, voids, felsic intrusions)")
        for i, anomaly in enumerate(low_anomalies[:5], 1):
            depth_label = _depth_category(anomaly["z_center"], nz)
            st.markdown(
                f"- **Lead L{i}:** Centroid at cell ({anomaly['x_center']}, {anomaly['y_center']}, {anomaly['z_center']}) | "
                f"Peak density: **{anomaly['peak_value']:.4f}** kg/m³ | "
                f"Volume: ~{anomaly['volume_cells']:,} cells | "
                f"Depth: {depth_label}"
            )
    else:
        st.info("No significant low-density anomalies detected below threshold.")

    # Geological context
    st.markdown("#### 📝 Interpretation Summary")

    interpretation_lines = []
    interpretation_lines.append(
        f"The inversion model spans a grid of **{nx}×{ny}×{nz}** cells with density contrast "
        f"ranging from **{vmin:.4f}** to **{vmax:.4f}** kg/m³ (mean: {mean_val:.4f}, σ: {std_val:.4f})."
    )

    if n_very_high > 0:
        interpretation_lines.append(
            f"**{n_very_high:,} cells** ({100*n_very_high/total_cells:.2f}%) exhibit very high density contrast (>{very_high_threshold:.4f}), "
            f"suggesting compact, dense geological bodies that may represent mineralized zones (e.g., massive sulfides, BIF, or mafic intrusions)."
        )

    if n_very_low > 0:
        interpretation_lines.append(
            f"**{n_very_low:,} cells** ({100*n_very_low/total_cells:.2f}%) exhibit very low density contrast (<{very_low_threshold:.4f}), "
            f"potentially indicating alteration halos, sedimentary infill, or structural voids that may be associated with hydrothermal systems."
        )

    if high_anomalies and low_anomalies:
        # Check for spatial association (high near low = possible mineralization system)
        for h in high_anomalies[:3]:
            for l in low_anomalies[:3]:
                dist = np.sqrt(
                    (h["x_center"] - l["x_center"])**2 +
                    (h["y_center"] - l["y_center"])**2 +
                    (h["z_center"] - l["z_center"])**2
                )
                if dist < max(nx, ny, nz) * 0.25:
                    interpretation_lines.append(
                        f"⭐ **Spatial association detected:** High-density Lead H and Low-density Lead L "
                        f"are in proximity (distance ~{dist:.0f} cells), which may indicate a "
                        f"mineralization system with dense ore adjacent to an alteration/structural corridor."
                    )
                    break
            else:
                continue
            break

    if not high_anomalies and not low_anomalies:
        interpretation_lines.append(
            "The model shows relatively uniform density distribution without significant anomalies exceeding 1.5σ. "
            "This may indicate a homogeneous geological setting or that the inversion regularization has smoothed subtle features. "
            "Consider adjusting inversion parameters or increasing iterations."
        )

    for line in interpretation_lines:
        st.markdown(line)

    # Recommendation
    st.markdown("#### 💡 Recommendations")
    recommendations = []
    if high_anomalies:
        recommendations.append("• Prioritize drilling targets at high-density Lead H1 for potential massive ore confirmation")
    if low_anomalies:
        recommendations.append("• Investigate low-density zones for structural controls and alteration mapping")
    if high_anomalies or low_anomalies:
        recommendations.append("• Cross-reference with magnetic data and surface geology for target validation")
        recommendations.append("• Consider follow-up ground geophysics (e.g., EM, IP) over anomaly clusters")
    recommendations.append("• Compare with regional geological maps and known mineral occurrences")

    for rec in recommendations:
        st.markdown(rec)


def _find_anomaly_clusters(model: np.ndarray, threshold: float, anomaly_type: str) -> list:
    """Find clusters of anomalous cells and return their properties."""
    nx, ny, nz = model.shape

    if anomaly_type == "high":
        mask = model > threshold
    else:
        mask = model < threshold

    if not mask.any():
        return []

    # Simple connected-component-like approach using scipy if available
    try:
        from scipy import ndimage
        labeled, n_features = ndimage.label(mask)

        clusters = []
        for label_id in range(1, min(n_features + 1, 20)):  # Limit to top 20
            cluster_mask = labeled == label_id
            volume = int(cluster_mask.sum())
            if volume < 3:  # Skip tiny clusters
                continue

            coords = np.argwhere(cluster_mask)
            centroid = coords.mean(axis=0)

            values_in_cluster = model[cluster_mask]
            if anomaly_type == "high":
                peak = float(values_in_cluster.max())
            else:
                peak = float(values_in_cluster.min())

            clusters.append({
                "x_center": int(round(centroid[0])),
                "y_center": int(round(centroid[1])),
                "z_center": int(round(centroid[2])),
                "volume_cells": volume,
                "peak_value": peak,
            })

        # Sort by volume (largest first)
        clusters.sort(key=lambda c: c["volume_cells"], reverse=True)
        return clusters[:5]

    except ImportError:
        # Fallback without scipy: just find the global extremum location
        if anomaly_type == "high":
            idx_flat = model.argmax()
            peak = float(model.max())
        else:
            idx_flat = model.argmin()
            peak = float(model.min())

        coords = np.unravel_index(idx_flat, model.shape)
        volume = int(mask.sum())

        return [{
            "x_center": int(coords[0]),
            "y_center": int(coords[1]),
            "z_center": int(coords[2]),
            "volume_cells": volume,
            "peak_value": peak,
        }]


def _depth_category(z_idx: int, nz: int) -> str:
    """Categorize depth based on z-index position."""
    ratio = z_idx / max(nz - 1, 1)
    if ratio < 0.25:
        return "Shallow"
    elif ratio < 0.5:
        return "Intermediate-Shallow"
    elif ratio < 0.75:
        return "Intermediate-Deep"
    else:
        return "Deep"


def _render_3d_model(model: np.ndarray, title: str):
    """Render a full 3D interactive isosurface visualization of the model."""
    nx, ny, nz = model.shape

    # Opacity slider
    opacity_val = st.slider("Opacity", 0.1, 1.0, 0.9, 0.05, key="iso_opacity")

    # Create coordinate arrays
    X, Y, Z = np.mgrid[0:nx, 0:ny, 0:nz]

    # Flatten for volume rendering
    values = model.flatten(order='C')
    x_flat = X.flatten()
    y_flat = Y.flatten()
    z_flat = Z.flatten()

    # Use isosurface for 3D visualization (efficient for large models)
    vmin, vmax = float(values.min()), float(values.max())

    # Focus on the anomalous body: use tighter iso range centered on high values
    # Skip the background (near-zero) values to highlight the body
    mean_val = float(values.mean())
    std_val = float(values.std())
    isomin = mean_val + 0.5 * std_val  # Only show above-average values
    isomax = vmax - 0.05 * (vmax - vmin)

    # If model has both positive and negative anomalies, show both
    if vmin < mean_val - std_val:
        isomin = vmin + 0.05 * (vmax - vmin)

    n_levels = 8  # More surfaces for better body definition

    fig = go.Figure(data=go.Isosurface(
        x=x_flat, y=y_flat, z=z_flat,
        value=values,
        isomin=isomin,
        isomax=isomax,
        surface_count=n_levels,
        colorscale='RdBu_r',
        caps=dict(x_show=True, y_show=True, z_show=True),
        opacity=opacity_val,
        colorbar=dict(title="Density (kg/m³)", thickness=20, len=0.7),
        hovertemplate="X: %{x}<br>Y: %{y}<br>Z: %{z}<br>Value: %{value:.4f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text=f"3D Model — {title}", x=0.5, font=dict(size=16)),
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z (depth)",
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.6, y=-1.6, z=-1.2),
                up=dict(x=0, y=0, z=-1),
            ),
        ),
        height=700,
        margin=dict(l=0, r=0, t=50, b=0),
    )

    st.plotly_chart(fig, use_container_width=True)


def _render_3d_sections(model: np.ndarray, title: str):
    """Render 3D model with interactive orthogonal slice planes."""
    nx, ny, nz = model.shape

    # Sliders for slice positions and opacity
    sl_col1, sl_col2, sl_col3, sl_col4 = st.columns(4)
    with sl_col1:
        sec_z = st.slider("Z-plane (depth)", 0, nz - 1, nz // 2, key="sec3d_z")
    with sl_col2:
        sec_x = st.slider("X-plane", 0, nx - 1, nx // 2, key="sec3d_x")
    with sl_col3:
        sec_y = st.slider("Y-plane", 0, ny - 1, ny // 2, key="sec3d_y")
    with sl_col4:
        sec_opacity = st.slider("Opacity", 0.1, 1.0, 0.9, 0.05, key="sec3d_opacity")

    fig = go.Figure()

    vmin, vmax = float(model.min()), float(model.max())

    # Z-slice (horizontal depth slice)
    z_slice = model[:, :, sec_z]
    x_coords, y_coords = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    fig.add_trace(go.Surface(
        x=x_coords, y=y_coords, z=np.full_like(x_coords, sec_z, dtype=float),
        surfacecolor=z_slice,
        colorscale='RdBu_r', cmin=vmin, cmax=vmax,
        showscale=True,
        colorbar=dict(title="Density (kg/m³)", thickness=20, len=0.7, x=1.02),
        opacity=sec_opacity,
        name=f"Z={sec_z}",
        hovertemplate="X: %{x}<br>Y: %{y}<br>Z: %{z}<br>Value: %{surfacecolor:.4f}<extra>Z-plane</extra>",
    ))

    # X-slice (vertical section along X)
    x_slice = model[sec_x, :, :]
    y_coords_xslice, z_coords_xslice = np.meshgrid(np.arange(ny), np.arange(nz), indexing='ij')
    fig.add_trace(go.Surface(
        x=np.full_like(y_coords_xslice, sec_x, dtype=float),
        y=y_coords_xslice, z=z_coords_xslice,
        surfacecolor=x_slice,
        colorscale='RdBu_r', cmin=vmin, cmax=vmax,
        showscale=False,
        opacity=sec_opacity,
        name=f"X={sec_x}",
        hovertemplate="X: %{x}<br>Y: %{y}<br>Z: %{z}<br>Value: %{surfacecolor:.4f}<extra>X-plane</extra>",
    ))

    # Y-slice (vertical section along Y)
    y_slice = model[:, sec_y, :]
    x_coords_yslice, z_coords_yslice = np.meshgrid(np.arange(nx), np.arange(nz), indexing='ij')
    fig.add_trace(go.Surface(
        x=x_coords_yslice,
        y=np.full_like(x_coords_yslice, sec_y, dtype=float),
        z=z_coords_yslice,
        surfacecolor=y_slice,
        colorscale='RdBu_r', cmin=vmin, cmax=vmax,
        showscale=False,
        opacity=sec_opacity,
        name=f"Y={sec_y}",
        hovertemplate="X: %{x}<br>Y: %{y}<br>Z: %{z}<br>Value: %{surfacecolor:.4f}<extra>Y-plane</extra>",
    ))

    fig.update_layout(
        title=dict(text=f"3D Cross-Sections — {title}", x=0.5, font=dict(size=16)),
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z (depth)",
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.6, y=-1.6, z=-1.2),
                up=dict(x=0, y=0, z=-1),
            ),
            xaxis=dict(range=[0, nx-1]),
            yaxis=dict(range=[0, ny-1]),
            zaxis=dict(range=[0, nz-1]),
        ),
        height=700,
        margin=dict(l=0, r=0, t=50, b=0),
        showlegend=True,
        legend=dict(x=0.01, y=0.99),
    )

    st.plotly_chart(fig, use_container_width=True)


def _render_slice(model: np.ndarray, title: str, idx: int, axis: str = "z"):
    """Render a cross-section slice as filled contour with contour lines."""
    if axis == "z":
        idx = min(idx, model.shape[2] - 1)
        slice_data = model[:, :, idx]
        xlabel, ylabel = "X", "Y"
    elif axis == "x":
        idx = min(idx, model.shape[0] - 1)
        slice_data = model[idx, :, :]
        xlabel, ylabel = "Y", "Z"
    elif axis == "y":
        idx = min(idx, model.shape[1] - 1)
        slice_data = model[:, idx, :]
        xlabel, ylabel = "X", "Z"
    else:
        return

    fig = go.Figure()

    # Filled contour with contour lines
    fig.add_trace(go.Contour(
        z=slice_data.T,
        colorscale="RdBu_r",
        contours=dict(
            coloring="heatmap",
            showlabels=True,
            showlines=True,
            labelfont=dict(size=10, color="black"),
        ),
        line=dict(width=1, color="black"),
        ncontours=20,
        colorbar=dict(title="Density (kg/m³)", thickness=15, len=0.9),
        hovertemplate=f"{xlabel}: %{{x}}<br>{ylabel}: %{{y}}<br>Density: %{{z:.4f}}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        height=400,
        margin=dict(l=50, r=20, t=40, b=40),
        yaxis=dict(scaleanchor="x"),
    )
    st.plotly_chart(fig, use_container_width=True)
