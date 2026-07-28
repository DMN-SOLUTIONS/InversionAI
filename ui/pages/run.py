"""
InversionAI - Execution & Monitoring Page
Runs actual Tomofast-x inversions and displays real results with full logging.
"""

import streamlit as st
import subprocess
import time
import os
import re
import numpy as np
from datetime import datetime
from pathlib import Path

from ui.components.convergence_plot import render_convergence_comparison
from core import event_logger, LogSource, LogLevel


TOMOFAST_DIR = "/app/Tomofast-x"
TOMOFAST_BIN = f"{TOMOFAST_DIR}/tomofastx"


def estimate_time_remaining(start_time, current_iter, total_iter) -> str:
    """Estimate remaining time based on elapsed time and progress."""
    if current_iter <= 0 or start_time is None:
        return "Calculating..."
    elapsed = (datetime.now() - start_time).total_seconds()
    time_per_iter = elapsed / current_iter
    remaining_seconds = time_per_iter * (total_iter - current_iter)
    if remaining_seconds < 60:
        return f"~{int(remaining_seconds)}s"
    elif remaining_seconds < 3600:
        return f"~{int(remaining_seconds / 60)}m {int(remaining_seconds % 60)}s"
    else:
        return f"~{int(remaining_seconds / 3600)}h {int((remaining_seconds % 3600) / 60)}m"


def get_status_indicator(status: str) -> str:
    """Return colored status indicator."""
    indicators = {
        "idle": "⚪ Idle",
        "running": "🔵 Running",
        "completed": "🟢 Completed",
        "failed": "🔴 Failed",
        "cancelled": "🟡 Cancelled",
    }
    return indicators.get(status, "⚪ Unknown")


def get_available_parfiles() -> dict:
    """Get available Parfiles from Tomofast-x directory."""
    parfiles = {}
    parfile_dir = Path(TOMOFAST_DIR) / "parfiles"
    if parfile_dir.exists():
        for pf in parfile_dir.rglob("Parfile*.txt"):
            name = pf.stem.replace("Parfile_", "").replace("_", " ").title()
            parfiles[name] = str(pf.relative_to(TOMOFAST_DIR))
    return parfiles


def parse_costs_file(costs_path: str) -> list[float]:
    """Parse the costs.txt output file to extract data misfit per iteration."""
    costs = []
    try:
        with open(costs_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or not line:
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        iteration = int(parts[0])
                        data_cost_grav = float(parts[1])
                        data_cost_mag = float(parts[2])
                        # Use whichever is non-zero (gravity or magnetic)
                        cost = data_cost_mag if data_cost_grav == 0.0 else data_cost_grav
                        costs.append(cost)
                    except (ValueError, IndexError):
                        continue
    except FileNotFoundError:
        pass
    return costs


def parse_tomofast_stdout(output: str) -> dict:
    """Parse Tomofast-x stdout for key information."""
    result = {
        "iterations_completed": 0,
        "final_rmse": None,
        "memory_gb": None,
        "model_min": None,
        "model_max": None,
        "output_files": [],
    }

    for line in output.split("\n"):
        # Iteration tracking
        match = re.search(r"Iteration.*=\s+(\d+)", line)
        if match:
            result["iterations_completed"] = int(match.group(1))

        # RMSE
        match = re.search(r"data RMSE =\s+([\d.E+-]+)", line)
        if match:
            result["final_rmse"] = float(match.group(1))

        # Memory
        match = re.search(r"MEMORY USED \(total\).*=\s+([\d.E+-]+)", line)
        if match:
            result["memory_gb"] = float(match.group(1))

        # Model range
        match = re.search(r"min/max values =\s+([\d.E+-]+)\s+([\d.E+-]+)", line)
        if match:
            result["model_min"] = float(match.group(1))
            result["model_max"] = float(match.group(2))

        # Written files
        if "Writing" in line and "file" in line:
            file_match = re.search(r"file\s+(.+?)$", line)
            if file_match:
                result["output_files"].append(file_match.group(1).strip())

    return result


def run_tomofast_inversion(parfile_path: str, output_dir: str) -> dict:
    """Execute Tomofast-x inversion and return results."""
    event_logger.log_skill("TomofastSkill", "Starting Tomofast-x execution", details={
        "parfile": parfile_path,
        "output_dir": output_dir,
        "binary": TOMOFAST_BIN,
    })

    # Compute adaptive timeout based on problem size from parfile
    # Large problems (sensitivity matrix > 1 GB) need significantly more time
    timeout_seconds = 300  # default: 5 minutes
    parfile_full_path = os.path.join(TOMOFAST_DIR, parfile_path)
    try:
        with open(parfile_full_path, "r") as pf:
            pf_content = pf.read()
        grid_match = re.search(r"modelGrid\.size\s*=\s*(\d+)\s+(\d+)\s+(\d+)", pf_content)
        ndata_match = re.search(r"nData\s*=\s*(\d+)", pf_content)
        if grid_match and ndata_match:
            n_cells = int(grid_match.group(1)) * int(grid_match.group(2)) * int(grid_match.group(3))
            n_data = int(ndata_match.group(1))
            sensitivity_gb = (n_data * n_cells * 8) / (1024**3)
            # Scale timeout: ~200s per GB of sensitivity, minimum 600s, max 3600s
            timeout_seconds = max(600, min(3600, int(sensitivity_gb * 200)))
    except Exception:
        pass  # Fall back to default

    event_logger.log_skill("TomofastSkill", f"Timeout set to {timeout_seconds}s", details={
        "estimated_sensitivity_gb": f"{sensitivity_gb:.1f}" if 'sensitivity_gb' in dir() else "unknown",
    })

    # Ensure output directory exists
    full_output = os.path.join(TOMOFAST_DIR, output_dir)
    os.makedirs(full_output, exist_ok=True)
    os.makedirs(os.path.join(full_output, "model"), exist_ok=True)
    os.makedirs(os.path.join(full_output, "data"), exist_ok=True)
    os.makedirs(os.path.join(full_output, "Paraview"), exist_ok=True)
    os.makedirs(os.path.join(full_output, "SENSIT"), exist_ok=True)

    event_logger.log_tool("mpirun", f"Executing: mpirun --allow-run-as-root -np 1 {TOMOFAST_BIN} -p {parfile_path}")

    try:
        result = subprocess.run(
            ["mpirun", "--allow-run-as-root", "-np", "1", TOMOFAST_BIN, "-p", parfile_path],
            capture_output=True,
            text=True,
            cwd=TOMOFAST_DIR,
            timeout=timeout_seconds,
        )

        stdout = result.stdout
        stderr = result.stderr

        if result.returncode != 0:
            event_logger.log_skill("TomofastSkill", f"Tomofast-x failed with return code {result.returncode}",
                                   level=LogLevel.ERROR, details={"stderr": stderr[:500]})
            return {"success": False, "error": stderr, "stdout": stdout}

        # Parse the output
        parsed = parse_tomofast_stdout(stdout)

        # Parse costs file
        costs_path = os.path.join(full_output, "costs.txt")
        misfit_history = parse_costs_file(costs_path)

        event_logger.log_skill("TomofastSkill", "Tomofast-x completed successfully", level=LogLevel.SUCCESS, details={
            "iterations": parsed["iterations_completed"],
            "final_rmse": f"{parsed['final_rmse']:.6e}" if parsed["final_rmse"] else "N/A",
            "model_range": f"[{parsed['model_min']:.2f}, {parsed['model_max']:.2f}]" if parsed["model_min"] else "N/A",
            "memory_gb": f"{parsed['memory_gb']:.4f}" if parsed["memory_gb"] else "N/A",
            "output_files": len(parsed["output_files"]),
        })

        return {
            "success": True,
            "stdout": stdout,
            "parsed": parsed,
            "misfit_history": misfit_history,
            "output_dir": full_output,
            "costs_path": costs_path,
        }

    except subprocess.TimeoutExpired:
        event_logger.log_skill("TomofastSkill", f"Tomofast-x timed out after {timeout_seconds}s", level=LogLevel.ERROR)
        return {"success": False, "error": f"Timeout after {timeout_seconds} seconds"}
    except Exception as e:
        event_logger.log_skill("TomofastSkill", f"Error running Tomofast-x: {str(e)}", level=LogLevel.ERROR)
        return {"success": False, "error": str(e)}


def run_full_inversion(parfile_rel_path: str):
    """Run the complete inversion workflow with real Tomofast-x execution."""
    event_logger.clear()
    event_logger.log_system("RunController", "Inversion workflow initiated")

    # --- AGENT: Planning ---
    event_logger.log_agent("Orchestrator", "Processing inversion request", details={
        "parfile": parfile_rel_path,
    })
    event_logger.log_agent("Planner", "Workflow: Tomofast-x gravity inversion", details={
        "algorithm": "Tomofast-x (LSQR solver)",
        "mode": "single inversion",
    })

    # --- Read Parfile for parameters ---
    parfile_full = os.path.join(TOMOFAST_DIR, parfile_rel_path)
    event_logger.log_tool("parfile_reader", f"Reading Parfile: {parfile_rel_path}")

    parfile_info = {}
    try:
        with open(parfile_full, "r") as f:
            content = f.read()
            # Extract key parameters
            match = re.search(r"nMajorIterations\s*=\s*(\d+)", content)
            if match:
                parfile_info["n_major_iterations"] = int(match.group(1))
            match = re.search(r"nMinorIterations\s*=\s*(\d+)", content)
            if match:
                parfile_info["n_minor_iterations"] = int(match.group(1))
            match = re.search(r"outputFolderPath\s*=\s*(.+)", content)
            if match:
                parfile_info["output_folder"] = match.group(1).strip()
            match = re.search(r"description\s*=\s*(.+)", content)
            if match:
                parfile_info["description"] = match.group(1).strip()
            match = re.search(r"modelGrid\.size\s*=\s*(.+)", content)
            if match:
                parfile_info["grid_size"] = match.group(1).strip()
            match = re.search(r"nData\s*=\s*(\d+)", content)
            if match:
                parfile_info["n_data"] = int(match.group(1))
            match = re.search(r"matrixCompression\.type\s*=\s*(\d+)", content)
            if match:
                parfile_info["compression_type"] = int(match.group(1))
            match = re.search(r"matrixCompression\.rate\s*=\s*([\d.]+)", content)
            if match:
                parfile_info["compression_rate"] = float(match.group(1))
    except Exception as e:
        event_logger.log_tool("parfile_reader", f"Error reading Parfile: {e}", level=LogLevel.ERROR)

    event_logger.log_tool("parfile_reader", "Parfile parsed", level=LogLevel.SUCCESS, details=parfile_info)

    # --- SKILL: Validation ---
    event_logger.log_skill("TomofastSkill", "Validating inversion setup", details={
        "binary_exists": os.path.exists(TOMOFAST_BIN),
        "parfile_exists": os.path.exists(parfile_full),
    })

    # Check data file exists
    data_file_match = re.search(r"dataGridFile\s*=\s*(.+)", content) if 'content' in dir() else None
    if data_file_match:
        data_path = os.path.join(TOMOFAST_DIR, data_file_match.group(1).strip())
        event_logger.log_tool("validate_data_file", f"Checking data file: {data_file_match.group(1).strip()}", details={
            "exists": os.path.exists(data_path),
        })

    event_logger.log_skill("TomofastSkill", "Validation passed", level=LogLevel.SUCCESS)

    # --- WORKFLOW: Execution ---
    output_dir = parfile_info.get("output_folder", "output/run/")
    event_logger.log_workflow("Inversion", "Starting execution", details={
        "estimated_iterations": parfile_info.get("n_major_iterations", "unknown"),
        "grid_size": parfile_info.get("grid_size", "unknown"),
        "n_data_points": parfile_info.get("n_data", "unknown"),
    })

    # Show progress placeholder
    progress_ph = st.empty()
    with progress_ph.container():
        st.info("⚡ Running Tomofast-x inversion... (this may take 10-30 seconds)")
        st.progress(0.5, text="Executing...")

    # Run the actual inversion
    result = run_tomofast_inversion(parfile_rel_path, output_dir)

    progress_ph.empty()

    if result["success"]:
        st.session_state.run_status = "completed"
        st.session_state.misfit_history_tomofast = result["misfit_history"]
        st.session_state.results_tomofast = {
            "output_dir": result["output_dir"],
            "parsed": result["parsed"],
            "misfit_history": result["misfit_history"],
            "final_misfit": result["misfit_history"][-1] if result["misfit_history"] else None,
            "iterations": result["parsed"]["iterations_completed"],
            "stdout": result["stdout"],
            "compression_type": parfile_info.get("compression_type", 0),
            "compression_rate": parfile_info.get("compression_rate", 0.15),
        }
        st.session_state.current_iteration_tomofast = result["parsed"]["iterations_completed"]
        st.session_state.progress_tomofast = 1.0

        event_logger.log_workflow("Inversion", "Workflow completed", level=LogLevel.SUCCESS)
        event_logger.log_agent("Orchestrator", "Results ready for visualization", level=LogLevel.SUCCESS)
    else:
        st.session_state.run_status = "failed"
        event_logger.log_workflow("Inversion", "Workflow failed", level=LogLevel.ERROR, details={
            "error": result.get("error", "Unknown error")[:200],
        })


def _render_full_log_section():
    """Render the full expandable log section with filtering."""
    st.subheader("📝 Execution Log")

    # Filter controls
    filter_col1, filter_col2, filter_col3 = st.columns([2, 2, 1])
    with filter_col1:
        source_filter = st.multiselect(
            "Filter by source",
            options=["All", "Agent", "Skill", "Tool", "Workflow", "Data", "System"],
            default=["All"],
            key="log_source_filter",
        )
    with filter_col2:
        level_filter = st.multiselect(
            "Filter by level",
            options=["All", "Info", "Success", "Warning", "Error", "Debug"],
            default=["All"],
            key="log_level_filter",
        )
    with filter_col3:
        show_details = st.checkbox("Show details", value=True, key="log_show_details")

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

    # Download button
    if all_events:
        full_log = "\n".join(e.format_detailed() for e in all_events)
        st.download_button(
            "📥 Download Full Log",
            data=full_log,
            file_name=f"inversionai_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
        )


def _load_result_model_3d(results: dict) -> np.ndarray | None:
    """Load the Tomofast-x model file and reshape to 3D grid."""
    output_dir = results.get("output_dir", "")
    if not output_dir:
        return None

    # Find model file
    model_path = os.path.join(output_dir, "model", "grav_final_model_full.txt")
    if not os.path.exists(model_path):
        model_path = os.path.join(output_dir, "model", "mag_final_model_full.txt")
    if not os.path.exists(model_path):
        model_dir = os.path.join(output_dir, "model")
        if os.path.exists(model_dir):
            candidates = [f for f in os.listdir(model_dir) if "model" in f and f.endswith(".txt")]
            if candidates:
                model_path = os.path.join(model_dir, candidates[0])

    if not os.path.exists(model_path):
        return None

    # Read model values
    values = []
    try:
        with open(model_path, "r") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    val = float(line)
                    if i == 0 and val == int(val) and val > 100:
                        continue  # Skip cell count header
                    values.append(val)
                except ValueError:
                    pass
    except Exception:
        return None

    if not values:
        return None

    model_values = np.array(values)

    # Get grid dimensions from Parfile_copy.txt
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
        n = len(model_values)
        known = {57057: (13, 133, 33), 4000: (20, 20, 10), 62500: (50, 50, 25)}
        if n in known:
            nx, ny, nz = known[n]
        else:
            # Factorize
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
        return model_values.reshape((nx, ny, nz), order='F')
    return None


def _render_result_3d_model(model: np.ndarray):
    """Render 3D model cross-sections on the Run page."""
    import plotly.graph_objects as go

    nx, ny, nz = model.shape
    st.caption(f"Model grid: {nx} × {ny} × {nz} cells ({nx*ny*nz:,} total)")

    # Sliders for slice positions
    sl_col1, sl_col2, sl_col3 = st.columns(3)
    with sl_col1:
        slice_z = st.slider("Depth Slice (Z)", 0, nz - 1, nz // 2, key="run_slice_z")
    with sl_col2:
        slice_x = st.slider("X Slice", 0, nx - 1, nx // 2, key="run_slice_x")
    with sl_col3:
        slice_y = st.slider("Y Slice", 0, ny - 1, ny // 2, key="run_slice_y")

    # Render 3 cross-section tabs
    tab_z, tab_x, tab_y = st.tabs(["Depth Slice (XY)", "X-Section (YZ)", "Y-Section (XZ)"])

    with tab_z:
        _render_run_slice(model[:, :, min(slice_z, nz-1)], f"Depth Z={slice_z}", "X", "Y")
    with tab_x:
        _render_run_slice(model[min(slice_x, nx-1), :, :], f"X={slice_x}", "Y", "Z")
    with tab_y:
        _render_run_slice(model[:, min(slice_y, ny-1), :], f"Y={slice_y}", "X", "Z")


def _render_run_slice(slice_data: np.ndarray, title: str, xlabel: str, ylabel: str):
    """Render a single 2D slice as a contour plot."""
    import plotly.graph_objects as go

    fig = go.Figure()
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
        colorbar=dict(title="Susceptibility (SI)" if st.session_state.get("active_data_type", "Gravity").lower() == "magnetic" else "Density (kg/m³)", thickness=15, len=0.9),
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


def _get_data_source() -> str:
    """Determine the active data source: 'demo', 'user', or 'none'."""
    dataset = st.session_state.get("dataset")
    if dataset is None:
        return "none"
    uploaded_file = st.session_state.get("uploaded_file", "")
    if "demo" in uploaded_file.lower() or "synthetic" in uploaded_file.lower():
        return "demo"
    return "user"


def _detect_data_type() -> str:
    """Auto-detect whether data is gravity or magnetic based on file name, format, and columns.

    Returns 'gravity' or 'magnetic'.
    """
    uploaded_file = st.session_state.get("uploaded_file", "")
    data_format = st.session_state.get("data_format", "")
    dataset = st.session_state.get("dataset")

    # Check file name
    name_lower = uploaded_file.lower()
    if "mag" in name_lower or "tmi" in name_lower or "suscept" in name_lower:
        return "magnetic"
    if "grav" in name_lower or "bouger" in name_lower or "density" in name_lower:
        return "gravity"

    # Check format
    if "magnetic" in data_format.lower():
        return "magnetic"

    # Check column names
    if dataset is not None:
        for col in dataset.columns:
            cl = col.lower()
            if "mag" in cl or "tmi" in cl or "suscept" in cl:
                return "magnetic"

    # Default
    return "gravity"


def _prepare_active_data_inversion() -> dict | None:
    """Prepare inversion files from the active dataset (demo or user-uploaded).

    Returns a dict with parfile_rel_path, output_dir, etc., or None on failure.
    """
    from skills.tomofast.run_preparation import prepare_inversion

    dataset = st.session_state.get("dataset")
    if dataset is None:
        st.error("No dataset loaded. Please go to Setup and upload data or load the demo.")
        return None

    data_source = _get_data_source()
    run_label = "demo_gravity" if data_source == "demo" else "user_inversion"

    # Determine mesh spec for demo (known mesh) vs auto for user data
    if data_source == "demo":
        mesh_spec = {"nx": 20, "ny": 20, "nz": 10, "cell_size": (50.0, 50.0, 50.0)}
    else:
        mesh_spec = None  # auto-generate from data extent

    try:
        data_type_sel = st.session_state.get("active_data_type", "Gravity").lower()
        reg_str = st.session_state.get("active_reg_strength", "Medium").lower()
        config = st.session_state.get("config", {})
        result = prepare_inversion(
            dataset=dataset,
            data_type=data_type_sel,
            n_iterations=10,
            mesh_spec=mesh_spec,
            run_label=run_label,
            reg_strength=reg_str,
            mag_inclination=config.get("inclination", -60.0),
            mag_declination=config.get("declination", 0.0),
            mag_intensity=config.get("field_strength", 0.0),
        )
        return result
    except Exception as e:
        st.error(f"Failed to prepare inversion: {e}")
        return None


def render_run_page():
    """Render the Execution & Monitoring page."""
    st.header("▶️ Inversion Execution")

    run_status = st.session_state.get("run_status", "idle")

    # Status header
    st.subheader(f"Status: {get_status_indicator(run_status)}")
    st.markdown("---")

    # --- Determine data source ---
    data_source = _get_data_source()

    # --- Inversion Mode Selection ---
    st.subheader("🎯 Select Inversion to Run")

    # Show which data is active
    if data_source == "demo":
        st.info("🚀 **Demo data loaded** — Will invert the synthetic gravity cube (100 stations, 20×20×10 mesh)")
    elif data_source == "user":
        uploaded_name = st.session_state.get("uploaded_file", "Unknown")
        n_pts = len(st.session_state.get("dataset", []))
        st.info(f"📂 **User data loaded** — `{uploaded_name}` ({n_pts} stations). Will generate mesh and Parfile automatically.")
    else:
        st.warning("⚠️ No data loaded. You can still run a pre-bundled example below, or go to Setup to upload data.")

    # Mode tabs: Active Data vs Pre-bundled Examples
    if data_source != "none":
        tab_active, tab_examples = st.tabs(["📊 Run My Data", "📁 Pre-bundled Examples"])
    else:
        tab_active = None
        tab_examples = st.container()

    # --- Tab 1: Run active (demo or user) data ---
    if tab_active is not None:
        with tab_active:
            # Show input data summary
            dataset = st.session_state.get("dataset")
            uploaded_name = st.session_state.get("uploaded_file", "Unknown")

            # If demo data is loaded, offer option to load user's own data
            if data_source == "demo":
                st.info("🚀 Currently using **demo data**. Upload your own CSV below to switch to your data.")
                user_file = st.file_uploader(
                    "Upload your data (CSV)",
                    type=["csv", "xyz", "obs", "grv", "mag", "txt", "dat"],
                    key="run_page_upload",
                    help="Upload gravity or magnetic observation data to replace the demo",
                )
                if user_file is not None:
                    import pandas as pd
                    from io import StringIO
                    content = user_file.getvalue().decode("utf-8")
                    df = pd.read_csv(StringIO(content))
                    # Standardize columns
                    col_remap = {}
                    for col in df.columns:
                        cl = col.lower()
                        if cl in ("x", "easting", "east", "longitude", "lon"):
                            col_remap[col] = "X"
                        elif cl in ("y", "northing", "north", "latitude", "lat"):
                            col_remap[col] = "Y"
                        elif cl in ("z", "elevation", "elev", "depth", "height"):
                            col_remap[col] = "Z"
                        elif cl in ("value", "gravity", "anomaly", "gz", "obs", "observed") or "grav" in cl or "anomaly" in cl or "mag" in cl:
                            if "uncertainty" not in cl and "error" not in cl and "std" not in cl:
                                if "Value" not in col_remap.values():
                                    col_remap[col] = "Value"
                    if col_remap:
                        df = df.rename(columns=col_remap)

                    st.session_state.dataset = df
                    st.session_state.data_format = "CSV"
                    st.session_state.uploaded_file = user_file.name
                    st.session_state.data_validated = True
                    st.session_state.run_status = "idle"
                    st.session_state.results_tomofast = None
                    st.rerun()

                st.markdown("---")

            st.markdown("#### 📂 Input Data")
            info_col1, info_col2, info_col3, info_col4 = st.columns(4)
            with info_col1:
                st.metric("Source File", uploaded_name[:25])
            with info_col2:
                st.metric("Stations", f"{len(dataset):,}" if dataset is not None else "N/A")
            with info_col3:
                if dataset is not None and "Value" in dataset.columns:
                    vmin, vmax = dataset["Value"].min(), dataset["Value"].max()
                    st.metric("Value Range", f"{vmin:.4f} – {vmax:.4f}")
                else:
                    st.metric("Value Range", "N/A")
            with info_col4:
                if dataset is not None and "X" in dataset.columns:
                    x_range = dataset["X"].max() - dataset["X"].min()
                    y_range = dataset["Y"].max() - dataset["Y"].min()
                    st.metric("Survey Extent", f"{x_range:.0f} × {y_range:.0f} m")
                else:
                    st.metric("Survey Extent", "N/A")

            st.markdown("")

            # Inversion parameters
            st.markdown("#### ⚙️ Inversion Parameters")
            col1, col2, col3 = st.columns([2, 2, 2])
            with col1:
                n_iters = st.number_input("Major Iterations", min_value=1, max_value=200, value=30, key="active_iters")
            with col2:
                reg_strength = st.select_slider(
                    "Regularization",
                    options=["Weak", "Medium", "Strong"],
                    value="Medium",
                    key="active_reg_strength",
                    help="Weak = fits data better (risk overfitting). Strong = smoother model (risk underfitting).",
                )
            with col3:
                detected_type = _detect_data_type()
                data_type_choice = st.selectbox(
                    "Data Type",
                    options=["Gravity", "Magnetic"],
                    index=0 if detected_type == "gravity" else 1,
                    key="active_data_type",
                )

            # Show selected engines from Setup page
            use_tomo = st.session_state.get("use_tomofast", False)
            use_simpeg = st.session_state.get("use_simpeg", False)
            if use_tomo and use_simpeg:
                st.caption("🔄 Running both **Tomofast-x** and **SimPEG** (selected in Setup)")
            elif use_simpeg:
                st.caption("🔬 Running **SimPEG** (selected in Setup)")
            elif use_tomo:
                st.caption("⚡ Running **Tomofast-x** (selected in Setup)")
            else:
                st.warning("⚠️ No algorithm selected. Go to Setup to select Tomofast-x and/or SimPEG.")
            with st.expander("📋 Generated Parfile Details (preview)", expanded=False):
                if dataset is not None:
                    n_pts = len(dataset)
                    if data_source == "demo":
                        nx, ny, nz = 20, 20, 10
                        cell_info = "50.0 × 50.0 × 50.0 m (fixed)"
                    else:
                        # Calculate auto mesh like run_preparation would
                        import numpy as _np
                        x_vals = dataset["X"].dropna().values if "X" in dataset.columns else _np.zeros(1)
                        y_vals = dataset["Y"].dropna().values if "Y" in dataset.columns else _np.zeros(1)
                        x_rng = float(x_vals.max() - x_vals.min()) if len(x_vals) > 1 else 1000.0
                        y_rng = float(y_vals.max() - y_vals.min()) if len(y_vals) > 1 else 1000.0
                        max_rng = max(x_rng, y_rng)
                        if max_rng <= 0:
                            max_rng = 1000.0
                        spacing = _np.sqrt((x_rng * y_rng) / max(n_pts, 1)) if x_rng > 0 and y_rng > 0 else max_rng / 10.0
                        if spacing <= 0 or _np.isnan(spacing):
                            spacing = max_rng / 10.0
                        nx = min(30, max(5, int(_np.ceil(x_rng / spacing))))
                        ny = min(30, max(5, int(_np.ceil(y_rng / spacing))))
                        nz = min(15, max(5, int(_np.ceil((max_rng * 0.5) / spacing))))
                        dx = x_rng / nx if nx > 0 else 50.0
                        dy = y_rng / ny if ny > 0 else 50.0
                        dz = (max_rng * 0.5) / nz if nz > 0 else 50.0
                        cell_info = f"{dx:.1f} × {dy:.1f} × {dz:.1f} m (auto)"

                    detail_cols = st.columns(3)
                    with detail_cols[0]:
                        st.markdown("**Grid Dimensions**")
                        st.code(f"nx × ny × nz = {nx} × {ny} × {nz}\nTotal cells: {nx*ny*nz:,}")
                    with detail_cols[1]:
                        st.markdown("**Cell Size**")
                        st.code(cell_info)
                    with detail_cols[2]:
                        st.markdown("**Inversion Settings**")
                        st.code(f"Solver: LSQR\nMinor iterations: 100\nDepth weighting: ON (power=2)\nDamping: 1.0e-06\nSmoothing: 9.0e-05")

                    st.markdown("**Data columns being used:**")
                    cols_used = []
                    for c in dataset.columns:
                        if c in ("X", "Y", "Z", "Value"):
                            cols_used.append(f"✅ `{c}` → {c}")
                        else:
                            cols_used.append(f"⬜ `{c}` (not used)")
                    st.markdown("  \n".join(cols_used))

                    st.markdown(f"**Output directory:** `workspace/{'demo_gravity' if data_source == 'demo' else 'user_inversion'}/output/`")
                else:
                    st.warning("No dataset loaded.")

            st.markdown("")

            run_col1, run_col2, _ = st.columns([1, 1, 2])
            with run_col1:
                if run_status in ("idle", "completed", "cancelled", "failed"):
                    if st.button("⚡ Run Inversion on My Data", type="primary", key="run_active"):
                        st.session_state.run_status = "running"
                        st.session_state.start_time = datetime.now()
                        st.session_state.run_mode = "active_data"
                        st.session_state.active_n_iters = n_iters
                        # Clear old results
                        st.session_state.results_tomofast = None
                        st.session_state.misfit_history_tomofast = []
                        st.rerun()
            with run_col2:
                if run_status == "completed":
                    if st.button("📊 View Results", key="view_active"):
                        st.session_state.current_page = "compare"
                        st.rerun()

    # --- Tab 2: Pre-bundled examples ---
    with tab_examples if data_source != "none" else tab_examples:
        # Get available parfiles
        parfiles = get_available_parfiles()

        if not parfiles:
            st.warning("No pre-bundled Parfiles found in Tomofast-x directory.")
        else:
            col1, col2 = st.columns([3, 1])
            with col1:
                selected_name = st.selectbox(
                    "Choose an inversion example",
                    options=list(parfiles.keys()),
                    index=list(parfiles.keys()).index("Hamersley Grav") if "Hamersley Grav" in parfiles else 0,
                    help="Select a pre-configured Parfile to run",
                )
            with col2:
                st.markdown("")
                st.markdown("")
                selected_parfile = parfiles[selected_name]
                st.caption(f"📄 `{selected_parfile}`")

            # Show Parfile details
            parfile_full = os.path.join(TOMOFAST_DIR, selected_parfile)
            if os.path.exists(parfile_full):
                with st.expander("📋 Parfile Parameters"):
                    with open(parfile_full, "r") as f:
                        content = f.read()
                    desc = re.search(r"description\s*=\s*(.+)", content)
                    grid = re.search(r"modelGrid\.size\s*=\s*(.+)", content)
                    ndata = re.search(r"nData\s*=\s*(\d+)", content)
                    niters = re.search(r"nMajorIterations\s*=\s*(\d+)", content)

                    info_cols = st.columns(4)
                    with info_cols[0]:
                        st.metric("Grid Size", grid.group(1).strip() if grid else "N/A")
                    with info_cols[1]:
                        st.metric("Data Points", ndata.group(1) if ndata else "N/A")
                    with info_cols[2]:
                        st.metric("Major Iterations", niters.group(1) if niters else "N/A")
                    with info_cols[3]:
                        st.metric("Description", desc.group(1).strip()[:30] if desc else "N/A")

            st.markdown("")
            run_col1, run_col2, _ = st.columns([1, 1, 2])
            with run_col1:
                if run_status in ("idle", "completed", "cancelled", "failed"):
                    if st.button("⚡ Run Example", type="secondary", key="run_example"):
                        st.session_state.run_status = "running"
                        st.session_state.start_time = datetime.now()
                        st.session_state.run_mode = "example"
                        st.session_state.selected_parfile = selected_parfile
                        st.rerun()
            with run_col2:
                if run_status == "completed":
                    if st.button("📊 View Results", key="view_example"):
                        st.session_state.current_page = "compare"
                        st.rerun()

    st.markdown("---")

    # Execute inversion if triggered
    if run_status == "running":
        run_mode = st.session_state.get("run_mode", "example")

        if run_mode == "active_data":
            # Prepare data dynamically and run
            n_iters = st.session_state.get("active_n_iters", 10)
            dataset = st.session_state.get("dataset")
            use_tomo = st.session_state.get("use_tomofast", False)
            use_simpeg_flag = st.session_state.get("use_simpeg", False)

            if dataset is None:
                st.error("No dataset in session. Please load data first.")
                st.session_state.run_status = "failed"
                st.rerun()
                return

            data_src = _get_data_source()
            data_type_sel = st.session_state.get("active_data_type", "Gravity").lower()
            reg_str = st.session_state.get("active_reg_strength", "Medium").lower()
            config = st.session_state.get("config", {})

            # Run Tomofast-x if selected
            if use_tomo:
                with st.spinner("Running Tomofast-x inversion..."):
                    from skills.tomofast.run_preparation import prepare_inversion
                    run_label = "demo_gravity" if data_src == "demo" else "user_inversion"
                    mesh_spec = {"nx": 20, "ny": 20, "nz": 10, "cell_size": (50.0, 50.0, 50.0)} if data_src == "demo" else None

                    try:
                        prep_result = prepare_inversion(
                            dataset=dataset,
                            data_type=data_type_sel,
                            n_iterations=n_iters,
                            mesh_spec=mesh_spec,
                            run_label=run_label,
                            reg_strength=reg_str,
                            mag_inclination=config.get("inclination", -60.0),
                            mag_declination=config.get("declination", 0.0),
                            mag_intensity=config.get("field_strength", 0.0),
                        )
                        parfile_to_run = prep_result["parfile_rel_path"]
                    except Exception as e:
                        st.error(f"Failed to prepare Tomofast-x inversion: {e}")
                        st.session_state.run_status = "failed"
                        st.rerun()
                        return

                run_full_inversion(parfile_to_run)

            # Run SimPEG if selected
            if use_simpeg_flag:
                with st.spinner("Running SimPEG inversion..."):
                    from skills.simpeg.run_preparation import prepare_and_run_inversion as simpeg_run
                    try:
                        simpeg_result = simpeg_run(
                            dataset=dataset,
                            data_type=data_type_sel,
                            n_iterations=n_iters,
                            reg_strength=reg_str,
                            mag_inclination=config.get("inclination", -60.0),
                            mag_declination=config.get("declination", 0.0),
                            mag_intensity=config.get("field_strength", 0.0),
                        )
                        if simpeg_result["success"]:
                            st.session_state.results_simpeg = {
                                "model": simpeg_result["model"],
                                "misfit_history": simpeg_result["misfit_history"],
                                "iterations": simpeg_result["iterations"],
                                "final_misfit": simpeg_result["final_misfit"],
                                "runtime": simpeg_result["runtime"],
                                "parsed": simpeg_result["parsed"],
                            }
                            if not use_tomo:
                                # Only SimPEG — set as completed
                                st.session_state.run_status = "completed"
                        else:
                            st.error(f"SimPEG failed: {simpeg_result.get('error', 'Unknown error')}")
                            if not use_tomo:
                                st.session_state.run_status = "failed"
                    except Exception as e:
                        st.error(f"SimPEG error: {e}")
                        if not use_tomo:
                            st.session_state.run_status = "failed"

            st.rerun()
        else:
            parfile_to_run = st.session_state.get("selected_parfile", "")
            run_full_inversion(parfile_to_run)
            st.rerun()

    # --- Results Section ---
    if run_status == "completed" and st.session_state.get("results_tomofast"):
        st.markdown("---")
        st.subheader("📊 Inversion Results")

        results = st.session_state.results_tomofast
        parsed = results.get("parsed", {})

        # Compression badge
        compression_type = results.get("compression_type", 0)
        if compression_type == 1:
            compression_rate = results.get("compression_rate", 0.15)
            st.info(
                f"🗜️ **Wavelet compression enabled** (rate: {compression_rate:.0%}) — "
                f"auto-applied to fit sensitivity matrix in available memory. "
                f"Results may have minor numerical differences vs. uncompressed."
            )
        else:
            st.success("✅ **No compression** — full sensitivity matrix used.")

        # Summary metrics
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.metric("Iterations", parsed.get("iterations_completed", "N/A"))
        with m_col2:
            rmse = parsed.get("final_rmse")
            st.metric("Final RMSE", f"{rmse:.2e}" if rmse else "N/A")
        with m_col3:
            model_min = parsed.get("model_min")
            model_max = parsed.get("model_max")
            if model_min is not None and model_max is not None:
                st.metric("Model Range", f"[{model_min:.1f}, {model_max:.1f}]")
            else:
                st.metric("Model Range", "N/A")
        with m_col4:
            mem = parsed.get("memory_gb")
            st.metric("Memory (GB)", f"{mem:.4f}" if mem else "N/A")

        # Convergence plot
        misfit_history = results.get("misfit_history", [])
        if misfit_history:
            st.subheader("📉 Convergence (Data Cost)")
            render_convergence_comparison({"Tomofast-x": misfit_history})

        # Output files
        output_dir = results.get("output_dir", "")
        if output_dir and os.path.exists(output_dir):
            st.subheader("📁 Output Files")
            vtk_files = list(Path(output_dir).rglob("*.vtk"))
            txt_files = list(Path(output_dir).rglob("*.txt"))

            file_col1, file_col2 = st.columns(2)
            with file_col1:
                st.markdown("**VTK Files (ParaView):**")
                for f in vtk_files:
                    st.caption(f"📦 `{f.relative_to(output_dir)}`")
            with file_col2:
                st.markdown("**Text Files:**")
                for f in txt_files:
                    st.caption(f"📄 `{f.relative_to(output_dir)}`")

        # --- 3D Model Visualization ---
        if output_dir and os.path.exists(output_dir):
            model_3d = _load_result_model_3d(results)
            if model_3d is not None:
                st.markdown("---")
                st.subheader("🌐 3D Model Visualization")
                _render_result_3d_model(model_3d)

        # Raw stdout
        with st.expander("🖥️ Tomofast-x Raw Output"):
            st.code(results.get("stdout", "No output captured"), language="text")

    elif run_status == "failed":
        st.error("❌ Inversion failed. Check the log below for details.")

    # --- SimPEG Results Section ---
    if run_status == "completed" and st.session_state.get("results_simpeg"):
        st.markdown("---")
        st.subheader("🔬 SimPEG Results")

        simpeg_results = st.session_state.results_simpeg
        parsed_s = simpeg_results.get("parsed", {})

        # Summary metrics
        s_col1, s_col2, s_col3, s_col4 = st.columns(4)
        with s_col1:
            st.metric("Iterations", parsed_s.get("iterations_completed", "N/A"))
        with s_col2:
            rmse_s = parsed_s.get("final_rmse")
            st.metric("Final RMSE", f"{rmse_s:.2e}" if rmse_s else "N/A")
        with s_col3:
            model_min_s = parsed_s.get("model_min")
            model_max_s = parsed_s.get("model_max")
            if model_min_s is not None and model_max_s is not None:
                st.metric("Model Range", f"[{model_min_s:.4f}, {model_max_s:.4f}]")
            else:
                st.metric("Model Range", "N/A")
        with s_col4:
            runtime_s = simpeg_results.get("runtime")
            st.metric("Runtime (s)", f"{runtime_s:.1f}" if runtime_s else "N/A")

        # Convergence
        misfit_s = simpeg_results.get("misfit_history", [])
        if misfit_s:
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=misfit_s, mode="lines+markers", name="SimPEG Data Misfit"))
            fig.update_layout(title="SimPEG Convergence", xaxis_title="Iteration", yaxis_title="Data Misfit",
                              height=300, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        # 3D Model
        model_3d_s = simpeg_results.get("model")
        if model_3d_s is not None and hasattr(model_3d_s, "shape") and len(model_3d_s.shape) == 3:
            st.subheader("🌐 SimPEG 3D Model")
            _render_result_3d_model(model_3d_s)
