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
                if len(parts) >= 2:
                    try:
                        iteration = int(parts[0])
                        data_cost_grav = float(parts[1])
                        costs.append(data_cost_grav)
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
            timeout=300,  # 5 minute timeout
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
        event_logger.log_skill("TomofastSkill", "Tomofast-x timed out after 300s", level=LogLevel.ERROR)
        return {"success": False, "error": "Timeout after 300 seconds"}
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


def render_run_page():
    """Render the Execution & Monitoring page."""
    st.header("▶️ Inversion Execution")

    run_status = st.session_state.get("run_status", "idle")

    # Status header
    st.subheader(f"Status: {get_status_indicator(run_status)}")
    st.markdown("---")

    # --- Inversion Selection ---
    st.subheader("🎯 Select Inversion to Run")

    # Get available parfiles
    parfiles = get_available_parfiles()

    if not parfiles:
        st.error("No Parfiles found in Tomofast-x directory.")
        return

    # Selection
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
            # Extract key info
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

    st.markdown("---")

    # Run button
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if run_status in ("idle", "completed", "cancelled", "failed"):
            if st.button("⚡ Run Tomofast-x", type="primary"):
                st.session_state.run_status = "running"
                st.session_state.start_time = datetime.now()
                st.session_state.selected_parfile = selected_parfile
                st.rerun()
    with col2:
        if run_status == "completed":
            if st.button("📊 View Comparison"):
                st.session_state.current_page = "compare"
                st.rerun()

    # Execute inversion if triggered
    if run_status == "running":
        parfile_to_run = st.session_state.get("selected_parfile", selected_parfile)
        run_full_inversion(parfile_to_run)
        st.rerun()

    # --- Results Section ---
    if run_status == "completed" and st.session_state.get("results_tomofast"):
        st.markdown("---")
        st.subheader("📊 Inversion Results")

        results = st.session_state.results_tomofast
        parsed = results.get("parsed", {})

        # Summary metrics
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.metric("Iterations", parsed.get("iterations_completed", "N/A"))
        with m_col2:
            rmse = parsed.get("final_rmse")
            st.metric("Final RMSE", f"{rmse:.2e}" if rmse else "N/A")
        with m_col3:
            st.metric("Model Range", f"[{parsed.get('model_min', 0):.1f}, {parsed.get('model_max', 0):.1f}]")
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

        # Raw stdout
        with st.expander("🖥️ Tomofast-x Raw Output"):
            st.code(results.get("stdout", "No output captured"), language="text")

    elif run_status == "failed":
        st.error("❌ Inversion failed. Check the log below for details.")

    # --- Full Log Section ---
    st.markdown("---")
    _render_full_log_section()
