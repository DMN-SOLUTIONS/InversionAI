"""
InversionAI - Streamlit UI Main Entry Point
A visual interface for non-experts to run geophysical inversions.
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


def render_page():
    """Import and render the current page based on navigation state."""
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


def main():
    """Main application entry point."""
    init_session_state()
    render_sidebar()
    render_page()


if __name__ == "__main__":
    main()
