"""
InversionAI - Setup & Data Upload Page
Handles file upload, format detection, data preview, and algorithm selection.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO


def detect_format(filename: str, content: str) -> str:
    """Auto-detect file format from filename extension and content structure."""
    filename_lower = filename.lower()

    if filename_lower.endswith(".csv"):
        return "CSV"
    elif filename_lower.endswith(".xyz"):
        return "XYZ"
    elif filename_lower.endswith(".obs") or filename_lower.endswith(".grv"):
        return "UBC-GIF"
    elif filename_lower.endswith(".mag"):
        return "UBC-GIF (Magnetic)"

    # Try content-based detection
    lines = content.strip().split("\n")
    first_line = lines[0].strip() if lines else ""

    # UBC format often starts with number of observations
    try:
        int(first_line)
        return "UBC-GIF"
    except ValueError:
        pass

    # CSV with header
    if "," in first_line:
        return "CSV"

    # Space/tab delimited XYZ
    if len(first_line.split()) >= 3:
        return "XYZ"

    return "Unknown"


def parse_data(content: str, fmt: str) -> pd.DataFrame:
    """Parse uploaded data into a standardized DataFrame."""
    try:
        if fmt == "CSV":
            df = pd.read_csv(StringIO(content))
            # Try to identify coordinate columns
            df = standardize_columns(df)
            return df

        elif fmt in ("XYZ", "Unknown"):
            # Try space-delimited
            df = pd.read_csv(StringIO(content), sep=r"\s+", header=None)
            if df.shape[1] >= 4:
                df.columns = ["X", "Y", "Z", "Value"] + [
                    f"Col{i}" for i in range(4, df.shape[1])
                ]
            elif df.shape[1] == 3:
                df.columns = ["X", "Y", "Value"]
            return df

        elif fmt.startswith("UBC"):
            lines = content.strip().split("\n")
            # Skip header line (number of observations)
            start_idx = 0
            try:
                int(lines[0].strip())
                start_idx = 1
            except ValueError:
                pass

            data_lines = "\n".join(lines[start_idx:])
            df = pd.read_csv(StringIO(data_lines), sep=r"\s+", header=None)
            if df.shape[1] >= 4:
                df.columns = ["X", "Y", "Z", "Value"] + [
                    f"Col{i}" for i in range(4, df.shape[1])
                ]
            elif df.shape[1] == 3:
                df.columns = ["X", "Y", "Value"]
            return df

    except Exception as e:
        st.error(f"Error parsing data: {e}")
        return None

    return None


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Try to identify and rename coordinate columns to X, Y, Z, Value."""
    col_map = {}
    lower_cols = {c.lower(): c for c in df.columns}

    # Map common column names
    x_names = ["x", "easting", "east", "longitude", "lon"]
    y_names = ["y", "northing", "north", "latitude", "lat"]
    z_names = ["z", "elevation", "elev", "depth", "height"]
    val_names = ["value", "data", "obs", "observed", "gravity", "magnetic", "anomaly", "gz", "bz"]

    for names, standard in [(x_names, "X"), (y_names, "Y"), (z_names, "Z"), (val_names, "Value")]:
        for name in names:
            if name in lower_cols and standard not in col_map.values():
                col_map[lower_cols[name]] = standard
                break

    if col_map:
        df = df.rename(columns=col_map)

    return df


def validate_data(df: pd.DataFrame) -> dict:
    """Validate the uploaded dataset and return status dict."""
    results = {
        "has_coordinates": False,
        "has_values": False,
        "no_nan_coords": False,
        "sufficient_points": False,
        "value_range_ok": False,
        "messages": [],
    }

    if df is None:
        results["messages"].append("❌ No data loaded")
        return results

    # Check coordinates
    if "X" in df.columns and "Y" in df.columns:
        results["has_coordinates"] = True
    else:
        results["messages"].append("❌ Missing X/Y coordinate columns")

    # Check values
    if "Value" in df.columns:
        results["has_values"] = True
    else:
        results["messages"].append("⚠️ No 'Value' column detected — may need manual mapping")

    # Check NaN in coordinates
    if results["has_coordinates"]:
        nan_count = df[["X", "Y"]].isna().sum().sum()
        if nan_count == 0:
            results["no_nan_coords"] = True
        else:
            results["messages"].append(f"⚠️ {nan_count} NaN values in coordinates")

    # Check point count
    if len(df) >= 10:
        results["sufficient_points"] = True
    else:
        results["messages"].append(f"⚠️ Only {len(df)} data points (recommend ≥10)")

    # Check value range
    if results["has_values"]:
        val_range = df["Value"].max() - df["Value"].min()
        if val_range > 0:
            results["value_range_ok"] = True
        else:
            results["messages"].append("⚠️ All values are identical")

    return results


def render_station_map(df: pd.DataFrame):
    """Render a map/scatter plot of station locations."""
    if df is None or "X" not in df.columns or "Y" not in df.columns:
        st.warning("Cannot render map: missing X/Y coordinates")
        return

    has_z = "Z" in df.columns
    has_value = "Value" in df.columns

    if has_z:
        # 3D scatter
        color_col = "Value" if has_value else "Z"
        fig = px.scatter_3d(
            df,
            x="X",
            y="Y",
            z="Z",
            color=color_col,
            color_continuous_scale="Viridis",
            title="Station Locations (3D)",
            labels={"X": "Easting (m)", "Y": "Northing (m)", "Z": "Elevation (m)"},
        )
        fig.update_layout(height=500)
    else:
        # 2D scatter
        color_col = "Value" if has_value else None
        fig = px.scatter(
            df,
            x="X",
            y="Y",
            color=color_col,
            color_continuous_scale="Viridis",
            title="Station Locations",
            labels={"X": "Easting (m)", "Y": "Northing (m)"},
        )
        fig.update_layout(height=450, yaxis_scaleanchor="x")

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=40, r=40, t=50, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


def _load_demo_data():
    """Load bundled demo gravity data and set up session state for immediate use."""
    from pathlib import Path

    demo_path = Path(__file__).resolve().parent.parent.parent / "data" / "sample_data" / "gravity_simple" / "observations.csv"

    if demo_path.exists():
        df = pd.read_csv(demo_path)
        # Standardize columns
        col_remap = {}
        for col in df.columns:
            cl = col.lower()
            if cl in ("x", "easting"):
                col_remap[col] = "X"
            elif cl in ("y", "northing"):
                col_remap[col] = "Y"
            elif cl in ("z", "elevation"):
                col_remap[col] = "Z"
            elif cl in ("value", "gravity", "anomaly", "gz", "obs", "observed"):
                col_remap[col] = "Value"
        if col_remap:
            df = df.rename(columns=col_remap)

        st.session_state.dataset = df
        st.session_state.data_format = "CSV"
        st.session_state.uploaded_file = "demo_gravity_simple.csv"
        st.session_state.data_validated = True
        st.session_state.use_tomofast = True
        st.session_state.use_simpeg = True
    else:
        # Fallback: generate synthetic data
        n_stations = 100
        x = np.repeat(np.linspace(0, 1000, 10), 10)
        y = np.tile(np.linspace(0, 1000, 10), 10)
        z = np.zeros(n_stations)
        values = np.random.uniform(0.02, 0.16, n_stations)
        df = pd.DataFrame({"X": x, "Y": y, "Z": z, "Value": values})

        st.session_state.dataset = df
        st.session_state.data_format = "CSV (synthetic)"
        st.session_state.uploaded_file = "synthetic_gravity_100pts.csv"
        st.session_state.data_validated = True
        st.session_state.use_tomofast = True
        st.session_state.use_simpeg = True


def render_setup_page():
    """Render the Setup & Data Upload page."""
    st.header("📤 Setup & Data Upload")
    st.markdown(
        "Upload your geophysical data file and select the inversion algorithms to use."
    )

    # File upload section
    st.subheader("1️⃣ Upload Data")

    # Quick demo option
    demo_col1, demo_col2 = st.columns([1, 3])
    with demo_col1:
        if st.button("🚀 Load Demo Data", type="primary"):
            _load_demo_data()
            st.rerun()
    with demo_col2:
        st.caption("Load bundled synthetic gravity data (100 stations) for a quick test run.")

    st.markdown("")
    st.info(
        "💡 Supported formats: **CSV** (with headers), **XYZ** (space-delimited), "
        "**UBC-GIF** (.obs, .grv, .mag). Files should contain at minimum X, Y coordinates "
        "and observed data values."
    )

    uploaded_file = st.file_uploader(
        "Choose a data file",
        type=["csv", "xyz", "obs", "grv", "mag", "txt", "dat"],
        help="Upload gravity or magnetic observation data",
    )

    if uploaded_file is not None:
        # Read and detect format
        content = uploaded_file.getvalue().decode("utf-8")
        detected_format = detect_format(uploaded_file.name, content)

        col1, col2 = st.columns([2, 1])
        with col1:
            st.success(f"📄 **{uploaded_file.name}** uploaded successfully")
        with col2:
            st.metric("Detected Format", detected_format)

        # Parse data
        df = parse_data(content, detected_format)

        if df is not None:
            st.session_state.dataset = df
            st.session_state.data_format = detected_format
            st.session_state.uploaded_file = uploaded_file.name

            # Data preview
            st.subheader("2️⃣ Data Preview")
            col1, col2 = st.columns([3, 1])

            with col1:
                st.dataframe(df.head(20), use_container_width=True, height=300)

            with col2:
                st.markdown("**Dataset Summary**")
                st.metric("Rows", f"{len(df):,}")
                st.metric("Columns", f"{df.shape[1]}")
                if "Value" in df.columns:
                    st.metric("Value Range", f"{df['Value'].min():.2f} – {df['Value'].max():.2f}")

            # Map visualization
            st.subheader("3️⃣ Station Map")
            render_station_map(df)

            # Validation
            st.subheader("4️⃣ Data Validation")
            validation = validate_data(df)

            val_cols = st.columns(5)
            checks = [
                ("Coordinates", validation["has_coordinates"]),
                ("Values", validation["has_values"]),
                ("No NaN Coords", validation["no_nan_coords"]),
                ("Sufficient Points", validation["sufficient_points"]),
                ("Value Range", validation["value_range_ok"]),
            ]
            for col, (label, status) in zip(val_cols, checks):
                with col:
                    if status:
                        st.markdown(f"🟢 **{label}**")
                    else:
                        st.markdown(f"🔴 **{label}**")

            # Show messages
            if validation["messages"]:
                for msg in validation["messages"]:
                    st.warning(msg)

            # Mark data as validated if key checks pass
            all_critical = validation["has_coordinates"] and validation["sufficient_points"]
            st.session_state.data_validated = all_critical

        else:
            st.error("Failed to parse the uploaded file. Please check the format.")

    elif st.session_state.get("dataset") is not None:
        # Show previously loaded data
        st.success(f"📄 **{st.session_state.uploaded_file}** already loaded")
        df = st.session_state.dataset
        st.dataframe(df.head(10), use_container_width=True)

    st.markdown("---")

    # Algorithm selection
    st.subheader("5️⃣ Algorithm Selection")
    st.markdown("Choose one or both inversion algorithms to run:")

    algo_col1, algo_col2 = st.columns(2)

    with algo_col1:
        st.markdown("### ⚡ Tomofast-x")
        st.markdown(
            "Fast parallel 3D inversion using forward modeling. "
            "Excellent for large-scale gravity/magnetic inversions."
        )
        use_tomofast = st.checkbox(
            "Use Tomofast-x",
            value=st.session_state.get("use_tomofast", False),
            key="cb_tomofast",
        )
        st.session_state.use_tomofast = use_tomofast

    with algo_col2:
        st.markdown("### 🔬 SimPEG")
        st.markdown(
            "Simulation and Parameter Estimation in Geophysics. "
            "Flexible framework with multiple regularization options."
        )
        use_simpeg = st.checkbox(
            "Use SimPEG",
            value=st.session_state.get("use_simpeg", False),
            key="cb_simpeg",
        )
        st.session_state.use_simpeg = use_simpeg

    if not use_tomofast and not use_simpeg:
        st.warning("⚠️ Please select at least one algorithm to proceed.")

    st.markdown("---")

    # Proceed button
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        can_proceed = (
            st.session_state.get("data_validated", False)
            and (use_tomofast or use_simpeg)
        )
        if st.button(
            "Proceed to Configure ➡️",
            disabled=not can_proceed,
            use_container_width=True,
            type="primary",
        ):
            st.session_state.current_page = "configure"
            st.rerun()

    if not can_proceed:
        if not st.session_state.get("data_validated", False):
            st.caption("Upload and validate data to proceed.")
        elif not (use_tomofast or use_simpeg):
            st.caption("Select at least one algorithm to proceed.")
