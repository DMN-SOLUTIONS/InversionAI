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
    elif filename_lower.endswith(".gif"):
        return "DAT"
    elif filename_lower.endswith(".dat"):
        return "DAT"

    # Try content-based detection
    lines = content.strip().split("\n")
    first_line = lines[0].strip() if lines else ""

    # Check for comment-style headers (/ or # prefix)
    if first_line.startswith("/") or first_line.startswith("#") or first_line.startswith("!"):
        return "DAT"

    # UBC format often starts with number of observations
    try:
        int(first_line)
        return "UBC-GIF"
    except ValueError:
        pass

    # If first line is not numeric (a title), check if second line starts with /
    if len(lines) > 1 and (lines[1].strip().startswith("/") or lines[1].strip().startswith("#")):
        return "DAT"

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

        elif fmt == "DAT":
            # DAT/GIF format: skip comment/header lines and count headers
            lines = content.strip().split("\n")
            data_lines = []
            typical_ncols = 0

            for line in lines:
                stripped = line.strip()
                if not stripped:
                    continue
                # Skip comment lines
                if stripped.startswith("/") or stripped.startswith("#") or stripped.startswith("!"):
                    continue
                # Skip lines that don't start with a number (title lines)
                tokens = stripped.split()
                first_token = tokens[0] if tokens else ""
                try:
                    float(first_token)
                except ValueError:
                    continue  # Skip non-numeric lines (titles, headers)

                # Determine typical column count from first multi-column data line
                ncols = len(tokens)
                if ncols >= 3 and typical_ncols == 0:
                    typical_ncols = ncols

                # Skip single-value lines (count headers like "762" or "113")
                if ncols == 1:
                    continue

                # Only include lines with consistent column count
                if typical_ncols > 0 and ncols >= typical_ncols - 1:
                    data_lines.append(stripped)

            if data_lines:
                data_content = "\n".join(data_lines)
                df = pd.read_csv(StringIO(data_content), sep=r"\s+", header=None)
                if df.shape[1] >= 4:
                    df.columns = ["X", "Y", "Z", "Value"] + [
                        f"Col{i}" for i in range(4, df.shape[1])
                    ]
                elif df.shape[1] == 3:
                    df.columns = ["X", "Y", "Value"]
                # Ensure numeric
                for col in ["X", "Y", "Z", "Value"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                return df
            return None

        elif fmt in ("XYZ", "Unknown"):
            # Try space-delimited
            df = pd.read_csv(StringIO(content), sep=r"\s+", header=None)
            if df.shape[1] >= 4:
                df.columns = ["X", "Y", "Z", "Value"] + [
                    f"Col{i}" for i in range(4, df.shape[1])
                ]
            elif df.shape[1] == 3:
                df.columns = ["X", "Y", "Value"]
            # Ensure numeric
            for col in ["X", "Y", "Z", "Value"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
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
            # Ensure numeric
            for col in ["X", "Y", "Z", "Value"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            return df

    except Exception as e:
        st.error(f"Error parsing data: {e}")
        return None

    return None


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Try to identify and rename coordinate columns to X, Y, Z, Value."""
    col_map = {}
    lower_cols = {c.lower(): c for c in df.columns}

    # Map common column names (exact match first)
    x_names = ["x", "easting", "east", "longitude", "lon"]
    y_names = ["y", "northing", "north", "latitude", "lat"]
    z_names = ["z", "elevation", "elev", "depth", "height"]
    val_names = ["value", "data", "obs", "observed", "gravity", "magnetic", "anomaly", "gz", "bz"]

    for names, standard in [(x_names, "X"), (y_names, "Y"), (z_names, "Z"), (val_names, "Value")]:
        for name in names:
            if name in lower_cols and standard not in col_map.values():
                col_map[lower_cols[name]] = standard
                break

    # Fallback: substring matching for Value column if not found yet
    if "Value" not in col_map.values():
        value_substrings = ["grav", "mag", "anomaly", "density", "suscept", "obs", "measured"]
        exclude_substrings = ["uncertainty", "error", "std", "sigma"]
        for col in df.columns:
            cl = col.lower()
            if any(sub in cl for sub in value_substrings) and not any(ex in cl for ex in exclude_substrings):
                if col not in col_map:
                    col_map[col] = "Value"
                    break

    if col_map:
        df = df.rename(columns=col_map)

    # Ensure coordinate and value columns are numeric
    for col in ["X", "Y", "Z", "Value"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

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
        try:
            val_min = pd.to_numeric(df["Value"], errors="coerce").min()
            val_max = pd.to_numeric(df["Value"], errors="coerce").max()
            val_range = val_max - val_min
            if val_range > 0:
                results["value_range_ok"] = True
            else:
                results["messages"].append("⚠️ All values are identical")
        except Exception:
            results["messages"].append("⚠️ Could not determine value range")

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


def _load_demo_data(demo_key: str = "gravity_simple"):
    """Load bundled demo data and set up session state for immediate use.

    Args:
        demo_key: Which demo to load. Options: 'gravity_simple', 'magnetic_simple', 'hamersley'
    """
    from pathlib import Path

    base_path = Path(__file__).resolve().parent.parent.parent

    if demo_key == "gravity_simple":
        demo_path = base_path / "data" / "sample_data" / "gravity_simple" / "observations.csv"
        file_label = "demo_gravity_simple.csv"
        data_format = "CSV"
    elif demo_key == "magnetic_simple":
        demo_path = base_path / "data" / "sample_data" / "magnetic_simple" / "observations.csv"
        file_label = "demo_magnetic_simple.csv"
        data_format = "CSV"
    elif demo_key == "hamersley":
        # Hamersley is in Tomofast-x native format — convert to DataFrame
        tomo_data_path = Path("/app/Tomofast-x/data/gravmag/hamersley/grav_observed_data.txt")
        if tomo_data_path.exists():
            # Read native format: first line = N, then x y z value
            lines = tomo_data_path.read_text().strip().split("\n")
            data_rows = []
            for line in lines[1:]:  # skip count header
                parts = line.split()
                if len(parts) >= 4:
                    data_rows.append([float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])])
            df = pd.DataFrame(data_rows, columns=["X", "Y", "Z", "Value"])
            st.session_state.dataset = df
            st.session_state.data_format = "Tomofast-x native"
            st.session_state.uploaded_file = "demo_hamersley_grav.txt"
            st.session_state.data_validated = True
            st.session_state.use_tomofast = True
            st.session_state.use_simpeg = True
            return
        else:
            st.error("Hamersley data not found in container.")
            return
    else:
        st.error(f"Unknown demo key: {demo_key}")
        return

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
            elif cl in ("value", "gravity", "anomaly", "gz", "obs", "observed") or "grav" in cl or "anomaly" in cl or "mag" in cl:
                if "uncertainty" not in cl and "error" not in cl and "std" not in cl:
                    col_remap[col] = "Value"
        if col_remap:
            df = df.rename(columns=col_remap)

        st.session_state.dataset = df
        st.session_state.data_format = data_format
        st.session_state.uploaded_file = file_label
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


def _parse_binary_gif(raw_bytes: bytes, filename: str) -> pd.DataFrame:
    """Parse a binary GIF file into a DataFrame with X, Y, Z, Value columns.

    Supports two binary formats:
    1. Grid format: int32(nx), int32(ny), 32-byte header, nx*ny float64 values
    2. Observation format: int32(1), int32(n_data), float64(easting), float64(northing_start),
       float64(spacing), float64(?), n_data float64 values (1D profile at regular spacing)
    """
    import struct

    if len(raw_bytes) < 16:
        return None

    try:
        i1 = struct.unpack('<i', raw_bytes[0:4])[0]
        i2 = struct.unpack('<i', raw_bytes[4:8])[0]

        # Format 1: Grid (nx > 10, ny > 10, file_size ≈ 40 + nx*ny*8)
        expected_grid_size = 40 + i1 * i2 * 8
        if (10 < i1 < 10000 and 10 < i2 < 10000
                and abs(len(raw_bytes) - expected_grid_size) < 100):
            # Grid format
            nx, ny = i1, i2
            values = np.frombuffer(raw_bytes[40:40 + nx * ny * 8], dtype='<f8')
            if len(values) != nx * ny:
                return None
            dx = 500.0
            dy = 500.0
            x_coords = np.arange(nx) * dx
            y_coords = np.arange(ny) * dy
            xx, yy = np.meshgrid(x_coords, y_coords, indexing='ij')
            return pd.DataFrame({
                "X": xx.flatten(), "Y": yy.flatten(),
                "Z": np.zeros(nx * ny), "Value": values,
            })

        # Format 2: Observation profile (i1=1, i2=n_data, then metadata + values)
        # Structure: int32(1), int32(n_data), f64(easting), f64(northing_start), f64(spacing), f64(?), n_data*f64(values)
        expected_obs_size = 8 + 4 * 8 + i2 * 8  # 8 header + 4 metadata doubles + n_data doubles
        if (i1 == 1 and 10 < i2 < 10000
                and abs(len(raw_bytes) - expected_obs_size) < 100):
            n_data = i2
            # Read metadata
            easting = struct.unpack('<d', raw_bytes[8:16])[0]
            northing_start = struct.unpack('<d', raw_bytes[16:24])[0]
            spacing = struct.unpack('<d', raw_bytes[24:32])[0]
            # Read data values starting at offset 40 (8 + 4*8 = 40)
            data_offset = 40
            values = np.frombuffer(raw_bytes[data_offset:data_offset + n_data * 8], dtype='<f8')
            if len(values) != n_data:
                return None

            # Generate station coordinates (regular spacing along Northing)
            northings = northing_start + np.arange(n_data) * spacing
            eastings = np.full(n_data, easting)

            return pd.DataFrame({
                "X": eastings, "Y": northings,
                "Z": np.zeros(n_data), "Value": values,
            })

        return None

    except Exception:
        return None


def render_setup_page():
    """Render the Setup & Data Upload page."""
    st.header("📤 Setup & Data Upload")
    st.markdown(
        "Upload your geophysical data file and select the inversion algorithms to use."
    )

    # File upload section
    st.subheader("1️⃣ Upload Your Data")

    st.info(
        "💡 Supported formats: **CSV** (with headers), **XYZ** (space-delimited), "
        "**UBC-GIF** (.obs, .grv, .mag, .gif), **DAT** (with comment headers). "
        "Files should contain at minimum X, Y coordinates and observed data values."
    )

    uploaded_file = st.file_uploader(
        "Choose a data file",
        type=["csv", "xyz", "obs", "grv", "mag", "txt", "dat", "gif"],
        help="Upload gravity or magnetic observation data",
    )

    st.markdown("")

    # Demo option — compact dropdown
    with st.expander("🧪 Or load a demo dataset", expanded=False):
        demo_choice = st.selectbox(
            "Select demo dataset",
            options=[
                "— Select —",
                "🌍 Gravity (Synthetic Cube) — 100 stations, 0.5 g/cm³ density contrast",
                "🧲 Magnetic (Synthetic Cube) — 100 stations, 0.05 SI susceptibility",
                "⛏️ Hamersley (Real Field Data) — 113 stations, Hamersley Basin WA",
            ],
            index=0,
            key="demo_select",
        )
        if st.button("Load Demo", key="load_demo_btn"):
            if "Gravity" in demo_choice:
                _load_demo_data("gravity_simple")
                st.rerun()
            elif "Magnetic" in demo_choice:
                _load_demo_data("magnetic_simple")
                st.rerun()
            elif "Hamersley" in demo_choice:
                _load_demo_data("hamersley")
                st.rerun()
            else:
                st.warning("Please select a demo dataset first.")

    if uploaded_file is not None:
        # Read and detect format - try multiple encodings
        raw_bytes = uploaded_file.getvalue()

        # Check if this is a binary file (GIF grid or observation format)
        is_binary = False
        if len(raw_bytes) > 40:
            import struct
            try:
                i1 = struct.unpack('<i', raw_bytes[0:4])[0]
                i2 = struct.unpack('<i', raw_bytes[4:8])[0]
                # Grid format: nx*ny*8 + 40 ≈ file_size
                expected_grid = 40 + i1 * i2 * 8
                # Observation format: 8 + 4*8 + i2*8 ≈ file_size
                expected_obs = 8 + 4 * 8 + i2 * 8
                if ((10 < i1 < 10000 and 10 < i2 < 10000 and abs(len(raw_bytes) - expected_grid) < 100)
                        or (i1 == 1 and 10 < i2 < 10000 and abs(len(raw_bytes) - expected_obs) < 100)):
                    # Verify first byte is non-text (binary files don't start with printable ASCII)
                    first_byte = raw_bytes[0]
                    if not (32 <= first_byte <= 126):
                        is_binary = True
            except (struct.error, ValueError):
                pass

        if is_binary:
            # Parse binary GIF grid file
            df = _parse_binary_gif(raw_bytes, uploaded_file.name)
            if df is not None:
                detected_format = "Binary GIF (grid)"
                st.session_state.dataset = df
                st.session_state.data_format = detected_format
                st.session_state.uploaded_file = uploaded_file.name

                col1, col2 = st.columns([2, 1])
                with col1:
                    st.success(f"📄 **{uploaded_file.name}** uploaded successfully")
                with col2:
                    st.metric("Detected Format", detected_format)

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
                        try:
                            vmin = float(df['Value'].min())
                            vmax = float(df['Value'].max())
                            st.metric("Value Range", f"{vmin:.2f} – {vmax:.2f}")
                        except (ValueError, TypeError):
                            st.metric("Value Range", "Non-numeric")

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
                if validation["messages"]:
                    for msg in validation["messages"]:
                        st.warning(msg)
                all_critical = validation["has_coordinates"] and validation["sufficient_points"]
                st.session_state.data_validated = all_critical
            else:
                # Binary parsing failed — try as text instead
                is_binary = False

        if not is_binary:
            # Text-based file
            content = None
            for encoding in ["utf-8", "latin-1", "ascii", "cp1252"]:
                try:
                    content = raw_bytes.decode(encoding)
                    break
                except (UnicodeDecodeError, ValueError):
                    continue

            if content is None:
                st.error("❌ Could not decode file. Please ensure it is a text-based data file.")
            else:
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
                            try:
                                vmin = float(df['Value'].min())
                                vmax = float(df['Value'].max())
                                st.metric("Value Range", f"{vmin:.2f} – {vmax:.2f}")
                            except (ValueError, TypeError):
                                st.metric("Value Range", "Non-numeric")

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
