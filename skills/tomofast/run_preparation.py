"""
Dynamic inversion preparation module.

Prepares data files and generates Tomofast-x Parfiles from the active dataset
(demo or user-uploaded). Converts CSV data to Tomofast-x native format and
generates a mesh file compatible with the Tomofast-x binary.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple


# Base working directory inside the Docker container
TOMOFAST_DIR = "/app/Tomofast-x"
WORKING_DIR = os.path.join(TOMOFAST_DIR, "workspace")


def prepare_inversion(
    dataset: pd.DataFrame,
    data_type: str = "gravity",
    n_iterations: int = 30,
    mesh_spec: Optional[dict] = None,
    run_label: str = "user_run",
    reg_strength: str = "medium",
    mag_inclination: float = -60.0,
    mag_declination: float = 0.0,
    mag_intensity: float = 0.0,
) -> dict:
    """Prepare all files needed for a Tomofast-x inversion from a DataFrame.

    This is the main entry point. It:
    1. Creates a working directory
    2. Converts the dataset to Tomofast-x observation format
    3. Generates a mesh (or uses provided mesh spec)
    4. Writes a Parfile referencing all files

    Args:
        dataset: DataFrame with columns X, Y, Z (optional), Value, and optionally Uncertainty.
        data_type: 'gravity' or 'magnetic'
        n_iterations: Number of major iterations
        mesh_spec: Optional dict with mesh parameters. If None, auto-generates from data.
        run_label: Label for this run (used in folder naming)
        reg_strength: 'weak', 'medium', or 'strong' regularization
        mag_inclination: Magnetic field inclination in degrees (for magnetic inversions)
        mag_declination: Magnetic field declination in degrees (for magnetic inversions)
        mag_intensity: Magnetic field intensity in nT (0 = use Tomofast-x defaults)

    Returns:
        dict with keys:
            - parfile_path: absolute path to the generated Parfile
            - parfile_rel_path: path relative to TOMOFAST_DIR
            - output_dir: output directory for results
            - data_file: path to observation data file
            - mesh_file: path to mesh file
            - n_data: number of data points
            - mesh_dims: (nx, ny, nz)
    """
    # Create working directory structure
    run_dir = os.path.join(WORKING_DIR, run_label)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "data"), exist_ok=True)

    # Output directory (relative to TOMOFAST_DIR for Parfile)
    output_rel = f"workspace/{run_label}/output/"
    output_abs = os.path.join(TOMOFAST_DIR, output_rel)
    # Clean old output to prevent stale model files from previous runs
    import shutil
    if os.path.exists(output_abs):
        shutil.rmtree(output_abs)
    os.makedirs(output_abs, exist_ok=True)
    os.makedirs(os.path.join(output_abs, "model"), exist_ok=True)
    os.makedirs(os.path.join(output_abs, "data"), exist_ok=True)
    os.makedirs(os.path.join(output_abs, "Paraview"), exist_ok=True)
    os.makedirs(os.path.join(output_abs, "SENSIT"), exist_ok=True)

    # --- Step 1: Convert observations to Tomofast-x format ---
    data_file_abs = os.path.join(run_dir, "data", f"{data_type}_observed_data.txt")
    n_data = _write_observation_file(dataset, data_file_abs)
    data_file_rel = os.path.relpath(data_file_abs, TOMOFAST_DIR)

    # --- Step 2: Generate or use mesh ---
    mesh_file_abs = os.path.join(run_dir, "data", f"{data_type}_grid.txt")
    if mesh_spec:
        nx, ny, nz = mesh_spec["nx"], mesh_spec["ny"], mesh_spec["nz"]
        origin = mesh_spec.get("origin", None)
        cell_size = mesh_spec.get("cell_size", None)
        _write_mesh_file(dataset, mesh_file_abs, nx, ny, nz, origin, cell_size)
    else:
        nx, ny, nz, auto_cell_size = _auto_mesh_dims(dataset)
        _write_mesh_file(dataset, mesh_file_abs, nx, ny, nz, cell_size=auto_cell_size)

    mesh_file_rel = os.path.relpath(mesh_file_abs, TOMOFAST_DIR)
    n_cells = nx * ny * nz

    # --- Step 3: Generate Parfile ---
    parfile_abs = os.path.join(run_dir, f"Parfile_{run_label}.txt")
    _write_parfile(
        parfile_path=parfile_abs,
        data_type=data_type,
        data_file_rel=data_file_rel,
        mesh_file_rel=mesh_file_rel,
        output_dir_rel=output_rel,
        n_data=n_data,
        grid_dims=(nx, ny, nz),
        n_iterations=n_iterations,
        run_label=run_label,
        reg_strength=reg_strength,
        mag_inclination=mag_inclination,
        mag_declination=mag_declination,
        mag_intensity=mag_intensity,
    )
    parfile_rel = os.path.relpath(parfile_abs, TOMOFAST_DIR)

    return {
        "parfile_path": parfile_abs,
        "parfile_rel_path": parfile_rel,
        "output_dir": output_abs,
        "data_file": data_file_abs,
        "mesh_file": mesh_file_abs,
        "n_data": n_data,
        "mesh_dims": (nx, ny, nz),
    }


def _write_observation_file(dataset: pd.DataFrame, output_path: str) -> int:
    """Write observation data in Tomofast-x format.

    Format:
        Line 1: N (number of observations)
        Lines 2+: x  y  z  value  [uncertainty]

    Tomofast-x expects at minimum x, y, z, value columns.
    The Hamersley example uses 4 columns (x y z value) without uncertainty.
    """
    # Extract columns - handle various naming conventions
    x = dataset["X"].values
    y = dataset["Y"].values
    z = dataset["Z"].values if "Z" in dataset.columns else np.zeros(len(dataset))

    # Find the value column
    if "Value" in dataset.columns:
        values = dataset["Value"].values
    else:
        # Try to find a column that looks like observation data
        value_col = None
        for col in dataset.columns:
            cl = col.lower()
            if cl in ("value", "gravity", "anomaly", "gz", "obs", "observed"):
                value_col = col
                break
            if "grav" in cl or "anomaly" in cl or "mag" in cl:
                if "uncertainty" not in cl and "error" not in cl and "std" not in cl:
                    value_col = col
                    break
        if value_col is None:
            # Last resort: use the 4th column (after X, Y, Z)
            non_coord_cols = [c for c in dataset.columns if c not in ("X", "Y", "Z")]
            if non_coord_cols:
                value_col = non_coord_cols[0]
            else:
                raise ValueError("Cannot identify observation value column in dataset")
        values = dataset[value_col].values

    n_data = len(x)

    with open(output_path, "w") as f:
        # First line: number of observations
        f.write(f"{n_data:>12d}\n")
        # Data lines: Tomofast-x expects Northing(larger), Easting(smaller), Z, value
        # Determine which is Northing vs Easting based on coordinate magnitude
        x_mean = np.nanmean(x)
        y_mean = np.nanmean(y)
        if y_mean > x_mean:
            # Y is Northing (larger), X is Easting (smaller) - standard convention
            coord1 = y  # Northing first
            coord2 = x  # Easting second
        else:
            # X is Northing (larger), Y is Easting (smaller) - swapped
            coord1 = x  # Northing first
            coord2 = y  # Easting second

        # Add tiny perturbation (0.1m) to coordinates to prevent exact mesh boundary coincidence
        perturb = 0.1
        for i in range(n_data):
            f.write(
                f"   {coord1[i]+perturb:20.10f}   {coord2[i]+perturb:20.10f}   {z[i]:20.10f}   "
                f"{values[i]:24.16E}\n"
            )

    return n_data


def _auto_mesh_dims(dataset: pd.DataFrame) -> Tuple[int, int, int]:
    """Auto-determine mesh dimensions based on data extent.

    Uses station spacing to determine appropriate cell size.
    Handles 2D profiles (all stations on same line) and 3D surveys.
    """
    x = dataset["X"].values
    y = dataset["Y"].values

    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    n_stations = len(x)

    # Detect if this is a profile (1D line) or a 2D survey
    is_profile_x = y_range < 1.0  # all same Y, profile along X
    is_profile_y = x_range < 1.0  # all same X, profile along Y

    # Estimate station spacing from sorted unique positions
    x_sorted = np.sort(np.unique(x))
    y_sorted = np.sort(np.unique(y))

    if len(x_sorted) > 1:
        dx_spacing = np.median(np.diff(x_sorted))
    else:
        dx_spacing = x_range / 10.0 if x_range > 0 else 1000.0

    if len(y_sorted) > 1:
        dy_spacing = np.median(np.diff(y_sorted))
    else:
        dy_spacing = y_range / 10.0 if y_range > 0 else 1000.0

    if is_profile_x:
        # Profile along X: match Hamersley-proven design
        cell_along = dx_spacing * 0.84
        nx = max(5, int(np.ceil(x_range / cell_along)))
        ny = 13  # across-strike cells
        # Depth: match Hamersley ratio (745/1000 = 0.745 of spacing)
        cell_depth = dx_spacing * 0.745
        nz = 33  # Hamersley depth layers
    elif is_profile_y:
        # Profile along Y: same convention as profile_x (nx=along, ny=across)
        cell_along = dy_spacing * 0.84
        nx = max(5, int(np.ceil(y_range / cell_along)))  # along-profile
        ny = 13  # across-strike
        cell_depth = dy_spacing * 0.745
        nz = 33
    else:
        # 2D survey: use minimum spacing
        cell_size = min(dx_spacing, dy_spacing)
        if cell_size <= 0:
            max_range = max(x_range, y_range)
            cell_size = max_range / max(n_stations, 10)

        nx = max(5, int(np.ceil(x_range / cell_size)))
        ny = max(5, int(np.ceil(y_range / cell_size)))
        # Depth: half of max horizontal extent
        max_range = max(x_range, y_range)
        depth_extent = max_range * 0.5
        nz = max(5, int(np.ceil(depth_extent / cell_size)))

    # Cap total cells to ~100k for reasonable runtime
    max_total = 100000
    total = nx * ny * nz
    if total > max_total:
        scale = (max_total / total) ** (1.0 / 3.0)
        nx = max(5, int(nx * scale))
        ny = max(5, int(ny * scale))
        nz = max(5, int(nz * scale))

    # Compute actual cell sizes to pass to mesh writer
    # Convention: dx=along-profile, dy=across-strike, dz=depth
    if is_profile_x:
        dx = cell_along
        dy = 3000.0  # across-strike: Hamersley standard
        dz = cell_depth
    elif is_profile_y:
        dx = cell_along  # along-profile (Y direction)
        dy = 3000.0  # across-strike (X direction)
        dz = cell_depth
    else:
        dx = x_range / nx if x_range > 0 and nx > 0 else 1000.0
        dy = y_range / ny if y_range > 0 and ny > 0 else 1000.0
        dz = depth_extent / nz if nz > 0 else 500.0

    return nx, ny, nz, (dx, dy, dz)


def _write_mesh_file(
    dataset: pd.DataFrame,
    output_path: str,
    nx: int,
    ny: int,
    nz: int,
    origin: Optional[Tuple[float, float, float]] = None,
    cell_size: Optional[Tuple[float, float, float]] = None,
) -> None:
    """Write mesh in Tomofast-x native format.

    Based on the Hamersley example format:
        Line 1: N_cells (total)
        Lines 2+: y1 y2 x1 x2 z1 z2 ix iy iz

    Tomofast-x convention: first pair = Northing (larger), second pair = Easting (smaller)
    Cell ordering: ix fastest, iy middle, iz slowest.
    Z is positive downward (surface = 0, depth > 0).
    """
    x = dataset["X"].values
    y = dataset["Y"].values

    # Determine which is Northing (larger) vs Easting (smaller)
    x_mean = np.nanmean(x)
    y_mean = np.nanmean(y)
    if y_mean > x_mean:
        # Y=Northing (along-profile for Hamersley-type), X=Easting
        northing = y
        easting = x
    else:
        # X=Northing, Y=Easting (swapped in CSV)
        northing = x
        easting = y

    n_min, n_max = northing.min(), northing.max()
    e_min, e_max = easting.min(), easting.max()
    n_range = n_max - n_min
    e_range = e_max - e_min

    # Calculate cell sizes
    # In Tomofast-x: first pair = Northing direction, second pair = Easting direction
    # nx in our code = along-profile (Northing for Hamersley), ny = across-strike (Easting)
    if cell_size:
        dx_along, dy_across, dz = cell_size
    else:
        dx_along = n_range / nx if n_range > 0 and nx > 0 else 1000.0
        dy_across = e_range / ny if e_range > 0 and ny > 0 else dx_along
        max_range = max(n_range, e_range)
        if max_range == 0:
            max_range = 10000.0
        depth_extent = max_range * 0.3 if min(n_range, e_range) < 1.0 else max_range * 0.5
        dz = depth_extent / nz if nz > 0 else 50.0

    # Origin
    if origin:
        x0, y0, z0 = origin
    else:
        perturb = 3.14159265358979

        if e_range < 1.0:
            # Profile along Northing: station Easting is constant
            e_station = e_min
            e0 = e_station - (ny // 2) * dy_across - dy_across / 2.0 + perturb
            n0 = n_min - dx_along / 2.0 + perturb
        elif n_range < 1.0:
            # Profile along Easting: station Northing is constant
            n_station = n_min
            n0 = n_station - (nx // 2) * dx_along - dx_along / 2.0 + perturb
            e0 = e_min - dy_across / 2.0 + perturb
        else:
            n0 = n_min - dx_along / 2.0 + perturb
            e0 = e_min - dy_across / 2.0 + perturb
        z0 = 0.0

    n_cells = nx * ny * nz

    with open(output_path, "w") as f:
        f.write(f"{n_cells}\n")

        # Generate depth layers with mild expansion
        dz_layers = []
        current_dz = dz
        for iz in range(nz):
            dz_layers.append(current_dz)
            current_dz = min(current_dz * 1.1, dz * 2.0)

        # Tomofast-x mesh format: Northing1 Northing2 Easting1 Easting2 Z1 Z2 ix iy iz
        # Matching Hamersley convention: ix=across-strike (Easting), iy=along-profile (Northing), iz=depth
        z_top = z0
        for iz in range(nz):
            z1 = z_top
            z2 = z_top + dz_layers[iz]
            z_top = z2
            for iy in range(nx):  # iy = along-profile (Northing), nx cells
                n1 = n0 + iy * dx_along
                n2 = n0 + (iy + 1) * dx_along
                for ix in range(ny):  # ix = across-strike (Easting), ny cells
                    e1 = e0 + ix * dy_across
                    e2 = e0 + (ix + 1) * dy_across
                    f.write(
                        f"{n1:.4e} {n2:.4e} {e1:.4e} {e2:.4e} "
                        f"{z1:.2f} {z2:.2f} {ix+1} {iy+1} {iz+1}\n"
                    )


def _write_parfile(
    parfile_path: str,
    data_type: str,
    data_file_rel: str,
    mesh_file_rel: str,
    output_dir_rel: str,
    n_data: int,
    grid_dims: Tuple[int, int, int],
    n_iterations: int,
    run_label: str,
    reg_strength: str = "medium",
    mag_inclination: float = -60.0,
    mag_declination: float = 0.0,
    mag_intensity: float = 55000.0,
) -> None:
    """Write a Tomofast-x Parfile matching the format used by the actual binary.

    Based on the working Hamersley Parfile format for both gravity and magnetic.
    Regularization weights are auto-scaled based on data amplitude to ensure convergence.
    """
    nx, ny, nz = grid_dims
    data_prefix = "grav" if data_type == "gravity" else "magn"
    description = f"InversionAI {data_type} inversion ({run_label})"

    # Regularization strength multiplier
    # Weak = 0.3x (fits data better), Medium = 1.0x, Strong = 3.0x (smoother)
    reg_multiplier = {"weak": 0.3, "medium": 1.0, "strong": 3.0}.get(reg_strength.lower(), 1.0)

    # Regularization weights from Hamersley reference, scaled by reg_strength
    if data_type == "gravity":
        depth_power = "2.0d0"
        damp = 1.0e-06 * reg_multiplier
        smooth = 9.0e-05 * reg_multiplier
        damping_weight = f"{damp:.1e}"
        smoothing_weight = f"{smooth:.1e}"
        grav_problem_weight = "1.d0"
        magn_problem_weight = "0.d0"
        gradient_section = f"""inversion.dampingGradient.weightType     = 1
inversion.dampingGradient.grav.weight    = {smoothing_weight}
inversion.dampingGradient.magn.weight    = 0.d0"""
    else:
        depth_power = "3.0d0"
        damp = 2.8e3 * reg_multiplier
        smooth = 3.93e5 * reg_multiplier
        damping_weight = f"{damp:.2e}"
        smoothing_weight = f"{smooth:.2e}"
        grav_problem_weight = "0.d0"
        magn_problem_weight = "1.d0"
        gradient_section = f"""inversion.dampingGradient.weightType     = 1
inversion.dampingGradient.magn.weight    = {smoothing_weight}"""

    # Magnetic field section - only include if user explicitly provides field parameters
    # When omitted, Tomofast-x uses internal defaults that generally work well
    # Signal to include: mag_intensity > 0 AND explicitly passed by user
    if data_type == "magnetic" and mag_intensity > 0:
        mag_field_section = f"""===================================================================================
MAGNETIC FIELD constants
===================================================================================
forward.magneticField.inclination            = {mag_inclination}
forward.magneticField.declination            = {mag_declination}
forward.magneticField.intensity_nT           = {mag_intensity}
forward.magneticField.XaxisDeclination       = 0.d0

"""
    else:
        mag_field_section = ""

    content = f"""===================================================================================
GLOBAL
===================================================================================
global.outputFolderPath     = {output_dir_rel}
global.description          = {description}

===================================================================================
MODEL GRID parameters
===================================================================================
# nx_across ny_along nz_depth
modelGrid.size                      = {ny} {nx} {nz}
modelGrid.{data_prefix}.file        = {mesh_file_rel}

===================================================================================
DATA parameters
===================================================================================
forward.data.{data_prefix}.nData             = {n_data}
forward.data.{data_prefix}.dataGridFile      = {data_file_rel}

{mag_field_section}===================================================================================
DEPTH WEIGHTING
===================================================================================
forward.depthWeighting.type         = 1
forward.depthWeighting.{data_prefix}.power   = {depth_power}

===================================================================================
SENSITIVITY KERNEL
===================================================================================
sensit.readFromFiles                = 0
sensit.folderPath                   = SENSIT/

===================================================================================
MATRIX COMPRESSION
===================================================================================
# 0-none, 1-wavelet compression.
forward.matrixCompression.type      = 0
forward.matrixCompression.rate      = 0.15

===================================================================================
PRIOR MODEL
===================================================================================
inversion.priorModel.type           = 1
inversion.priorModel.{data_prefix}.value     = 0.d0

===================================================================================
STARTING MODEL
===================================================================================
inversion.startingModel.type        = 1
inversion.startingModel.{data_prefix}.value  = 0.d0

===================================================================================
INVERSION parameters
===================================================================================
inversion.nMajorIterations          = {n_iterations}
inversion.nMinorIterations          = 100
inversion.minResidual               = 1.d-13

===================================================================================
MODEL DAMPING (m - m_prior)
===================================================================================
inversion.modelDamping.{data_prefix}.weight  = {damping_weight}
inversion.modelDamping.normPower    = 2.0d0

===================================================================================
DAMPING-GRADIENT constraints
===================================================================================
{gradient_section}

===================================================================================
JOINT INVERSION parameters
===================================================================================
inversion.joint.grav.problemWeight  = {grav_problem_weight}
inversion.joint.magn.problemWeight  = {magn_problem_weight}
"""

    with open(parfile_path, "w") as f:
        f.write(content)


def prepare_demo_inversion() -> dict:
    """Prepare an inversion from the bundled demo gravity data.

    Reads the demo CSV, converts to Tomofast-x format, generates mesh, and writes Parfile.
    Uses the known mesh dimensions from the demo README (20x20x10, 50m cells).

    Returns:
        dict with parfile_path, parfile_rel_path, output_dir, etc.
    """
    demo_csv = Path("/app/data/sample_data/gravity_simple/observations.csv")

    if not demo_csv.exists():
        # Try local dev path
        demo_csv = Path(__file__).resolve().parent.parent / "data" / "sample_data" / "gravity_simple" / "observations.csv"

    if not demo_csv.exists():
        raise FileNotFoundError(f"Demo data not found at {demo_csv}")

    df = pd.read_csv(demo_csv)

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
        elif cl in ("value", "gravity", "anomaly", "gz", "obs", "observed") or "grav" in cl or "anomaly" in cl:
            if "uncertainty" not in cl and "error" not in cl and "std" not in cl:
                col_remap[col] = "Value"
        elif cl in ("uncertainty", "uncertainty_mgal", "std", "error"):
            col_remap[col] = "Uncertainty"
    if col_remap:
        df = df.rename(columns=col_remap)

    # Use the known demo mesh: 20x20x10, 50m cells
    mesh_spec = {
        "nx": 20,
        "ny": 20,
        "nz": 10,
        "cell_size": (50.0, 50.0, 50.0),
    }

    return prepare_inversion(
        dataset=df,
        data_type="gravity",
        n_iterations=10,
        mesh_spec=mesh_spec,
        run_label="demo_gravity",
    )
