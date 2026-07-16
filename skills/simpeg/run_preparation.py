"""
SimPEG inversion preparation and execution module.

Runs gravity or magnetic inversions using SimPEG's Python API.
Same interface as tomofast/run_preparation.py for seamless engine switching.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple


def prepare_and_run_inversion(
    dataset: pd.DataFrame,
    data_type: str = "gravity",
    n_iterations: int = 30,
    reg_strength: str = "medium",
    mag_inclination: float = -60.0,
    mag_declination: float = 0.0,
    mag_intensity: float = 55000.0,
) -> dict:
    """Run a SimPEG inversion directly from a DataFrame.

    Unlike Tomofast-x which needs file preparation + subprocess,
    SimPEG runs in-process as pure Python.

    Args:
        dataset: DataFrame with columns X, Y, Z (optional), Value.
        data_type: 'gravity' or 'magnetic'
        n_iterations: Maximum number of iterations
        reg_strength: 'weak', 'medium', or 'strong'
        mag_inclination: Magnetic field inclination (degrees)
        mag_declination: Magnetic field declination (degrees)
        mag_intensity: Magnetic field intensity (nT), 0 = use default 50000

    Returns:
        dict with keys:
            - success: bool
            - model: 3D numpy array (nx, ny, nz)
            - misfit_history: list of data misfit per iteration
            - mesh_dims: (nx, ny, nz)
            - iterations: number completed
            - final_misfit: last misfit value
            - runtime: seconds
            - error: error message if failed
    """
    import time
    start_time = time.time()

    try:
        import discretize
        from simpeg import (
            maps,
            data,
            data_misfit,
            regularization,
            optimization,
            inverse_problem,
            inversion,
            directives,
        )
        from simpeg.potential_fields import gravity as grav_mod
        from simpeg.potential_fields import magnetics as mag_mod

        # --- Extract data ---
        x = dataset["X"].dropna().values
        y = dataset["Y"].dropna().values
        z = dataset["Z"].dropna().values if "Z" in dataset.columns else np.zeros(len(x))

        if "Value" in dataset.columns:
            obs_values = dataset["Value"].dropna().values
        else:
            # Find value column
            non_coord = [c for c in dataset.columns if c not in ("X", "Y", "Z")]
            obs_values = dataset[non_coord[0]].dropna().values if non_coord else None

        if obs_values is None or len(obs_values) == 0:
            return {"success": False, "error": "No observation values found in dataset"}

        n_data = len(obs_values)
        receiver_locations = np.column_stack([x[:n_data], y[:n_data], z[:n_data]])

        # --- Create mesh ---
        mesh = _create_mesh(receiver_locations, data_type)
        n_cells = mesh.nC

        # --- Set up survey ---
        if data_type == "gravity":
            receivers = grav_mod.Point(receiver_locations, components="gz")
            source_field = grav_mod.SourceField(receiver_list=[receivers])
            survey = grav_mod.Survey(source_field)

            # Simulation
            simulation = grav_mod.Simulation3DIntegral(
                mesh,
                survey=survey,
                rhoMap=maps.IdentityMap(nP=n_cells),
            )
        else:
            # Magnetic
            receivers = mag_mod.Point(receiver_locations, components="tmi")
            source_field = mag_mod.UniformBackgroundField(
                receiver_list=[receivers],
                amplitude=mag_intensity if mag_intensity > 0 else 50000.0,
                inclination=mag_inclination,
                declination=mag_declination,
            )
            survey = mag_mod.Survey(source_field)

            simulation = mag_mod.Simulation3DIntegral(
                mesh,
                survey=survey,
                chiMap=maps.IdentityMap(nP=n_cells),
            )

        # --- Set up inversion ---
        # Data misfit
        uncertainties = np.abs(obs_values) * 0.02 + np.abs(obs_values).max() * 0.01
        data_obj = data.Data(survey, dobs=obs_values, standard_deviation=uncertainties)
        dmis = data_misfit.L2DataMisfit(data=data_obj, simulation=simulation)

        # Regularization
        reg_multiplier = {"weak": 0.1, "medium": 1.0, "strong": 10.0}.get(reg_strength.lower(), 1.0)
        reg = regularization.WeightedLeastSquares(
            mesh,
            alpha_s=1.0 * reg_multiplier,
            alpha_x=1.0 * reg_multiplier,
            alpha_y=1.0 * reg_multiplier,
            alpha_z=1.0 * reg_multiplier,
        )

        # Optimization
        opt = optimization.InexactGaussNewton(
            maxIter=n_iterations,
            maxIterCG=20,
            tolCG=1e-3,
        )

        # Inverse problem
        inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)

        # Directives
        beta_schedule = directives.BetaSchedule(coolingFactor=2, coolingRate=1)
        beta_est = directives.BetaEstimate_ByEig(beta0_ratio=1e1)
        target_misfit = directives.TargetMisfit(chifact=1.0)

        inv = inversion.BaseInversion(
            inv_prob,
            directiveList=[beta_est, beta_schedule, target_misfit],
        )

        # --- Run ---
        starting_model = np.zeros(n_cells)
        model_result = inv.run(starting_model)

        # --- Collect results ---
        runtime = time.time() - start_time

        # Get misfit history from optimization
        misfit_history = []
        if hasattr(opt, "recall") and callable(getattr(opt, "recall", None)):
            try:
                recalled = opt.recall("phi_d")
                if recalled is not None:
                    misfit_history = [float(v) for v in recalled]
            except Exception:
                pass

        # Fallback: compute final misfit
        if not misfit_history:
            try:
                misfit_history = [float(dmis(model_result))]
            except Exception:
                misfit_history = []

        # Reshape model to 3D
        nx, ny, nz = mesh.shape_cells
        model_3d = model_result.reshape((nx, ny, nz), order="F")

        return {
            "success": True,
            "model": model_3d,
            "misfit_history": misfit_history if misfit_history else [float(dmis(model_result))],
            "mesh_dims": (nx, ny, nz),
            "iterations": opt.iter,
            "final_misfit": float(dmis(model_result)),
            "runtime": runtime,
            "parsed": {
                "iterations_completed": opt.iter,
                "final_rmse": float(np.sqrt(dmis(model_result) / n_data)),
                "model_min": float(model_result.min()),
                "model_max": float(model_result.max()),
                "memory_gb": None,
            },
        }

    except Exception as e:
        runtime = time.time() - start_time
        return {
            "success": False,
            "error": str(e),
            "runtime": runtime,
        }


def _create_mesh(
    receiver_locations: np.ndarray,
    data_type: str,
) -> Tuple:
    """Create a TensorMesh appropriate for the survey data.

    Returns:
        mesh object
    """
    import discretize

    # Get data extent
    x_min, x_max = receiver_locations[:, 0].min(), receiver_locations[:, 0].max()
    y_min, y_max = receiver_locations[:, 1].min(), receiver_locations[:, 1].max()
    z_max = receiver_locations[:, 2].max()

    x_range = x_max - x_min
    y_range = y_max - y_min
    n_stations = len(receiver_locations)

    # Determine cell sizes based on station spacing
    x_unique = np.sort(np.unique(receiver_locations[:, 0]))
    y_unique = np.sort(np.unique(receiver_locations[:, 1]))

    if len(x_unique) > 1:
        dx = np.median(np.diff(x_unique))
    else:
        dx = max(x_range, y_range) / 10.0

    if len(y_unique) > 1:
        dy = np.median(np.diff(y_unique))
    else:
        dy = dx

    # Use station spacing as cell size
    cell_size = min(dx, dy) if dx > 0 and dy > 0 else max(x_range, y_range) / 20.0
    if cell_size <= 0:
        cell_size = 1000.0

    # Profile handling
    is_profile = (x_range < 1.0 or y_range < 1.0)

    if is_profile:
        profile_range = max(x_range, y_range)
        cell_along = cell_size
        # Along profile
        n_along = max(5, int(np.ceil(profile_range / cell_along)))
        # Across: few cells
        n_across = 5
        cell_across = cell_along * 2.0
        # Depth
        dz = cell_along * 0.75
        depth = profile_range * 0.25
        n_depth = max(5, int(np.ceil(depth / dz)))

        if x_range > y_range:
            hx = np.ones(n_along) * cell_along
            hy = np.ones(n_across) * cell_across
        else:
            hx = np.ones(n_across) * cell_across
            hy = np.ones(n_along) * cell_along
        hz = np.ones(n_depth) * dz
    else:
        # 2D survey
        nx = max(5, min(50, int(np.ceil(x_range / cell_size))))
        ny = max(5, min(50, int(np.ceil(y_range / cell_size))))
        nz = max(5, min(30, int(np.ceil(max(x_range, y_range) * 0.3 / cell_size))))

        hx = np.ones(nx) * (x_range / nx)
        hy = np.ones(ny) * (y_range / ny)
        hz = np.ones(nz) * (max(x_range, y_range) * 0.3 / nz)

    # Origin: below and slightly outside data extent
    origin_x = x_min - hx[0] * 0.5
    origin_y = y_min - hy[0] * 0.5
    origin_z = z_max - hz.sum()  # mesh goes downward from surface

    mesh = discretize.TensorMesh([hx, hy, hz], origin=[origin_x, origin_y, origin_z])

    return mesh
