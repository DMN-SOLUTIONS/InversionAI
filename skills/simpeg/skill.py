"""
SimPEG inversion skill implementation.

Wraps the SimPEG (Simulation and Parameter Estimation in Geophysics)
Python library for in-process geophysical inversions with flexible
regularization and mesh options.
"""

import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..base_skill import (
    BaseSkill,
    InversionResult,
    ProgressCallback,
    RunConfig,
    SkillStatus,
    ValidationResult,
)
from .mesh_builder import MeshBuilder

logger = logging.getLogger(__name__)


class SimpegSkill(BaseSkill):
    """Skill for running SimPEG geophysical inversions.

    SimPEG is a Python-native framework for geophysical inversions that
    provides flexible regularization (Tikhonov, IRLS), multiple mesh types
    (TensorMesh, TreeMesh), and various optimization algorithms.

    This skill manages:
    - SimPEG installation verification
    - Mesh construction from data extents
    - Regularization and optimization setup
    - In-process inversion execution with progress callbacks
    - UBC mesh format output
    """

    def __init__(self, work_dir: str = "./simpeg_work") -> None:
        """Initialize SimpegSkill.

        Args:
            work_dir: Working directory for outputs.
        """
        super().__init__()
        self._work_dir = Path(work_dir)
        self._mesh_builder = MeshBuilder()
        self._simpeg_available: bool = False
        self._discretize_available: bool = False

    @property
    def name(self) -> str:
        return "simpeg"

    @property
    def version(self) -> str:
        return "0.21"

    @property
    def description(self) -> str:
        return (
            "SimPEG: Python-native geophysical inversion framework. "
            "Supports flexible regularization (Tikhonov, IRLS for compact models), "
            "multiple mesh types (TensorMesh, TreeMesh), and in-process execution."
        )

    @property
    def supported_data_types(self) -> list[str]:
        return ["gravity", "magnetic"]

    def setup(self) -> bool:
        """Verify SimPEG and dependencies are installed.

        Returns:
            True if SimPEG is importable and functional.
        """
        self._logger.info("Setting up SimPEG skill...")

        # Check SimPEG
        try:
            import SimPEG  # noqa: F401
            self._simpeg_available = True
            self._logger.info(f"SimPEG version: {SimPEG.__version__}")
        except ImportError:
            self._simpeg_available = False
            self._logger.error(
                "SimPEG not installed. Install with: pip install SimPEG"
            )

        # Check discretize
        try:
            import discretize  # noqa: F401
            self._discretize_available = True
            self._logger.info(f"discretize version: {discretize.__version__}")
        except ImportError:
            self._discretize_available = False
            self._logger.error(
                "discretize not installed. Install with: pip install discretize"
            )

        if self._simpeg_available and self._discretize_available:
            self._status = SkillStatus.READY
            return True
        else:
            self._status = SkillStatus.SETUP_REQUIRED
            return False

    def validate(self, data_config: dict[str, Any]) -> ValidationResult:
        """Validate data configuration for SimPEG inversion.

        Args:
            data_config: Expected keys:
                - 'data_path': Path to observation data (CSV or UBC format)
                - 'data_type': 'gravity' or 'magnetic'
                - Optional: 'mesh_type', 'n_cells', 'padding'

        Returns:
            ValidationResult with validation status.
        """
        errors: list[str] = []
        warnings: list[str] = []
        suggestions: list[str] = []

        # Check required keys
        if "data_path" not in data_config:
            errors.append("Missing required key 'data_path'")
        if "data_type" not in data_config:
            errors.append("Missing required key 'data_type'")

        if errors:
            return ValidationResult(valid=False, errors=errors, warnings=warnings, suggestions=suggestions)

        data_type = data_config["data_type"]
        if data_type not in self.supported_data_types:
            errors.append(f"Unsupported data type '{data_type}'. Supported: {self.supported_data_types}")
            return ValidationResult(valid=False, errors=errors, warnings=warnings, suggestions=suggestions)

        # Validate data file
        data_path = Path(data_config["data_path"])
        if not data_path.exists():
            errors.append(f"Data file not found: {data_path}")
        elif data_path.stat().st_size == 0:
            errors.append(f"Data file is empty: {data_path}")
        else:
            # Try to load and validate data format
            validation = self._validate_data_format(data_path, data_type)
            errors.extend(validation.get("errors", []))
            warnings.extend(validation.get("warnings", []))

        # Check coordinate system
        if "coordinate_system" not in data_config:
            suggestions.append(
                "Consider specifying 'coordinate_system' (default: local Cartesian)"
            )

        # Mesh type suggestions
        mesh_type = data_config.get("mesh_type", "TensorMesh")
        if mesh_type == "TreeMesh":
            suggestions.append(
                "TreeMesh provides adaptive refinement near data. "
                "Good for focused targets but slower to build."
            )

        # IRLS suggestion for compact bodies
        if data_config.get("params", {}).get("optimization") == "IRLS":
            suggestions.append(
                "IRLS inversion produces compact models. "
                "Requires more iterations (typically 40-60 IRLS iterations)."
            )

        # Check SimPEG availability
        if not self._simpeg_available:
            errors.append("SimPEG is not installed. Run setup() first.")

        valid = len(errors) == 0
        return ValidationResult(valid=valid, errors=errors, warnings=warnings, suggestions=suggestions)

    def _validate_data_format(self, data_path: Path, data_type: str) -> dict[str, list[str]]:
        """Validate data file format for SimPEG.

        Expects columns: x, y, z, value [, uncertainty]
        Supports CSV and space-delimited formats.

        Args:
            data_path: Path to data file.
            data_type: 'gravity' or 'magnetic'.

        Returns:
            Dict with 'errors' and 'warnings' lists.
        """
        result: dict[str, list[str]] = {"errors": [], "warnings": []}

        try:
            # Detect delimiter
            with open(data_path, "r") as f:
                first_lines = [f.readline() for _ in range(5)]

            # Skip comment/header lines
            data_lines = [l for l in first_lines if l.strip() and not l.strip().startswith(("#", "!", "%"))]
            if not data_lines:
                result["errors"].append("No data lines found")
                return result

            # Check column count
            sample_line = data_lines[0]
            if "," in sample_line:
                parts = sample_line.split(",")
            else:
                parts = sample_line.split()

            n_cols = len(parts)
            min_cols = 4  # x, y, z, value
            if n_cols < min_cols:
                result["errors"].append(
                    f"Expected at least {min_cols} columns (x, y, z, value), found {n_cols}"
                )

            if n_cols < 5:
                result["warnings"].append(
                    "No uncertainty column detected. Will use uniform weighting."
                )

            # Check numeric values
            try:
                [float(v.strip()) for v in parts[:min(4, n_cols)]]
            except ValueError:
                result["errors"].append("Non-numeric values in data columns")

        except IOError as e:
            result["errors"].append(f"Could not read data file: {e}")

        return result

    def configure(self, user_intent: str, data_config: dict[str, Any]) -> RunConfig:
        """Generate SimPEG inversion configuration from user intent.

        Args:
            user_intent: Natural language description of desired inversion.
            data_config: Data paths and parameters.

        Returns:
            RunConfig with mesh and inversion parameters.
        """
        self._logger.info(f"Configuring SimPEG from intent: '{user_intent}'")
        import re

        intent_lower = user_intent.lower()

        # Parse mesh type
        mesh_type = "TensorMesh"
        if "tree" in intent_lower or "octree" in intent_lower:
            mesh_type = "TreeMesh"

        # Parse optimization method
        optimization = "Gauss-Newton"
        if "irls" in intent_lower or "compact" in intent_lower or "sparse" in intent_lower:
            optimization = "IRLS"

        # Parse iterations
        n_iterations = 20  # SimPEG default
        iter_match = re.search(r"(\d+)\s*iteration", intent_lower)
        if iter_match:
            n_iterations = int(iter_match.group(1))

        if optimization == "IRLS" and n_iterations < 40:
            n_iterations = 40  # IRLS needs more iterations

        # Parse regularization
        alpha_s = 1e-4  # Smallness
        alpha_x = 1.0   # Smoothness x
        alpha_y = 1.0   # Smoothness y
        alpha_z = 1.0   # Smoothness z

        if "smooth" in intent_lower:
            alpha_s = 1e-6
        elif "compact" in intent_lower or "blocky" in intent_lower:
            alpha_s = 1e-2

        # Parse cell size
        cell_size = data_config.get("params", {}).get("cell_size")
        if not cell_size:
            # Will be auto-calculated from data spacing
            cell_size = None

        # Output directory
        output_dir = data_config.get("output_dir", str(self._work_dir / "output"))
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        mesh_config = {
            "mesh_type": mesh_type,
            "cell_size": cell_size,
            "padding_factor": data_config.get("params", {}).get("padding_factor", 1.5),
            "depth_factor": data_config.get("params", {}).get("depth_factor", 2.0),
            "n_pad_cells": data_config.get("params", {}).get("n_pad_cells", 5),
        }

        inversion_params = {
            "data_type": data_config.get("data_type", "gravity"),
            "optimization": optimization,
            "n_iterations": n_iterations,
            "alpha_s": alpha_s,
            "alpha_x": alpha_x,
            "alpha_y": alpha_y,
            "alpha_z": alpha_z,
            "chi_factor": data_config.get("params", {}).get("chi_factor", 1.0),
            "upper_bound": data_config.get("params", {}).get("upper_bound"),
            "lower_bound": data_config.get("params", {}).get("lower_bound"),
        }

        return RunConfig(
            algorithm="simpeg",
            data_path=data_config["data_path"],
            mesh_config=mesh_config,
            inversion_params=inversion_params,
            output_dir=output_dir,
        )

    def run(
        self,
        config: RunConfig,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> InversionResult:
        """Execute SimPEG inversion in-process.

        Args:
            config: RunConfig from configure().
            progress_callback: Optional callback for progress.

        Returns:
            InversionResult with model path and metrics.

        Raises:
            RuntimeError: If inversion fails.
        """
        self._status = SkillStatus.RUNNING
        self._logger.info(f"Starting SimPEG inversion: {config.inversion_params}")

        start_time = time.time()
        misfit_history: list[float] = []

        try:
            import SimPEG
            from SimPEG import (
                data_misfit,
                directives,
                inverse_problem,
                inversion,
                maps,
                optimization,
                regularization,
            )
            from SimPEG.potential_fields import gravity as pf_gravity
            from SimPEG.potential_fields import magnetics as pf_magnetics
            import discretize

            # Load data
            data_array = self._load_data(config.data_path)
            locations = data_array[:, :3]
            observations = data_array[:, 3]
            uncertainties = (
                data_array[:, 4] if data_array.shape[1] > 4
                else np.abs(observations) * 0.02 + np.std(observations) * 0.01
            )

            # Build mesh
            mesh = self._mesh_builder.build_mesh(
                locations=locations,
                mesh_type=config.mesh_config.get("mesh_type", "TensorMesh"),
                cell_size=config.mesh_config.get("cell_size"),
                padding_factor=config.mesh_config.get("padding_factor", 1.5),
                depth_factor=config.mesh_config.get("depth_factor", 2.0),
                n_pad_cells=config.mesh_config.get("n_pad_cells", 5),
            )

            n_cells = mesh.nC
            self._logger.info(f"Mesh built: {n_cells} cells")

            # Set up the inversion components
            data_type = config.inversion_params.get("data_type", "gravity")

            if data_type == "gravity":
                # Gravity inversion setup
                receiver_list = [pf_gravity.receivers.Point(locations, components=["gz"])]
                source_field = pf_gravity.sources.SourceField(receiver_list=receiver_list)
                survey = pf_gravity.survey.Survey(source_field)

                simulation = pf_gravity.simulation.Simulation3DIntegral(
                    mesh=mesh,
                    survey=survey,
                    rhoMap=maps.IdentityMap(nP=n_cells),
                    ind_active=np.ones(n_cells, dtype=bool),
                    store_sensitivities="forward_only",
                )
            elif data_type == "magnetic":
                # Magnetic inversion setup
                receiver_list = [pf_magnetics.receivers.Point(locations, components=["tmi"])]
                source_field = pf_magnetics.sources.UniformBackgroundField(
                    receiver_list=receiver_list,
                    amplitude=50000.0,
                    inclination=90.0,
                    declination=0.0,
                )
                survey = pf_magnetics.survey.Survey(source_field)

                simulation = pf_magnetics.simulation.Simulation3DIntegral(
                    mesh=mesh,
                    survey=survey,
                    chiMap=maps.IdentityMap(nP=n_cells),
                    ind_active=np.ones(n_cells, dtype=bool),
                    store_sensitivities="forward_only",
                )
            else:
                raise RuntimeError(f"Unsupported data type for SimPEG: {data_type}")

            # Data object
            data_obj = SimPEG.data.Data(
                survey=survey,
                dobs=observations,
                standard_deviation=uncertainties,
            )

            # Data misfit
            dmis = data_misfit.L2DataMisfit(data=data_obj, simulation=simulation)

            # Regularization
            reg = regularization.WeightedLeastSquares(
                mesh,
                alpha_s=config.inversion_params.get("alpha_s", 1e-4),
                alpha_x=config.inversion_params.get("alpha_x", 1.0),
                alpha_y=config.inversion_params.get("alpha_y", 1.0),
                alpha_z=config.inversion_params.get("alpha_z", 1.0),
            )

            # Optimization
            n_iterations = config.inversion_params.get("n_iterations", 20)
            opt = optimization.ProjectedGNCG(
                maxIter=n_iterations,
                upper=config.inversion_params.get("upper_bound", np.inf),
                lower=config.inversion_params.get("lower_bound", -np.inf),
            )

            # Inverse problem
            inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)

            # Directives
            directive_list = [
                directives.BetaEstimate_ByEig(beta0_ratio=1e1),
                directives.TargetMisfit(chifact=config.inversion_params.get("chi_factor", 1.0)),
                directives.BetaSchedule(coolingFactor=2, coolingRate=1),
            ]

            # Add IRLS if requested
            optimization_type = config.inversion_params.get("optimization", "Gauss-Newton")
            if optimization_type == "IRLS":
                irls = directives.Update_IRLS(
                    f_min_change=1e-4,
                    max_irls_iterations=40,
                    coolingFactor=2,
                    coolingRate=1,
                )
                directive_list.append(irls)

            # Run inversion
            inv = inversion.BaseInversion(inv_prob, directiveList=directive_list)

            # Custom callback to capture progress
            original_callback = opt.callback

            def _progress_hook(opt_instance: Any) -> None:
                iteration = opt_instance.iter
                phi_d = float(inv_prob.phi_d) if hasattr(inv_prob, "phi_d") else 0.0
                misfit_history.append(phi_d)
                if progress_callback:
                    progress_callback(iteration, n_iterations, phi_d)
                self._logger.debug(f"SimPEG iteration {iteration}, phi_d={phi_d:.4f}")
                if original_callback:
                    original_callback(opt_instance)

            opt.callback = _progress_hook

            # Initial model
            m0 = np.zeros(n_cells)

            # Execute
            self._logger.info("Running SimPEG inversion...")
            recovered_model = inv.run(m0)

            runtime = time.time() - start_time

            # Save results
            output_dir = Path(config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            model_path = str(output_dir / "recovered_model.txt")
            mesh_path = str(output_dir / "mesh.ubc")

            # Save model
            np.savetxt(model_path, recovered_model)

            # Save mesh in UBC format
            self._save_ubc_mesh(mesh, mesh_path)

            final_misfit = misfit_history[-1] if misfit_history else 0.0

            result = InversionResult(
                model_path=model_path,
                mesh_path=mesh_path,
                misfit_history=misfit_history,
                final_misfit=final_misfit,
                n_iterations=len(misfit_history),
                runtime_seconds=runtime,
                metadata={
                    "algorithm": "simpeg",
                    "mesh_type": config.mesh_config.get("mesh_type", "TensorMesh"),
                    "n_cells": n_cells,
                    "optimization": optimization_type,
                    "data_type": data_type,
                },
            )

            self._last_result = result
            self._status = SkillStatus.COMPLETED
            self._logger.info(f"SimPEG inversion completed: {result.summary()}")
            return result

        except ImportError as e:
            self._status = SkillStatus.FAILED
            raise RuntimeError(f"SimPEG import error: {e}. Run setup() first.")
        except Exception as e:
            self._status = SkillStatus.FAILED
            raise RuntimeError(f"SimPEG inversion failed: {e}")

    def _load_data(self, data_path: str) -> "np.ndarray":
        """Load observation data from file.

        Supports CSV and space-delimited formats.

        Args:
            data_path: Path to data file.

        Returns:
            NumPy array with columns [x, y, z, value, ...].
        """
        path = Path(data_path)

        # Try comma-separated first, then whitespace
        try:
            if path.suffix.lower() == ".csv":
                data = np.loadtxt(data_path, delimiter=",", comments=("#", "!", "%"))
            else:
                data = np.loadtxt(data_path, comments=("#", "!", "%"))
        except ValueError:
            # Try skipping header row
            data = np.loadtxt(data_path, skiprows=1, comments=("#", "!", "%"))

        if data.ndim == 1:
            raise ValueError(f"Data file has only one column or row: {data_path}")
        if data.shape[1] < 4:
            raise ValueError(
                f"Data file needs at least 4 columns (x, y, z, value), "
                f"found {data.shape[1]}"
            )

        self._logger.info(f"Loaded {data.shape[0]} data points from {data_path}")
        return data

    def _save_ubc_mesh(self, mesh: Any, output_path: str) -> None:
        """Save mesh in UBC format.

        Args:
            mesh: discretize mesh object.
            output_path: Path to write UBC mesh file.
        """
        try:
            # discretize meshes have write_UBC method
            if hasattr(mesh, "write_UBC"):
                mesh.write_UBC(output_path)
            else:
                # Manual UBC format for TensorMesh
                with open(output_path, "w") as f:
                    # Line 1: number of cells in each direction
                    if hasattr(mesh, "shape_cells"):
                        nx, ny, nz = mesh.shape_cells
                    else:
                        nx, ny, nz = mesh.vnC

                    f.write(f"{nx} {ny} {nz}\n")

                    # Line 2: origin (top SW corner in UBC convention)
                    origin = mesh.origin
                    f.write(f"{origin[0]} {origin[1]} {origin[2]}\n")

                    # Lines 3+: cell widths
                    f.write(" ".join(f"{w:.4f}" for w in mesh.h[0]) + "\n")
                    f.write(" ".join(f"{w:.4f}" for w in mesh.h[1]) + "\n")
                    f.write(" ".join(f"{w:.4f}" for w in mesh.h[2]) + "\n")

            self._logger.info(f"Mesh saved to UBC format: {output_path}")
        except Exception as e:
            self._logger.warning(f"Could not save UBC mesh: {e}")

    def get_results(self) -> InversionResult:
        """Retrieve the last inversion result.

        Returns:
            InversionResult from the most recent run.

        Raises:
            RuntimeError: If no results available.
        """
        if self._last_result is None:
            raise RuntimeError("No results available. Run an inversion first.")
        return self._last_result
