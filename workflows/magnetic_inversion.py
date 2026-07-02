"""
Magnetic Inversion Workflow for InversionAI.

Orchestrates a complete magnetic susceptibility inversion pipeline using
both Tomofast-x and SimPEG, then compares results.

Steps: validate_data -> configure_algorithms -> run_tomofast -> run_simpeg -> compare_results
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .base_workflow import (
    BaseWorkflow,
    ComparisonResult,
    WorkflowCancelledError,
    WorkflowPlan,
    WorkflowResult,
    WorkflowStatus,
    WorkflowStep,
)

logger = logging.getLogger(__name__)

# Sensible defaults for magnetic inversion
DEFAULT_ITERATIONS = 50
DEFAULT_DEPTH_WEIGHTING_BETA = 2.0
DEFAULT_SUSCEPTIBILITY_BOUNDS_MIN = 0.0    # SI units (non-negative for susceptibility)
DEFAULT_SUSCEPTIBILITY_BOUNDS_MAX = 0.1    # SI units
DEFAULT_INCLINATION = -60.0                # degrees (typical mid-latitude southern hemisphere)
DEFAULT_DECLINATION = 0.0                  # degrees
DEFAULT_FIELD_STRENGTH = 50000.0           # nT (typical Earth's field)
DEFAULT_TOLERANCE = 1e-4


class MagneticInversionWorkflow(BaseWorkflow):
    """Workflow for running magnetic susceptibility inversion.

    This workflow accepts observed magnetic anomaly data and station locations,
    configures both Tomofast-x and SimPEG for magnetic susceptibility inversion,
    runs both algorithms, and produces a comparison dashboard.

    Additional magnetic-specific parameters:
        - inclination: Geomagnetic field inclination (degrees)
        - declination: Geomagnetic field declination (degrees)
        - field_strength: Total field intensity (nT)

    Defaults:
        - 50 iterations
        - Depth weighting beta=2
        - Susceptibility bounds: [0.0, 0.1] SI
        - Inclination: -60° (southern hemisphere mid-latitude)
        - Declination: 0°
        - Field strength: 50000 nT

    Example:
        >>> workflow = MagneticInversionWorkflow()
        >>> config = {
        ...     "data_file": "magnetic_obs.csv",
        ...     "station_file": "stations.csv",
        ...     "mesh_file": "mesh.vtk",
        ...     "inclination": -63.5,
        ...     "declination": 2.1,
        ...     "field_strength": 54200.0,
        ...     "output_dir": "./results/magnetic"
        ... }
        >>> plan = workflow.plan("Run magnetic inversion", config)
        >>> result = workflow.execute(plan)
    """

    @property
    def name(self) -> str:
        return "Magnetic Inversion"

    @property
    def description(self) -> str:
        return (
            "Complete magnetic susceptibility inversion pipeline using Tomofast-x "
            "and SimPEG. Validates input data, configures both algorithms with "
            "inducing field parameters, runs inversions, and compares results."
        )

    @property
    def required_skills(self) -> List[str]:
        return ["tomofast", "simpeg", "data_validator", "visualization"]

    @property
    def steps(self) -> List[str]:
        return [
            "validate_data",
            "configure_algorithms",
            "run_tomofast",
            "run_simpeg",
            "compare_results",
        ]

    def plan(self, user_intent: str, data_config: Dict[str, Any]) -> WorkflowPlan:
        """Create a magnetic inversion execution plan.

        Args:
            user_intent: Natural language description of the task.
            data_config: Must contain:
                - data_file: Path to observed magnetic anomaly data (TMI or component).
                - station_file: Path to station/receiver locations (optional if in data_file).
                - mesh_file: Path to inversion mesh (optional, will generate).
                - inclination: Geomagnetic field inclination in degrees.
                - declination: Geomagnetic field declination in degrees.
                - field_strength: Total magnetic field intensity in nT.
                - output_dir: Directory for results (optional).
                - iterations: Number of iterations (default: 50).
                - depth_weighting_beta: Depth weighting exponent (default: 2.0).
                - bounds: [min, max] susceptibility (default: [0.0, 0.1] SI).

        Returns:
            WorkflowPlan with magnetic inversion steps.

        Raises:
            ValueError: If required data files are not specified.
        """
        self._set_status(WorkflowStatus.PLANNING)

        # Validate required fields
        if "data_file" not in data_config:
            raise ValueError(
                "data_config must include 'data_file' (path to observed magnetic anomaly). "
                "Expected format: CSV or UBC-GIF with columns [x, y, z, mag_anomaly]."
            )

        # Extract parameters with defaults
        iterations = data_config.get("iterations", DEFAULT_ITERATIONS)
        beta = data_config.get("depth_weighting_beta", DEFAULT_DEPTH_WEIGHTING_BETA)
        bounds = data_config.get(
            "bounds", [DEFAULT_SUSCEPTIBILITY_BOUNDS_MIN, DEFAULT_SUSCEPTIBILITY_BOUNDS_MAX]
        )
        inclination = data_config.get("inclination", DEFAULT_INCLINATION)
        declination = data_config.get("declination", DEFAULT_DECLINATION)
        field_strength = data_config.get("field_strength", DEFAULT_FIELD_STRENGTH)
        output_dir = data_config.get("output_dir", "./results/magnetic_inversion")

        # Magnetic field parameters dict
        field_params = {
            "inclination": inclination,
            "declination": declination,
            "field_strength": field_strength,
        }

        # Build workflow steps
        workflow_steps = [
            WorkflowStep(
                name="validate_data",
                skill="data_validator",
                action="validate_magnetic_data",
                params={
                    "data_file": data_config["data_file"],
                    "station_file": data_config.get("station_file", ""),
                    "mesh_file": data_config.get("mesh_file", ""),
                    "field_params": field_params,
                },
                status=WorkflowStatus.PLANNING,
            ),
            WorkflowStep(
                name="configure_algorithms",
                skill="configurator",
                action="configure_magnetic",
                params={
                    "iterations": iterations,
                    "depth_weighting_beta": beta,
                    "bounds": bounds,
                    "field_params": field_params,
                    "mesh_file": data_config.get("mesh_file", ""),
                    "output_dir": output_dir,
                },
                status=WorkflowStatus.PLANNING,
            ),
            WorkflowStep(
                name="run_tomofast",
                skill="tomofast",
                action="run_magnetic_inversion",
                params={
                    "data_file": data_config["data_file"],
                    "iterations": iterations,
                    "depth_weighting_beta": beta,
                    "bounds": bounds,
                    "field_params": field_params,
                    "output_dir": os.path.join(output_dir, "tomofast"),
                },
                status=WorkflowStatus.PLANNING,
            ),
            WorkflowStep(
                name="run_simpeg",
                skill="simpeg",
                action="run_magnetic_inversion",
                params={
                    "data_file": data_config["data_file"],
                    "iterations": iterations,
                    "depth_weighting_beta": beta,
                    "bounds": bounds,
                    "field_params": field_params,
                    "output_dir": os.path.join(output_dir, "simpeg"),
                },
                status=WorkflowStatus.PLANNING,
            ),
            WorkflowStep(
                name="compare_results",
                skill="visualization",
                action="compare_inversions",
                params={
                    "tomofast_output": os.path.join(output_dir, "tomofast"),
                    "simpeg_output": os.path.join(output_dir, "simpeg"),
                    "output_dir": os.path.join(output_dir, "comparison"),
                    "data_type": "magnetic_susceptibility",
                },
                status=WorkflowStatus.PLANNING,
            ),
        ]

        estimated_runtime = self._estimate_runtime(data_config, iterations)

        plan = WorkflowPlan(
            steps=workflow_steps,
            estimated_runtime=estimated_runtime,
            metadata={
                "user_intent": user_intent,
                "algorithm_count": 2,
                "iterations": iterations,
                "depth_weighting_beta": beta,
                "bounds": bounds,
                "field_params": field_params,
            },
        )

        logger.info(
            "Planned magnetic inversion workflow: %d steps, est. %s "
            "(inc=%.1f°, dec=%.1f°, B=%.0f nT)",
            len(workflow_steps), estimated_runtime,
            inclination, declination, field_strength,
        )
        return plan

    def execute(
        self,
        plan: WorkflowPlan,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> WorkflowResult:
        """Execute the magnetic inversion workflow plan.

        Args:
            plan: WorkflowPlan from self.plan().
            progress_callback: Optional callback(step_name, fraction).

        Returns:
            WorkflowResult with all outputs and comparison.

        Raises:
            RuntimeError: If validation or critical steps fail.
            WorkflowCancelledError: If cancelled during execution.
        """
        self.reset(preserve_cancellation=True)
        self._start_time = time.time()
        result = WorkflowResult()
        total_steps = len(plan.steps)

        try:
            for idx, step in enumerate(plan.steps):
                self._check_cancellation(step.name)
                self._current_step = step.name
                step.status = WorkflowStatus.RUNNING

                if step.name == "validate_data":
                    self._set_status(WorkflowStatus.VALIDATING)
                elif step.name == "compare_results":
                    self._set_status(WorkflowStatus.COMPARING)
                else:
                    self._set_status(WorkflowStatus.RUNNING)

                logger.info("Executing step %d/%d: %s", idx + 1, total_steps, step.name)

                if progress_callback:
                    progress_callback(step.name, idx / total_steps)

                step_result = self._execute_step(step)

                step.status = WorkflowStatus.COMPLETED
                result.steps_completed.append(step.name)
                result.results[step.name] = step_result
                self._save_intermediate(step.name, step_result)

                logger.info("Step '%s' completed successfully.", step.name)

            if "compare_results" in result.results:
                result.comparison = self._build_comparison(result.results["compare_results"])

            self._set_status(WorkflowStatus.COMPLETED)
            result.elapsed_time = time.time() - self._start_time

            if progress_callback:
                progress_callback("completed", 1.0)

            logger.info(
                "Magnetic inversion workflow completed in %.1fs", result.elapsed_time
            )

        except WorkflowCancelledError:
            result.elapsed_time = time.time() - self._start_time
            logger.warning("Workflow cancelled after %.1fs", result.elapsed_time)
            raise

        except Exception as e:
            self._set_status(WorkflowStatus.FAILED)
            result.elapsed_time = time.time() - self._start_time
            logger.error(
                "Magnetic inversion workflow failed at step '%s': %s",
                self._current_step, str(e),
            )
            raise RuntimeError(
                f"Magnetic inversion failed at step '{self._current_step}': {e}. "
                f"Completed steps: {result.steps_completed}. "
                f"Check intermediate results in output directory."
            ) from e

        return result

    def _execute_step(self, step: WorkflowStep) -> Dict[str, Any]:
        """Execute a single workflow step."""
        handlers = {
            "validate_data": self._step_validate_data,
            "configure_algorithms": self._step_configure_algorithms,
            "run_tomofast": self._step_run_tomofast,
            "run_simpeg": self._step_run_simpeg,
            "compare_results": self._step_compare_results,
        }

        handler = handlers.get(step.name)
        if handler is None:
            raise RuntimeError(f"Unknown step: {step.name}")

        return handler(step.params)

    def _step_validate_data(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input magnetic data files and field parameters."""
        data_file = params["data_file"]

        if not Path(data_file).exists():
            raise RuntimeError(
                f"Magnetic data file not found: {data_file}. "
                f"Please provide a valid path to observed magnetic anomaly data."
            )

        station_file = params.get("station_file", "")
        if station_file and not Path(station_file).exists():
            raise RuntimeError(
                f"Station file not found: {station_file}. "
                f"Please provide valid station/receiver locations."
            )

        # Validate field parameters
        field_params = params.get("field_params", {})
        inc = field_params.get("inclination", 0)
        dec = field_params.get("declination", 0)
        strength = field_params.get("field_strength", 0)

        if not (-90 <= inc <= 90):
            raise RuntimeError(
                f"Invalid inclination: {inc}°. Must be between -90° and 90°."
            )
        if not (-180 <= dec <= 180):
            raise RuntimeError(
                f"Invalid declination: {dec}°. Must be between -180° and 180°."
            )
        if strength <= 0:
            raise RuntimeError(
                f"Invalid field strength: {strength} nT. Must be positive."
            )

        logger.info("Magnetic data validation passed for: %s", data_file)
        return {
            "valid": True,
            "data_file": data_file,
            "station_file": station_file,
            "field_params": field_params,
            "message": "All input data and field parameters validated successfully.",
        }

    def _step_configure_algorithms(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Configure Tomofast-x and SimPEG for magnetic inversion."""
        output_dir = params.get("output_dir", "./results/magnetic_inversion")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        field_params = params["field_params"]

        config = {
            "tomofast": {
                "iterations": params["iterations"],
                "depth_weighting_beta": params["depth_weighting_beta"],
                "bounds": params["bounds"],
                "method": "LSQR",
                "data_type": "magnetic",
                "inclination": field_params["inclination"],
                "declination": field_params["declination"],
                "field_strength": field_params["field_strength"],
            },
            "simpeg": {
                "iterations": params["iterations"],
                "depth_weighting_beta": params["depth_weighting_beta"],
                "bounds": params["bounds"],
                "solver": "ProjectedGNCG",
                "data_type": "magnetic",
                "inducing_field": [
                    field_params["field_strength"],
                    field_params["inclination"],
                    field_params["declination"],
                ],
            },
        }

        config_path = os.path.join(output_dir, "algorithm_config.json")
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        logger.info("Magnetic algorithm configuration saved to: %s", config_path)
        return {"config": config, "config_path": config_path}

    def _step_run_tomofast(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run Tomofast-x magnetic susceptibility inversion."""
        output_dir = params.get("output_dir", "./results/magnetic_inversion/tomofast")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        field_params = params["field_params"]
        logger.info(
            "Running Tomofast-x magnetic inversion: %d iterations, "
            "inc=%.1f°, dec=%.1f°, B=%.0f nT",
            params["iterations"],
            field_params["inclination"],
            field_params["declination"],
            field_params["field_strength"],
        )

        # TODO: Integrate with actual Tomofast-x skill
        return {
            "algorithm": "tomofast-x",
            "output_dir": output_dir,
            "iterations_run": params["iterations"],
            "field_params": field_params,
            "status": "pending_integration",
            "expected_outputs": [
                os.path.join(output_dir, "model_susceptibility.vtk"),
                os.path.join(output_dir, "predicted_data.csv"),
                os.path.join(output_dir, "convergence.csv"),
            ],
        }

    def _step_run_simpeg(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run SimPEG magnetic susceptibility inversion."""
        output_dir = params.get("output_dir", "./results/magnetic_inversion/simpeg")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        field_params = params["field_params"]
        logger.info(
            "Running SimPEG magnetic inversion: %d iterations, "
            "inc=%.1f°, dec=%.1f°, B=%.0f nT",
            params["iterations"],
            field_params["inclination"],
            field_params["declination"],
            field_params["field_strength"],
        )

        # TODO: Integrate with actual SimPEG skill
        return {
            "algorithm": "simpeg",
            "output_dir": output_dir,
            "iterations_run": params["iterations"],
            "field_params": field_params,
            "status": "pending_integration",
            "expected_outputs": [
                os.path.join(output_dir, "model_susceptibility.vtk"),
                os.path.join(output_dir, "predicted_data.csv"),
                os.path.join(output_dir, "convergence.csv"),
            ],
        }

    def _step_compare_results(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Compare magnetic inversion results from Tomofast-x and SimPEG."""
        output_dir = params.get("output_dir", "./results/magnetic_inversion/comparison")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        logger.info("Comparing magnetic inversion results")

        # TODO: Integrate with visualization skill
        return {
            "comparison_dir": output_dir,
            "data_type": "magnetic_susceptibility",
            "metrics": {
                "rms_difference": 0.0,
                "correlation": 0.0,
                "model_norm_tomofast": 0.0,
                "model_norm_simpeg": 0.0,
            },
            "status": "pending_integration",
        }

    def _build_comparison(self, compare_data: Dict[str, Any]) -> ComparisonResult:
        """Build a ComparisonResult from comparison step output."""
        return ComparisonResult(
            metrics=compare_data.get("metrics", {}),
            summary=(
                "Magnetic susceptibility inversion comparison between Tomofast-x "
                "and SimPEG. See metrics for quantitative differences."
            ),
            figures=[],
            raw_data=compare_data,
        )

    def _estimate_runtime(self, data_config: Dict[str, Any], iterations: int) -> str:
        """Estimate total runtime based on data size and iterations."""
        # Magnetic inversions slightly slower due to forward operator complexity
        base_time_per_iter = 2.5  # seconds
        algo_count = 2
        estimated_seconds = base_time_per_iter * iterations * algo_count
        overhead = 30

        total = estimated_seconds + overhead
        if total < 60:
            return f"~{int(total)} seconds"
        elif total < 3600:
            return f"~{int(total / 60)} minutes"
        else:
            return f"~{total / 3600:.1f} hours"
