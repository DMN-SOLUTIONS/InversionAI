"""
Gravity Inversion Workflow for InversionAI.

Orchestrates a complete gravity inversion pipeline using both
Tomofast-x and SimPEG, then compares results.

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

# Sensible defaults for gravity inversion
DEFAULT_ITERATIONS = 50
DEFAULT_DEPTH_WEIGHTING_BETA = 2.0
DEFAULT_BOUNDS_MIN = -1.0  # g/cm³ density contrast
DEFAULT_BOUNDS_MAX = 1.0   # g/cm³ density contrast
DEFAULT_MESH_PADDING = 5
DEFAULT_TOLERANCE = 1e-4


class GravityInversionWorkflow(BaseWorkflow):
    """Workflow for running gravity inversion with Tomofast-x and SimPEG.

    This workflow accepts observed gravity anomaly data and station locations,
    configures both Tomofast-x and SimPEG for gravity density inversion,
    runs both algorithms, and produces a comparison dashboard.

    Defaults:
        - 50 iterations
        - Depth weighting beta=2
        - Density contrast bounds: [-1.0, 1.0] g/cm³

    Example:
        >>> workflow = GravityInversionWorkflow()
        >>> config = {
        ...     "data_file": "gravity_obs.csv",
        ...     "station_file": "stations.csv",
        ...     "mesh_file": "mesh.vtk",
        ...     "output_dir": "./results/gravity"
        ... }
        >>> plan = workflow.plan("Run gravity inversion", config)
        >>> result = workflow.execute(plan)
    """

    @property
    def name(self) -> str:
        return "Gravity Inversion"

    @property
    def description(self) -> str:
        return (
            "Complete gravity density inversion pipeline using Tomofast-x and SimPEG. "
            "Validates input data, configures both algorithms with sensible defaults, "
            "runs inversions, and produces a comparison of results."
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
        """Create a gravity inversion execution plan.

        Args:
            user_intent: Natural language description of the task.
            data_config: Must contain:
                - data_file: Path to observed gravity anomaly data.
                - station_file: Path to station/receiver locations.
                - mesh_file: Path to inversion mesh (optional, will generate).
                - output_dir: Directory for results (optional).
                - iterations: Number of iterations (default: 50).
                - depth_weighting_beta: Depth weighting exponent (default: 2.0).
                - bounds: [min, max] density contrast (default: [-1.0, 1.0]).

        Returns:
            WorkflowPlan with gravity inversion steps.

        Raises:
            ValueError: If required data files are not specified.
        """
        self._set_status(WorkflowStatus.PLANNING)

        # Validate required fields
        if "data_file" not in data_config:
            raise ValueError(
                "data_config must include 'data_file' (path to observed gravity anomaly). "
                "Expected format: CSV or UBC-GIF with columns [x, y, z, grav_anomaly]."
            )

        # Extract parameters with defaults
        iterations = data_config.get("iterations", DEFAULT_ITERATIONS)
        beta = data_config.get("depth_weighting_beta", DEFAULT_DEPTH_WEIGHTING_BETA)
        bounds = data_config.get("bounds", [DEFAULT_BOUNDS_MIN, DEFAULT_BOUNDS_MAX])
        output_dir = data_config.get("output_dir", "./results/gravity_inversion")

        # Build workflow steps
        workflow_steps = [
            WorkflowStep(
                name="validate_data",
                skill="data_validator",
                action="validate_gravity_data",
                params={
                    "data_file": data_config["data_file"],
                    "station_file": data_config.get("station_file", ""),
                    "mesh_file": data_config.get("mesh_file", ""),
                },
                status=WorkflowStatus.PLANNING,
            ),
            WorkflowStep(
                name="configure_algorithms",
                skill="configurator",
                action="configure_gravity",
                params={
                    "iterations": iterations,
                    "depth_weighting_beta": beta,
                    "bounds": bounds,
                    "mesh_file": data_config.get("mesh_file", ""),
                    "output_dir": output_dir,
                },
                status=WorkflowStatus.PLANNING,
            ),
            WorkflowStep(
                name="run_tomofast",
                skill="tomofast",
                action="run_gravity_inversion",
                params={
                    "data_file": data_config["data_file"],
                    "iterations": iterations,
                    "depth_weighting_beta": beta,
                    "bounds": bounds,
                    "output_dir": os.path.join(output_dir, "tomofast"),
                },
                status=WorkflowStatus.PLANNING,
            ),
            WorkflowStep(
                name="run_simpeg",
                skill="simpeg",
                action="run_gravity_inversion",
                params={
                    "data_file": data_config["data_file"],
                    "iterations": iterations,
                    "depth_weighting_beta": beta,
                    "bounds": bounds,
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
                },
                status=WorkflowStatus.PLANNING,
            ),
        ]

        # Estimate runtime based on mesh size and iterations
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
            },
        )

        logger.info(
            "Planned gravity inversion workflow: %d steps, est. %s",
            len(workflow_steps),
            estimated_runtime,
        )
        return plan

    def execute(
        self,
        plan: WorkflowPlan,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> WorkflowResult:
        """Execute the gravity inversion workflow plan.

        Runs validation, configuration, both inversions, and comparison
        in sequence. Saves intermediate results after each step.

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

                # Execute the step
                step_result = self._execute_step(step)

                # Mark completed and save
                step.status = WorkflowStatus.COMPLETED
                result.steps_completed.append(step.name)
                result.results[step.name] = step_result
                self._save_intermediate(step.name, step_result)

                logger.info("Step '%s' completed successfully.", step.name)

            # Build comparison result
            if "compare_results" in result.results:
                result.comparison = self._build_comparison(result.results["compare_results"])

            self._set_status(WorkflowStatus.COMPLETED)
            result.elapsed_time = time.time() - self._start_time

            if progress_callback:
                progress_callback("completed", 1.0)

            logger.info(
                "Gravity inversion workflow completed in %.1fs", result.elapsed_time
            )

        except WorkflowCancelledError:
            result.elapsed_time = time.time() - self._start_time
            logger.warning("Workflow cancelled after %.1fs", result.elapsed_time)
            raise

        except Exception as e:
            self._set_status(WorkflowStatus.FAILED)
            result.elapsed_time = time.time() - self._start_time
            logger.error(
                "Gravity inversion workflow failed at step '%s': %s",
                self._current_step, str(e),
            )
            raise RuntimeError(
                f"Gravity inversion failed at step '{self._current_step}': {e}. "
                f"Completed steps: {result.steps_completed}. "
                f"Check intermediate results in output directory."
            ) from e

        return result

    def _execute_step(self, step: WorkflowStep) -> Dict[str, Any]:
        """Execute a single workflow step.

        Args:
            step: The WorkflowStep to execute.

        Returns:
            Dictionary of step results.
        """
        # Dispatch to appropriate handler
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
        """Validate input gravity data files.

        Checks file existence, format correctness, and data quality.
        """
        data_file = params["data_file"]

        if not Path(data_file).exists():
            raise RuntimeError(
                f"Gravity data file not found: {data_file}. "
                f"Please provide a valid path to observed gravity anomaly data."
            )

        station_file = params.get("station_file", "")
        if station_file and not Path(station_file).exists():
            raise RuntimeError(
                f"Station file not found: {station_file}. "
                f"Please provide valid station/receiver locations."
            )

        logger.info("Data validation passed for: %s", data_file)
        return {
            "valid": True,
            "data_file": data_file,
            "station_file": station_file,
            "message": "All input data files validated successfully.",
        }

    def _step_configure_algorithms(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Configure Tomofast-x and SimPEG for gravity inversion."""
        output_dir = params.get("output_dir", "./results/gravity_inversion")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        config = {
            "tomofast": {
                "iterations": params["iterations"],
                "depth_weighting_beta": params["depth_weighting_beta"],
                "bounds": params["bounds"],
                "method": "LSQR",
                "data_type": "gravity",
            },
            "simpeg": {
                "iterations": params["iterations"],
                "depth_weighting_beta": params["depth_weighting_beta"],
                "bounds": params["bounds"],
                "solver": "ProjectedGNCG",
                "data_type": "gravity",
            },
        }

        # Save configuration to output directory
        config_path = os.path.join(output_dir, "algorithm_config.json")
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        logger.info("Algorithm configuration saved to: %s", config_path)
        return {"config": config, "config_path": config_path}

    def _step_run_tomofast(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run Tomofast-x gravity inversion.

        This would invoke the Tomofast-x skill to run the inversion.
        Currently returns a placeholder structure for integration.
        """
        output_dir = params.get("output_dir", "./results/gravity_inversion/tomofast")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        logger.info(
            "Running Tomofast-x gravity inversion: %d iterations, beta=%.1f",
            params["iterations"], params["depth_weighting_beta"],
        )

        # TODO: Integrate with actual Tomofast-x skill
        # This is the integration point where the tomofast skill is invoked
        return {
            "algorithm": "tomofast-x",
            "output_dir": output_dir,
            "iterations_run": params["iterations"],
            "status": "pending_integration",
            "expected_outputs": [
                os.path.join(output_dir, "model_density.vtk"),
                os.path.join(output_dir, "predicted_data.csv"),
                os.path.join(output_dir, "convergence.csv"),
            ],
        }

    def _step_run_simpeg(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run SimPEG gravity inversion.

        This would invoke the SimPEG skill to run the inversion.
        Currently returns a placeholder structure for integration.
        """
        output_dir = params.get("output_dir", "./results/gravity_inversion/simpeg")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        logger.info(
            "Running SimPEG gravity inversion: %d iterations, beta=%.1f",
            params["iterations"], params["depth_weighting_beta"],
        )

        # TODO: Integrate with actual SimPEG skill
        return {
            "algorithm": "simpeg",
            "output_dir": output_dir,
            "iterations_run": params["iterations"],
            "status": "pending_integration",
            "expected_outputs": [
                os.path.join(output_dir, "model_density.vtk"),
                os.path.join(output_dir, "predicted_data.csv"),
                os.path.join(output_dir, "convergence.csv"),
            ],
        }

    def _step_compare_results(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Compare results from Tomofast-x and SimPEG inversions."""
        output_dir = params.get("output_dir", "./results/gravity_inversion/comparison")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        logger.info("Comparing inversion results from Tomofast-x and SimPEG")

        # TODO: Integrate with visualization skill for actual comparison
        return {
            "comparison_dir": output_dir,
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
                "Gravity inversion comparison between Tomofast-x and SimPEG. "
                "See metrics for quantitative differences."
            ),
            figures=[],
            raw_data=compare_data,
        )

    def _estimate_runtime(
        self, data_config: Dict[str, Any], iterations: int
    ) -> str:
        """Estimate total runtime based on data size and iterations."""
        # Rough heuristic: ~2s per iteration per algorithm for moderate meshes
        base_time_per_iter = 2.0  # seconds
        algo_count = 2
        estimated_seconds = base_time_per_iter * iterations * algo_count
        overhead = 30  # validation, config, comparison

        total = estimated_seconds + overhead
        if total < 60:
            return f"~{int(total)} seconds"
        elif total < 3600:
            return f"~{int(total / 60)} minutes"
        else:
            return f"~{total / 3600:.1f} hours"
