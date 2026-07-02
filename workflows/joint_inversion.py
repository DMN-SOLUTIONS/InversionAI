"""
Joint Inversion Workflow for InversionAI.

Orchestrates joint gravity + magnetic inversion using cross-gradient coupling
(Tomofast-x natively) and structural coupling via Haber & Gazit approach (SimPEG).
Compares joint vs individual inversion results.

Steps: validate_data -> configure_algorithms -> run_tomofast_joint ->
       run_simpeg_coupled -> run_individual_inversions -> compare_results
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

# Sensible defaults for joint inversion
DEFAULT_ITERATIONS = 50
DEFAULT_DEPTH_WEIGHTING_BETA = 2.0
DEFAULT_DENSITY_BOUNDS = [-1.0, 1.0]       # g/cm³
DEFAULT_SUSCEPTIBILITY_BOUNDS = [0.0, 0.1]  # SI
DEFAULT_INCLINATION = -60.0
DEFAULT_DECLINATION = 0.0
DEFAULT_FIELD_STRENGTH = 50000.0
DEFAULT_CROSS_GRADIENT_WEIGHT = 1.0
DEFAULT_STRUCTURAL_COUPLING_WEIGHT = 0.5


class JointInversionWorkflow(BaseWorkflow):
    """Workflow for joint gravity + magnetic inversion.

    Tomofast-x supports joint inversion natively via cross-gradient coupling.
    For SimPEG, separate inversions are run with structural coupling using
    the Haber & Gazit (2013) approach.

    The workflow also runs individual (uncoupled) inversions for comparison,
    demonstrating the benefit of joint inversion.

    Example:
        >>> workflow = JointInversionWorkflow()
        >>> config = {
        ...     "gravity_data_file": "gravity_obs.csv",
        ...     "magnetic_data_file": "magnetic_obs.csv",
        ...     "mesh_file": "mesh.vtk",
        ...     "inclination": -63.5,
        ...     "declination": 2.1,
        ...     "field_strength": 54200.0,
        ...     "output_dir": "./results/joint"
        ... }
        >>> plan = workflow.plan("Run joint inversion", config)
        >>> result = workflow.execute(plan)
    """

    @property
    def name(self) -> str:
        return "Joint Gravity-Magnetic Inversion"

    @property
    def description(self) -> str:
        return (
            "Joint gravity and magnetic inversion with cross-gradient coupling "
            "(Tomofast-x) and structural coupling via Haber & Gazit approach "
            "(SimPEG). Compares joint results against individual inversions."
        )

    @property
    def required_skills(self) -> List[str]:
        return ["tomofast", "simpeg", "data_validator", "visualization"]

    @property
    def steps(self) -> List[str]:
        return [
            "validate_data",
            "configure_algorithms",
            "run_tomofast_joint",
            "run_simpeg_coupled",
            "run_individual_inversions",
            "compare_results",
        ]

    def plan(self, user_intent: str, data_config: Dict[str, Any]) -> WorkflowPlan:
        """Create a joint inversion execution plan.

        Args:
            user_intent: Natural language description of the task.
            data_config: Must contain:
                - gravity_data_file: Path to observed gravity anomaly data.
                - magnetic_data_file: Path to observed magnetic anomaly data.
                - mesh_file: Path to shared inversion mesh (optional).
                - inclination: Geomagnetic field inclination (degrees).
                - declination: Geomagnetic field declination (degrees).
                - field_strength: Total field intensity (nT).
                - output_dir: Directory for results (optional).
                - iterations: Number of iterations (default: 50).
                - cross_gradient_weight: Weight for cross-gradient term (default: 1.0).
                - structural_coupling_weight: Weight for SimPEG coupling (default: 0.5).
                - density_bounds: [min, max] density contrast (default: [-1.0, 1.0]).
                - susceptibility_bounds: [min, max] susceptibility (default: [0.0, 0.1]).

        Returns:
            WorkflowPlan with joint inversion steps.

        Raises:
            ValueError: If required data files are not specified.
        """
        self._set_status(WorkflowStatus.PLANNING)

        # Validate required fields
        if "gravity_data_file" not in data_config:
            raise ValueError(
                "data_config must include 'gravity_data_file' for joint inversion. "
                "Provide path to observed gravity anomaly data."
            )
        if "magnetic_data_file" not in data_config:
            raise ValueError(
                "data_config must include 'magnetic_data_file' for joint inversion. "
                "Provide path to observed magnetic anomaly data."
            )

        # Extract parameters with defaults
        iterations = data_config.get("iterations", DEFAULT_ITERATIONS)
        beta = data_config.get("depth_weighting_beta", DEFAULT_DEPTH_WEIGHTING_BETA)
        density_bounds = data_config.get("density_bounds", DEFAULT_DENSITY_BOUNDS)
        susc_bounds = data_config.get("susceptibility_bounds", DEFAULT_SUSCEPTIBILITY_BOUNDS)
        inclination = data_config.get("inclination", DEFAULT_INCLINATION)
        declination = data_config.get("declination", DEFAULT_DECLINATION)
        field_strength = data_config.get("field_strength", DEFAULT_FIELD_STRENGTH)
        cross_grad_weight = data_config.get("cross_gradient_weight", DEFAULT_CROSS_GRADIENT_WEIGHT)
        coupling_weight = data_config.get("structural_coupling_weight", DEFAULT_STRUCTURAL_COUPLING_WEIGHT)
        output_dir = data_config.get("output_dir", "./results/joint_inversion")

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
                action="validate_joint_data",
                params={
                    "gravity_data_file": data_config["gravity_data_file"],
                    "magnetic_data_file": data_config["magnetic_data_file"],
                    "mesh_file": data_config.get("mesh_file", ""),
                    "field_params": field_params,
                },
                status=WorkflowStatus.PLANNING,
            ),
            WorkflowStep(
                name="configure_algorithms",
                skill="configurator",
                action="configure_joint",
                params={
                    "iterations": iterations,
                    "depth_weighting_beta": beta,
                    "density_bounds": density_bounds,
                    "susceptibility_bounds": susc_bounds,
                    "field_params": field_params,
                    "cross_gradient_weight": cross_grad_weight,
                    "structural_coupling_weight": coupling_weight,
                    "output_dir": output_dir,
                },
                status=WorkflowStatus.PLANNING,
            ),
            WorkflowStep(
                name="run_tomofast_joint",
                skill="tomofast",
                action="run_joint_inversion",
                params={
                    "gravity_data_file": data_config["gravity_data_file"],
                    "magnetic_data_file": data_config["magnetic_data_file"],
                    "iterations": iterations,
                    "depth_weighting_beta": beta,
                    "density_bounds": density_bounds,
                    "susceptibility_bounds": susc_bounds,
                    "field_params": field_params,
                    "cross_gradient_weight": cross_grad_weight,
                    "output_dir": os.path.join(output_dir, "tomofast_joint"),
                },
                status=WorkflowStatus.PLANNING,
            ),
            WorkflowStep(
                name="run_simpeg_coupled",
                skill="simpeg",
                action="run_coupled_inversion",
                params={
                    "gravity_data_file": data_config["gravity_data_file"],
                    "magnetic_data_file": data_config["magnetic_data_file"],
                    "iterations": iterations,
                    "depth_weighting_beta": beta,
                    "density_bounds": density_bounds,
                    "susceptibility_bounds": susc_bounds,
                    "field_params": field_params,
                    "structural_coupling_weight": coupling_weight,
                    "output_dir": os.path.join(output_dir, "simpeg_coupled"),
                },
                status=WorkflowStatus.PLANNING,
            ),
            WorkflowStep(
                name="run_individual_inversions",
                skill="simpeg",
                action="run_individual_inversions",
                params={
                    "gravity_data_file": data_config["gravity_data_file"],
                    "magnetic_data_file": data_config["magnetic_data_file"],
                    "iterations": iterations,
                    "depth_weighting_beta": beta,
                    "density_bounds": density_bounds,
                    "susceptibility_bounds": susc_bounds,
                    "field_params": field_params,
                    "output_dir": os.path.join(output_dir, "individual"),
                },
                status=WorkflowStatus.PLANNING,
            ),
            WorkflowStep(
                name="compare_results",
                skill="visualization",
                action="compare_joint_inversions",
                params={
                    "tomofast_joint_output": os.path.join(output_dir, "tomofast_joint"),
                    "simpeg_coupled_output": os.path.join(output_dir, "simpeg_coupled"),
                    "individual_output": os.path.join(output_dir, "individual"),
                    "output_dir": os.path.join(output_dir, "comparison"),
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
                "joint_methods": ["cross_gradient", "structural_coupling"],
                "iterations": iterations,
                "cross_gradient_weight": cross_grad_weight,
                "structural_coupling_weight": coupling_weight,
                "field_params": field_params,
            },
        )

        logger.info(
            "Planned joint inversion workflow: %d steps, est. %s",
            len(workflow_steps), estimated_runtime,
        )
        return plan

    def execute(
        self,
        plan: WorkflowPlan,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> WorkflowResult:
        """Execute the joint inversion workflow plan.

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
                "Joint inversion workflow completed in %.1fs", result.elapsed_time
            )

        except WorkflowCancelledError:
            result.elapsed_time = time.time() - self._start_time
            logger.warning("Workflow cancelled after %.1fs", result.elapsed_time)
            raise

        except Exception as e:
            self._set_status(WorkflowStatus.FAILED)
            result.elapsed_time = time.time() - self._start_time
            logger.error(
                "Joint inversion workflow failed at step '%s': %s",
                self._current_step, str(e),
            )
            raise RuntimeError(
                f"Joint inversion failed at step '{self._current_step}': {e}. "
                f"Completed steps: {result.steps_completed}. "
                f"Check intermediate results in output directory."
            ) from e

        return result

    def _execute_step(self, step: WorkflowStep) -> Dict[str, Any]:
        """Execute a single workflow step."""
        handlers = {
            "validate_data": self._step_validate_data,
            "configure_algorithms": self._step_configure_algorithms,
            "run_tomofast_joint": self._step_run_tomofast_joint,
            "run_simpeg_coupled": self._step_run_simpeg_coupled,
            "run_individual_inversions": self._step_run_individual,
            "compare_results": self._step_compare_results,
        }

        handler = handlers.get(step.name)
        if handler is None:
            raise RuntimeError(f"Unknown step: {step.name}")

        return handler(step.params)

    def _step_validate_data(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input gravity and magnetic data files."""
        gravity_file = params["gravity_data_file"]
        magnetic_file = params["magnetic_data_file"]

        if not Path(gravity_file).exists():
            raise RuntimeError(
                f"Gravity data file not found: {gravity_file}. "
                f"Joint inversion requires both gravity and magnetic data."
            )
        if not Path(magnetic_file).exists():
            raise RuntimeError(
                f"Magnetic data file not found: {magnetic_file}. "
                f"Joint inversion requires both gravity and magnetic data."
            )

        # Validate field parameters
        field_params = params.get("field_params", {})
        inc = field_params.get("inclination", 0)
        dec = field_params.get("declination", 0)
        strength = field_params.get("field_strength", 0)

        if not (-90 <= inc <= 90):
            raise RuntimeError(f"Invalid inclination: {inc}°. Must be between -90° and 90°.")
        if not (-180 <= dec <= 180):
            raise RuntimeError(f"Invalid declination: {dec}°. Must be between -180° and 180°.")
        if strength <= 0:
            raise RuntimeError(f"Invalid field strength: {strength} nT. Must be positive.")

        logger.info("Joint data validation passed: gravity=%s, magnetic=%s", gravity_file, magnetic_file)
        return {
            "valid": True,
            "gravity_data_file": gravity_file,
            "magnetic_data_file": magnetic_file,
            "field_params": field_params,
            "message": "Both gravity and magnetic data files validated for joint inversion.",
        }

    def _step_configure_algorithms(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Configure Tomofast-x (cross-gradient) and SimPEG (structural coupling)."""
        output_dir = params.get("output_dir", "./results/joint_inversion")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        field_params = params.get("field_params", {})

        config = {
            "tomofast_joint": {
                "iterations": params["iterations"],
                "depth_weighting_beta": params["depth_weighting_beta"],
                "density_bounds": params["density_bounds"],
                "susceptibility_bounds": params["susceptibility_bounds"],
                "method": "joint_cross_gradient",
                "cross_gradient_weight": params["cross_gradient_weight"],
                "inclination": field_params.get("inclination", DEFAULT_INCLINATION),
                "declination": field_params.get("declination", DEFAULT_DECLINATION),
                "field_strength": field_params.get("field_strength", DEFAULT_FIELD_STRENGTH),
            },
            "simpeg_coupled": {
                "iterations": params["iterations"],
                "depth_weighting_beta": params["depth_weighting_beta"],
                "density_bounds": params["density_bounds"],
                "susceptibility_bounds": params["susceptibility_bounds"],
                "method": "structural_coupling_haber_gazit",
                "coupling_weight": params["structural_coupling_weight"],
                "inducing_field": [
                    field_params.get("field_strength", DEFAULT_FIELD_STRENGTH),
                    field_params.get("inclination", DEFAULT_INCLINATION),
                    field_params.get("declination", DEFAULT_DECLINATION),
                ],
            },
        }

        config_path = os.path.join(output_dir, "joint_algorithm_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        logger.info("Joint algorithm configuration saved to: %s", config_path)
        return {"config": config, "config_path": config_path}

    def _step_run_tomofast_joint(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run Tomofast-x joint inversion with cross-gradient coupling.

        Tomofast-x supports native joint inversion via cross-gradient
        regularization that enforces structural similarity between the
        density and susceptibility models.
        """
        output_dir = params.get("output_dir", "./results/joint_inversion/tomofast_joint")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        logger.info(
            "Running Tomofast-x joint inversion: %d iterations, "
            "cross-gradient weight=%.2f",
            params["iterations"], params["cross_gradient_weight"],
        )

        # TODO: Integrate with actual Tomofast-x skill (joint mode)
        return {
            "algorithm": "tomofast-x",
            "mode": "joint_cross_gradient",
            "output_dir": output_dir,
            "iterations_run": params["iterations"],
            "cross_gradient_weight": params["cross_gradient_weight"],
            "status": "pending_integration",
            "expected_outputs": [
                os.path.join(output_dir, "model_density.vtk"),
                os.path.join(output_dir, "model_susceptibility.vtk"),
                os.path.join(output_dir, "cross_gradient_norm.csv"),
                os.path.join(output_dir, "convergence.csv"),
            ],
        }

    def _step_run_simpeg_coupled(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run SimPEG coupled inversion using Haber & Gazit structural coupling.

        Since SimPEG doesn't natively support joint inversion with cross-gradient,
        we implement the Haber & Gazit (2013) approach: run separate inversions
        with a structural coupling term that penalizes differences in model
        gradients between density and susceptibility.
        """
        output_dir = params.get("output_dir", "./results/joint_inversion/simpeg_coupled")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        logger.info(
            "Running SimPEG coupled inversion (Haber & Gazit): %d iterations, "
            "coupling weight=%.2f",
            params["iterations"], params["structural_coupling_weight"],
        )

        # TODO: Integrate with actual SimPEG skill (coupled mode)
        return {
            "algorithm": "simpeg",
            "mode": "structural_coupling_haber_gazit",
            "output_dir": output_dir,
            "iterations_run": params["iterations"],
            "structural_coupling_weight": params["structural_coupling_weight"],
            "status": "pending_integration",
            "expected_outputs": [
                os.path.join(output_dir, "model_density.vtk"),
                os.path.join(output_dir, "model_susceptibility.vtk"),
                os.path.join(output_dir, "convergence_gravity.csv"),
                os.path.join(output_dir, "convergence_magnetic.csv"),
            ],
        }

    def _step_run_individual(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run individual (uncoupled) gravity and magnetic inversions for comparison.

        These serve as a baseline to demonstrate the improvement from joint inversion.
        """
        output_dir = params.get("output_dir", "./results/joint_inversion/individual")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        logger.info(
            "Running individual inversions for comparison: %d iterations each",
            params["iterations"],
        )

        # TODO: Integrate with SimPEG/Tomofast skills for individual runs
        return {
            "algorithm": "simpeg",
            "mode": "individual_baseline",
            "output_dir": output_dir,
            "iterations_run": params["iterations"],
            "status": "pending_integration",
            "expected_outputs": [
                os.path.join(output_dir, "gravity_model_density.vtk"),
                os.path.join(output_dir, "magnetic_model_susceptibility.vtk"),
                os.path.join(output_dir, "gravity_convergence.csv"),
                os.path.join(output_dir, "magnetic_convergence.csv"),
            ],
        }

    def _step_compare_results(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Compare joint vs individual inversion results.

        Evaluates:
        - Structural similarity between density and susceptibility models
        - Data misfit for each approach
        - Cross-gradient norm (measure of structural coupling)
        - Improvement metrics: joint vs individual
        """
        output_dir = params.get("output_dir", "./results/joint_inversion/comparison")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        logger.info("Comparing joint vs individual inversion results")

        # TODO: Integrate with visualization skill
        return {
            "comparison_dir": output_dir,
            "comparisons": [
                "tomofast_joint_vs_simpeg_coupled",
                "joint_vs_individual_gravity",
                "joint_vs_individual_magnetic",
            ],
            "metrics": {
                "cross_gradient_norm_tomofast": 0.0,
                "cross_gradient_norm_simpeg": 0.0,
                "cross_gradient_norm_individual": 0.0,
                "gravity_misfit_joint": 0.0,
                "gravity_misfit_individual": 0.0,
                "magnetic_misfit_joint": 0.0,
                "magnetic_misfit_individual": 0.0,
                "structural_similarity_joint": 0.0,
                "structural_similarity_individual": 0.0,
            },
            "status": "pending_integration",
        }

    def _build_comparison(self, compare_data: Dict[str, Any]) -> ComparisonResult:
        """Build a ComparisonResult from comparison step output."""
        return ComparisonResult(
            metrics=compare_data.get("metrics", {}),
            summary=(
                "Joint inversion comparison: Tomofast-x (cross-gradient) vs "
                "SimPEG (Haber & Gazit structural coupling) vs individual inversions. "
                "Cross-gradient norm indicates structural similarity between "
                "density and susceptibility models."
            ),
            figures=[],
            raw_data=compare_data,
        )

    def _estimate_runtime(self, data_config: Dict[str, Any], iterations: int) -> str:
        """Estimate total runtime for joint inversion.

        Joint inversion involves more runs: tomofast joint + simpeg coupled
        + individual baselines.
        """
        base_time_per_iter = 2.5  # seconds
        # 4 inversion runs total (joint tomofast, coupled simpeg, individual grav, individual mag)
        algo_runs = 4
        estimated_seconds = base_time_per_iter * iterations * algo_runs
        overhead = 60  # validation, config, comparison (more complex)

        total = estimated_seconds + overhead
        if total < 60:
            return f"~{int(total)} seconds"
        elif total < 3600:
            return f"~{int(total / 60)} minutes"
        else:
            return f"~{total / 3600:.1f} hours"
