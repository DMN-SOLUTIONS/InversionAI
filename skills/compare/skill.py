"""
Comparison skill for evaluating inversion results from different algorithms.

Takes two InversionResult objects and produces quantitative comparison
metrics and side-by-side visualizations.
"""

import logging
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
from .metrics import (
    convergence_rate,
    depth_weighted_rms,
    misfit_reduction_rate,
    model_correlation,
    runtime_comparison,
    structural_similarity_index,
)
from .normalize import interpolate_to_common_grid
from .visualize import generate_comparison_plots

logger = logging.getLogger(__name__)


class CompareSkill(BaseSkill):
    """Skill for comparing inversion results from different algorithms.

    Takes two InversionResult objects (e.g., from TomofastSkill and SimpegSkill)
    and produces:
    - Quantitative metrics (correlation, SSIM, RMS difference)
    - Convergence analysis
    - Side-by-side visualizations
    - Difference maps
    """

    def __init__(self, work_dir: str = "./compare_work") -> None:
        """Initialize CompareSkill.

        Args:
            work_dir: Working directory for output files and plots.
        """
        super().__init__()
        self._work_dir = Path(work_dir)
        self._result_a: Optional[InversionResult] = None
        self._result_b: Optional[InversionResult] = None
        self._comparison_metrics: Optional[dict[str, Any]] = None

    @property
    def name(self) -> str:
        return "compare"

    @property
    def version(self) -> str:
        return "1.0"

    @property
    def description(self) -> str:
        return (
            "Compare inversion results from different algorithms. "
            "Computes correlation, structural similarity, convergence rates, "
            "and generates side-by-side visualization plots."
        )

    @property
    def supported_data_types(self) -> list[str]:
        return ["gravity", "magnetic"]

    def setup(self) -> bool:
        """Verify required libraries are available.

        Returns:
            True if numpy, scipy, and plotly are available.
        """
        self._logger.info("Setting up Compare skill...")
        missing = []

        try:
            import numpy  # noqa: F401
        except ImportError:
            missing.append("numpy")

        try:
            import scipy  # noqa: F401
        except ImportError:
            missing.append("scipy")

        try:
            import plotly  # noqa: F401
        except ImportError:
            missing.append("plotly")

        if missing:
            self._logger.error(
                f"Missing dependencies: {missing}. "
                f"Install with: pip install {' '.join(missing)}"
            )
            self._status = SkillStatus.SETUP_REQUIRED
            return False

        self._status = SkillStatus.READY
        self._logger.info("Compare skill ready")
        return True

    def set_results(
        self,
        result_a: InversionResult,
        result_b: InversionResult,
    ) -> None:
        """Set the two inversion results to compare.

        Args:
            result_a: First inversion result (e.g., Tomofast-x).
            result_b: Second inversion result (e.g., SimPEG).
        """
        self._result_a = result_a
        self._result_b = result_b
        self._logger.info(
            f"Comparison set: '{result_a.metadata.get('algorithm', 'A')}' vs "
            f"'{result_b.metadata.get('algorithm', 'B')}'"
        )

    def validate(self, data_config: dict[str, Any]) -> ValidationResult:
        """Validate that comparison inputs are available and compatible.

        Args:
            data_config: Expected keys:
                - 'result_a': InversionResult or path to model file
                - 'result_b': InversionResult or path to model file
                - Optional: 'mesh_a', 'mesh_b' for different meshes

        Returns:
            ValidationResult
        """
        errors: list[str] = []
        warnings: list[str] = []
        suggestions: list[str] = []

        # Check that results are provided
        result_a = data_config.get("result_a", self._result_a)
        result_b = data_config.get("result_b", self._result_b)

        if result_a is None:
            errors.append("Missing 'result_a': first inversion result not provided")
        if result_b is None:
            errors.append("Missing 'result_b': second inversion result not provided")

        if errors:
            return ValidationResult(valid=False, errors=errors, warnings=warnings, suggestions=suggestions)

        # Validate model files exist
        if isinstance(result_a, InversionResult):
            if not result_a.model_path or not Path(result_a.model_path).exists():
                errors.append(f"Model file A not found: {result_a.model_path}")
        if isinstance(result_b, InversionResult):
            if not result_b.model_path or not Path(result_b.model_path).exists():
                errors.append(f"Model file B not found: {result_b.model_path}")

        # Check for mesh compatibility
        if isinstance(result_a, InversionResult) and isinstance(result_b, InversionResult):
            mesh_a = result_a.metadata.get("n_cells")
            mesh_b = result_b.metadata.get("n_cells")
            if mesh_a and mesh_b and mesh_a != mesh_b:
                warnings.append(
                    f"Different mesh sizes ({mesh_a} vs {mesh_b} cells). "
                    "Models will be interpolated to common grid."
                )
                suggestions.append(
                    "For best comparison, use the same mesh for both inversions."
                )

        valid = len(errors) == 0
        return ValidationResult(valid=valid, errors=errors, warnings=warnings, suggestions=suggestions)

    def configure(self, user_intent: str, data_config: dict[str, Any]) -> RunConfig:
        """Configure comparison parameters.

        Args:
            user_intent: Description of desired comparison (e.g.,
                "Compare models at depth slices 100m, 200m, 500m").
            data_config: Contains result objects and output preferences.

        Returns:
            RunConfig for the comparison.
        """
        self._logger.info(f"Configuring comparison: '{user_intent}'")

        # Parse depth slices from intent
        import re
        depth_slices = []
        depth_matches = re.findall(r"(\d+)\s*m", user_intent.lower())
        if depth_matches:
            depth_slices = [float(d) for d in depth_matches]

        if not depth_slices:
            depth_slices = [50.0, 100.0, 200.0, 500.0]  # Default depths

        # Set results if provided in data_config
        if "result_a" in data_config:
            self._result_a = data_config["result_a"]
        if "result_b" in data_config:
            self._result_b = data_config["result_b"]

        output_dir = data_config.get("output_dir", str(self._work_dir / "output"))
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        return RunConfig(
            algorithm="compare",
            data_path="",  # Not applicable for comparison
            mesh_config={},
            inversion_params={
                "depth_slices": depth_slices,
                "generate_plots": data_config.get("generate_plots", True),
                "plot_format": data_config.get("plot_format", "html"),
                "label_a": data_config.get("label_a", self._result_a.metadata.get("algorithm", "Model A") if self._result_a else "Model A"),
                "label_b": data_config.get("label_b", self._result_b.metadata.get("algorithm", "Model B") if self._result_b else "Model B"),
            },
            output_dir=output_dir,
        )

    def run(
        self,
        config: RunConfig,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> InversionResult:
        """Execute the comparison analysis.

        Args:
            config: RunConfig from configure().
            progress_callback: Optional callback for progress.

        Returns:
            InversionResult containing comparison outputs.

        Raises:
            RuntimeError: If comparison fails.
        """
        self._status = SkillStatus.RUNNING
        start_time = time.time()

        if self._result_a is None or self._result_b is None:
            self._status = SkillStatus.FAILED
            raise RuntimeError("Both result_a and result_b must be set before running comparison")

        try:
            if progress_callback:
                progress_callback(1, 5, 0.0)

            # Step 1: Load and normalize models to common grid
            self._logger.info("Interpolating models to common grid...")
            model_a, model_b, common_grid = interpolate_to_common_grid(
                model_path_a=self._result_a.model_path,
                model_path_b=self._result_b.model_path,
                mesh_path_a=self._result_a.mesh_path,
                mesh_path_b=self._result_b.mesh_path,
            )

            if progress_callback:
                progress_callback(2, 5, 0.0)

            # Step 2: Compute metrics
            self._logger.info("Computing comparison metrics...")
            metrics = self._compute_all_metrics(model_a, model_b, common_grid)

            if progress_callback:
                progress_callback(3, 5, 0.0)

            # Step 3: Convergence analysis
            metrics["convergence"] = {
                "rate_a": convergence_rate(self._result_a.misfit_history),
                "rate_b": convergence_rate(self._result_b.misfit_history),
                "misfit_reduction_a": misfit_reduction_rate(self._result_a.misfit_history),
                "misfit_reduction_b": misfit_reduction_rate(self._result_b.misfit_history),
            }
            metrics["runtime"] = runtime_comparison(
                self._result_a.runtime_seconds,
                self._result_b.runtime_seconds,
            )

            if progress_callback:
                progress_callback(4, 5, 0.0)

            # Step 4: Generate visualizations
            output_dir = Path(config.output_dir)
            plot_paths: list[str] = []

            if config.inversion_params.get("generate_plots", True):
                self._logger.info("Generating comparison plots...")
                plot_paths = generate_comparison_plots(
                    model_a=model_a,
                    model_b=model_b,
                    common_grid=common_grid,
                    misfit_a=self._result_a.misfit_history,
                    misfit_b=self._result_b.misfit_history,
                    label_a=config.inversion_params.get("label_a", "Model A"),
                    label_b=config.inversion_params.get("label_b", "Model B"),
                    depth_slices=config.inversion_params.get("depth_slices", []),
                    output_dir=str(output_dir),
                    plot_format=config.inversion_params.get("plot_format", "html"),
                )

            if progress_callback:
                progress_callback(5, 5, 0.0)

            runtime = time.time() - start_time

            # Save metrics report
            report_path = str(output_dir / "comparison_report.json")
            self._save_metrics_report(metrics, report_path)

            self._comparison_metrics = metrics

            result = InversionResult(
                model_path=report_path,
                mesh_path="",
                misfit_history=[],
                final_misfit=0.0,
                n_iterations=0,
                runtime_seconds=runtime,
                metadata={
                    "algorithm": "compare",
                    "metrics": metrics,
                    "plot_paths": plot_paths,
                    "label_a": config.inversion_params.get("label_a", "Model A"),
                    "label_b": config.inversion_params.get("label_b", "Model B"),
                },
            )

            self._last_result = result
            self._status = SkillStatus.COMPLETED
            self._logger.info(f"Comparison completed in {runtime:.1f}s")
            return result

        except Exception as e:
            self._status = SkillStatus.FAILED
            raise RuntimeError(f"Comparison failed: {e}")

    def _compute_all_metrics(
        self,
        model_a: "np.ndarray",
        model_b: "np.ndarray",
        common_grid: dict[str, Any],
    ) -> dict[str, Any]:
        """Compute all comparison metrics.

        Args:
            model_a: First model values on common grid.
            model_b: Second model values on common grid.
            common_grid: Grid specification dictionary.

        Returns:
            Dictionary of all computed metrics.
        """
        metrics: dict[str, Any] = {}

        metrics["model_correlation"] = model_correlation(model_a, model_b)
        metrics["structural_similarity"] = structural_similarity_index(model_a, model_b)
        metrics["depth_weighted_rms"] = depth_weighted_rms(
            model_a, model_b, common_grid
        )

        # Basic statistics
        metrics["statistics"] = {
            "model_a": {
                "min": float(np.min(model_a)),
                "max": float(np.max(model_a)),
                "mean": float(np.mean(model_a)),
                "std": float(np.std(model_a)),
            },
            "model_b": {
                "min": float(np.min(model_b)),
                "max": float(np.max(model_b)),
                "mean": float(np.mean(model_b)),
                "std": float(np.std(model_b)),
            },
            "difference": {
                "max_abs_diff": float(np.max(np.abs(model_a - model_b))),
                "mean_abs_diff": float(np.mean(np.abs(model_a - model_b))),
                "rms_diff": float(np.sqrt(np.mean((model_a - model_b) ** 2))),
            },
        }

        return metrics

    def _save_metrics_report(self, metrics: dict[str, Any], output_path: str) -> None:
        """Save metrics to a JSON report file.

        Args:
            metrics: Computed metrics dictionary.
            output_path: Path to write JSON file.
        """
        import json

        # Convert numpy types to Python native for JSON serialization
        def convert(obj: Any) -> Any:
            if isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(i) for i in obj]
            return obj

        with open(output_path, "w") as f:
            json.dump(convert(metrics), f, indent=2)

        self._logger.info(f"Metrics report saved to: {output_path}")

    def get_results(self) -> InversionResult:
        """Retrieve comparison results.

        Returns:
            InversionResult containing comparison report and plots.

        Raises:
            RuntimeError: If no comparison has been run.
        """
        if self._last_result is None:
            raise RuntimeError("No comparison results available. Run compare first.")
        return self._last_result

    def get_metrics(self) -> dict[str, Any]:
        """Get the comparison metrics dictionary directly.

        Returns:
            Dictionary of all computed metrics.

        Raises:
            RuntimeError: If no comparison has been run.
        """
        if self._comparison_metrics is None:
            raise RuntimeError("No metrics available. Run compare first.")
        return self._comparison_metrics
