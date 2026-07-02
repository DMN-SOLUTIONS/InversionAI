"""Tests for CompareSkill: normalization, metrics calculation, visualization."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from skills.compare import CompareSkill
from skills.base_skill import InversionResult, RunConfig, ValidationResult


@pytest.fixture
def compare_skill(tmp_path):
    """Create a CompareSkill instance for testing."""
    return CompareSkill(work_dir=str(tmp_path / "compare_work"))


@pytest.fixture
def result_pair(tmp_path):
    """Create two InversionResult objects for comparison."""
    np.random.seed(42)

    # Result A - Tomofast
    model_a = np.random.uniform(-0.4, 0.4, size=500)
    path_a = tmp_path / "model_a.txt"
    np.savetxt(path_a, model_a)
    mesh_a = tmp_path / "mesh_a.vtk"
    mesh_a.write_text("# mock mesh")

    result_a = InversionResult(
        model_path=str(path_a),
        mesh_path=str(mesh_a),
        misfit_history=[100.0, 50.0, 20.0, 5.0, 1.5, 1.02],
        final_misfit=1.02,
        n_iterations=6,
        runtime_seconds=42.0,
        metadata={"algorithm": "tomofast-x"},
    )

    # Result B - SimPEG
    model_b = model_a + np.random.normal(0, 0.05, size=500)
    path_b = tmp_path / "model_b.txt"
    np.savetxt(path_b, model_b)
    mesh_b = tmp_path / "mesh_b.vtk"
    mesh_b.write_text("# mock mesh")

    result_b = InversionResult(
        model_path=str(path_b),
        mesh_path=str(mesh_b),
        misfit_history=[90.0, 40.0, 15.0, 5.0, 2.0, 1.05],
        final_misfit=1.05,
        n_iterations=6,
        runtime_seconds=120.0,
        metadata={"algorithm": "simpeg"},
    )

    return result_a, result_b


class TestCompareSkillValidation:
    """Test CompareSkill validation of inputs."""

    def test_validate_with_two_results(self, compare_skill, result_pair):
        """Validation passes when two valid results are provided."""
        result_a, result_b = result_pair
        config = {
            "result_a": result_a,
            "result_b": result_b,
            "data_type": "gravity",
        }
        validation = compare_skill.validate(config)
        assert validation.valid is True

    def test_validate_missing_model_file(self, compare_skill, tmp_path):
        """Validation fails when model file doesn't exist."""
        result_a = InversionResult(
            model_path=str(tmp_path / "nonexistent.txt"),
            mesh_path=str(tmp_path / "mesh.vtk"),
        )
        result_b = InversionResult(
            model_path=str(tmp_path / "also_missing.txt"),
            mesh_path=str(tmp_path / "mesh.vtk"),
        )
        config = {"result_a": result_a, "result_b": result_b, "data_type": "gravity"}
        validation = compare_skill.validate(config)
        assert validation.valid is False


class TestNormalization:
    """Test model normalization and interpolation for fair comparison."""

    def test_interpolate_to_common_grid_returns_arrays(self, compare_skill, result_pair):
        """Interpolation onto common grid returns resampled model arrays."""
        from skills.compare.normalize import interpolate_to_common_grid
        result_a, result_b = result_pair
        model_a, model_b, grid_info = interpolate_to_common_grid(
            model_path_a=result_a.model_path,
            model_path_b=result_b.model_path,
        )
        assert model_a is not None
        assert model_b is not None
        assert len(model_a) == len(model_b)

    def test_interpolation_preserves_scale(self, compare_skill, result_pair):
        """Interpolated models maintain similar value ranges."""
        from skills.compare.normalize import interpolate_to_common_grid
        result_a, result_b = result_pair
        model_a, model_b, _ = interpolate_to_common_grid(
            model_path_a=result_a.model_path,
            model_path_b=result_b.model_path,
        )
        # Models should be in a similar range since they were generated similarly
        assert abs(model_a.mean() - model_b.mean()) < 1.0


class TestMetricsCalculation:
    """Test comparison metrics between two inversion results."""

    def test_correlation_identical_models(self):
        """Correlation is 1.0 for identical models."""
        from skills.compare.metrics import model_correlation
        model = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        corr = model_correlation(model, model)
        assert corr == pytest.approx(1.0)

    def test_correlation_anticorrelated(self):
        """Correlation is -1.0 for perfectly anti-correlated models."""
        from skills.compare.metrics import model_correlation
        model_a = np.array([1.0, 2.0, 3.0, 4.0])
        model_b = np.array([4.0, 3.0, 2.0, 1.0])
        corr = model_correlation(model_a, model_b)
        assert corr == pytest.approx(-1.0)

    def test_structural_similarity(self):
        """Structural similarity metric returns value in [0, 1]."""
        from skills.compare.metrics import structural_similarity_index
        np.random.seed(7)
        model_a = np.random.uniform(0, 1, size=100)
        model_b = model_a + np.random.normal(0, 0.1, size=100)
        ssim = structural_similarity_index(model_a, model_b)
        assert 0.0 <= ssim <= 1.0

    def test_convergence_rate(self):
        """Convergence rate measures how quickly misfit decreases."""
        from skills.compare.metrics import convergence_rate
        history = [100.0, 50.0, 25.0, 10.0, 5.0, 2.0, 1.0]
        rate = convergence_rate(history)
        assert rate > 0

    def test_runtime_comparison(self):
        """Runtime comparison produces a meaningful result dict."""
        from skills.compare.metrics import runtime_comparison
        result = runtime_comparison(42.0, 120.0)
        assert isinstance(result, dict)
        assert result["runtime_a"] == 42.0
        assert result["runtime_b"] == 120.0
        assert "speedup" in result or "ratio" in result


class TestCompareExecution:
    """Test full comparison execution."""

    def test_compare_produces_metrics(self, compare_skill, result_pair):
        """Full comparison produces a metrics dictionary in metadata."""
        result_a, result_b = result_pair
        config = {
            "result_a": result_a,
            "result_b": result_b,
            "data_type": "gravity",
        }
        # Configure
        run_config = compare_skill.configure(
            user_intent="Compare Tomofast and SimPEG results",
            data_config=config,
        )
        assert isinstance(run_config, RunConfig)
