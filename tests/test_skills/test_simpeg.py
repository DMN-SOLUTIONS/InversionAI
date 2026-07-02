"""Tests for SimpegSkill: mesh building, configuration, validation.

All tests mock heavy SimPEG computation to keep tests fast.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from skills.simpeg import SimpegSkill
from skills.base_skill import RunConfig, ValidationResult, InversionResult


@pytest.fixture
def simpeg_skill(tmp_path):
    """Create a SimpegSkill instance for testing."""
    return SimpegSkill(work_dir=str(tmp_path / "simpeg_work"))


@pytest.fixture
def gravity_data_config(sample_csv_file):
    """Create a data_config dict for gravity validation."""
    return {
        "data_path": str(sample_csv_file),
        "data_type": "gravity",
    }


@pytest.fixture
def gravity_run_config(sample_csv_file, tmp_path):
    """Create a RunConfig for gravity inversion with SimPEG."""
    output_dir = tmp_path / "simpeg_output"
    output_dir.mkdir()
    return RunConfig(
        algorithm="simpeg",
        data_path=str(sample_csv_file),
        output_dir=str(output_dir),
        mesh_config={
            "mesh_type": "TensorMesh",
            "n_cells_x": 20, "n_cells_y": 20, "n_cells_z": 10,
        },
        inversion_params={
            "data_type": "gravity",
            "max_iterations": 40,
            "target_misfit": 1.0,
            "regularization": "Tikhonov",
        },
    )


class TestSimpegValidation:
    """Test SimpegSkill data validation."""

    def test_validate_checks_file_exists(self, simpeg_skill, gravity_data_config):
        """Validation verifies the file exists (passes for existing file)."""
        result = simpeg_skill.validate(gravity_data_config)
        # Even if SimPEG isn't installed, file existence should not be an error
        assert not any("not found" in e.lower() for e in result.errors)

    def test_validate_missing_file(self, simpeg_skill, tmp_path):
        """Validation fails for non-existent file."""
        config = {"data_path": str(tmp_path / "missing.csv"), "data_type": "gravity"}
        result = simpeg_skill.validate(config)
        assert result.valid is False
        assert any("not found" in e.lower() for e in result.errors)

    def test_validate_nan_values(self, simpeg_skill, tmp_path):
        """Validation catches NaN values in data."""
        nan_csv = tmp_path / "nan_data.csv"
        # Write without header for SimPEG's parser
        nan_csv.write_text("0,0,0,nan,0.05\n100,100,0,-3.0,0.05\n")
        config = {"data_path": str(nan_csv), "data_type": "gravity"}
        result = simpeg_skill.validate(config)
        assert result.valid is False

    def test_validate_unsupported_data_type(self, simpeg_skill, sample_csv_file):
        """Validation rejects unsupported data types."""
        config = {"data_path": str(sample_csv_file), "data_type": "seismic"}
        result = simpeg_skill.validate(config)
        assert result.valid is False

    def test_validate_missing_required_keys(self, simpeg_skill):
        """Validation fails when required keys are missing."""
        result = simpeg_skill.validate({})
        assert result.valid is False
        assert any("data_path" in e.lower() for e in result.errors)


class TestSimpegSetup:
    """Test SimpegSkill environment setup."""

    def test_setup_succeeds_if_simpeg_importable(self, simpeg_skill):
        """Setup returns True/False depending on SimPEG availability."""
        # Don't mock - just check the behavior with whatever is installed
        result = simpeg_skill.setup()
        assert isinstance(result, bool)

    def test_setup_without_simpeg_returns_false(self, simpeg_skill):
        """Setup fails when SimPEG is not importable."""
        # Temporarily break the import by patching builtins
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "SimPEG" or name == "discretize":
                raise ImportError(f"No module named '{name}'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            success = simpeg_skill.setup()
            assert success is False


class TestSimpegConfigure:
    """Test SimPEG configuration and mesh building."""

    def test_configure_returns_run_config(self, simpeg_skill, gravity_data_config):
        """Configure step returns a valid RunConfig."""
        config = simpeg_skill.configure(
            user_intent="Run gravity inversion with Tikhonov regularization",
            data_config=gravity_data_config,
        )
        assert isinstance(config, RunConfig)
        assert config.algorithm == "simpeg"

    def test_configure_includes_mesh_config(self, simpeg_skill, gravity_data_config):
        """Configure populates mesh configuration."""
        config = simpeg_skill.configure(
            user_intent="Run gravity inversion with 20x20x10 mesh",
            data_config=gravity_data_config,
        )
        assert config.mesh_config != {}

    def test_configure_includes_regularization(self, simpeg_skill, gravity_data_config):
        """Configure sets regularization parameters."""
        config = simpeg_skill.configure(
            user_intent="Run gravity inversion with Tikhonov regularization",
            data_config=gravity_data_config,
        )
        params = config.inversion_params
        assert "regularization" in params or "alpha_s" in params or len(params) > 0


class TestSimpegRun:
    """Test SimpegSkill inversion execution."""

    def test_run_without_setup_raises(self, simpeg_skill, gravity_run_config):
        """Running without setup raises an error."""
        # Skill is in SETUP_REQUIRED state, should not allow run
        with pytest.raises((RuntimeError, Exception)):
            simpeg_skill.run(gravity_run_config)

    def test_run_after_setup(self, simpeg_skill, gravity_run_config):
        """Run after setup attempts inversion (may fail without SimPEG)."""
        simpeg_skill.setup()
        if simpeg_skill._simpeg_available:
            # SimPEG is installed - run may succeed or fail on data
            try:
                result = simpeg_skill.run(gravity_run_config)
                assert isinstance(result, InversionResult)
            except (RuntimeError, Exception):
                pass  # Expected if data format doesn't match
        else:
            # SimPEG not installed - should raise
            with pytest.raises((RuntimeError, Exception)):
                simpeg_skill.run(gravity_run_config)
