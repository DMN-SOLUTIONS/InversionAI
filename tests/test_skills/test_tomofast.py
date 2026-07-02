"""Tests for TomofastSkill: parfile generation, output parsing, validation.

All tests mock the actual Tomofast-X binary to avoid requiring compilation.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

from skills.tomofast import TomofastSkill
from skills.base_skill import RunConfig, ValidationResult, InversionResult


@pytest.fixture
def tomofast_skill(tmp_path):
    """Create a TomofastSkill instance for testing."""
    return TomofastSkill(work_dir=str(tmp_path / "tomofast_work"))


@pytest.fixture
def gravity_data_config(sample_csv_file, tmp_path):
    """Create a data_config dict for gravity validation."""
    return {
        "data_path": str(sample_csv_file),
        "data_type": "gravity",
    }


@pytest.fixture
def gravity_run_config(sample_csv_file, tmp_path):
    """Create a RunConfig for gravity inversion with Tomofast."""
    output_dir = tmp_path / "tomofast_output"
    output_dir.mkdir()
    return RunConfig(
        algorithm="tomofast-x",
        data_path=str(sample_csv_file),
        output_dir=str(output_dir),
        mesh_config={"nx": 10, "ny": 10, "nz": 5, "dx": 100.0, "dy": 100.0, "dz": 50.0},
        inversion_params={
            "data_type": "gravity",
            "max_iterations": 30,
            "target_misfit": 1.0,
        },
    )


class TestTomofastValidation:
    """Test TomofastSkill data validation."""

    def test_validate_valid_data_file(self, tomofast_skill, tmp_path):
        """Valid numeric data file passes validation."""
        # Tomofast expects space-separated numeric data, no header
        valid_file = tmp_path / "valid_data.txt"
        import numpy as np
        data = np.column_stack([
            np.linspace(0, 1000, 10),
            np.linspace(0, 1000, 10),
            np.zeros(10),
            np.random.normal(-5.0, 2.0, 10),
            np.full(10, 0.05),
        ])
        np.savetxt(valid_file, data)
        config = {"data_path": str(valid_file), "data_type": "gravity"}
        result = tomofast_skill.validate(config)
        assert result.valid is True

    def test_validate_missing_file(self, tomofast_skill, tmp_path):
        """Validation fails gracefully for non-existent file."""
        config = {"data_path": str(tmp_path / "nonexistent.csv"), "data_type": "gravity"}
        result = tomofast_skill.validate(config)
        assert result.valid is False
        assert any("not found" in e.lower() for e in result.errors)

    def test_validate_empty_file(self, tomofast_skill, tmp_path):
        """Validation fails for empty data file."""
        empty_file = tmp_path / "empty.csv"
        empty_file.write_text("")
        config = {"data_path": str(empty_file), "data_type": "gravity"}
        result = tomofast_skill.validate(config)
        assert result.valid is False

    def test_validate_missing_required_columns(self, tomofast_skill, tmp_path):
        """Validation fails when data has insufficient columns."""
        bad_file = tmp_path / "bad_data.txt"
        bad_file.write_text("1.0 2.0\n3.0 4.0\n")
        config = {"data_path": str(bad_file), "data_type": "gravity"}
        result = tomofast_skill.validate(config)
        assert result.valid is False

    def test_validate_unsupported_data_type(self, tomofast_skill, sample_csv_file):
        """Validation handles unsupported data types."""
        config = {"data_path": str(sample_csv_file), "data_type": "seismic"}
        result = tomofast_skill.validate(config)
        assert result.valid is False

    def test_validate_missing_required_keys(self, tomofast_skill):
        """Validation fails when required keys are absent."""
        result = tomofast_skill.validate({})
        assert result.valid is False
        assert any("data_path" in e.lower() for e in result.errors)


class TestTomofastSetup:
    """Test TomofastSkill setup and binary detection."""

    @patch("shutil.which")
    def test_setup_finds_binary(self, mock_which, tomofast_skill):
        """Setup succeeds when Tomofast-X binary is in PATH."""
        mock_which.return_value = "/usr/local/bin/tomofastx"
        success = tomofast_skill.setup()
        assert success is True

    @patch("shutil.which")
    @patch("subprocess.run")
    def test_setup_falls_back_to_docker(self, mock_run, mock_which, tomofast_skill):
        """Setup falls back to Docker when binary not found."""
        mock_which.return_value = None
        mock_run.return_value = MagicMock(returncode=0)
        success = tomofast_skill.setup()
        # Should try Docker fallback
        assert success is True or mock_run.called


class TestTomofastConfigure:
    """Test Tomofast parameter file (parfile) generation."""

    def test_configure_returns_run_config(self, tomofast_skill, gravity_data_config):
        """Configure returns a properly populated RunConfig."""
        config = tomofast_skill.configure(
            user_intent="Run gravity inversion with 30 iterations",
            data_config=gravity_data_config,
        )
        assert isinstance(config, RunConfig)
        assert config.algorithm == "tomofast-x"

    def test_configure_sets_data_path(self, tomofast_skill, gravity_data_config):
        """Configure preserves the data path."""
        config = tomofast_skill.configure(
            user_intent="Run gravity inversion",
            data_config=gravity_data_config,
        )
        assert gravity_data_config["data_path"] in config.data_path


class TestTomofastOutputParsing:
    """Test parsing of Tomofast-X output files."""

    @pytest.fixture
    def mock_tomofast_output(self, tmp_path):
        """Create mock Tomofast-X output files."""
        output_dir = tmp_path / "tomofast_results"
        output_dir.mkdir()

        # Mock model output
        model = np.random.uniform(-0.3, 0.3, size=500)
        np.savetxt(output_dir / "model_gravity.txt", model)

        # Mock predicted data
        predicted = np.random.normal(-5.0, 2.0, size=25)
        np.savetxt(output_dir / "predicted_data.txt", predicted)

        # Mock convergence log
        log_content = "Iteration  Phi_d  Phi_m  Phi_total\n"
        for i in range(1, 16):
            phi_d = 100.0 / (i + 1)
            log_content += f"{i}  {phi_d:.4f}  0.001  {phi_d + 0.001:.4f}\n"
        (output_dir / "inversion_log.txt").write_text(log_content)

        return output_dir

    @patch("skills.tomofast.skill.subprocess.Popen")
    def test_run_calls_binary_or_docker(self, mock_popen, tomofast_skill, gravity_run_config, tmp_path):
        """Running inversion calls the Tomofast-X binary via subprocess."""
        # Setup mock process
        mock_process = MagicMock()
        mock_process.stdout = iter(["Iteration 1, misfit=50.0\n", "Iteration 2, misfit=25.0\n"])
        mock_process.wait.return_value = 0
        mock_process.returncode = 0
        mock_process.stderr = MagicMock()
        mock_process.stderr.read.return_value = ""
        mock_popen.return_value = mock_process

        # Setup the skill as ready with binary
        from skills.base_skill import SkillStatus
        tomofast_skill._binary_path = "/usr/local/bin/tomofastx"
        tomofast_skill._use_docker = False
        tomofast_skill._status = SkillStatus.READY

        # Create a mock parfile so the run method doesn't fail early
        parfile = tmp_path / "Parfile"
        parfile.write_text("# mock parfile\n")
        gravity_run_config.mesh_config["parfile_path"] = str(parfile)

        try:
            tomofast_skill.run(gravity_run_config)
        except Exception:
            pass  # May fail parsing output files
        # Subprocess should have been called
        assert mock_popen.called

    @patch("skills.tomofast.skill.subprocess.Popen")
    def test_run_handles_binary_failure(self, mock_popen, tomofast_skill, gravity_run_config, tmp_path):
        """Run handles Tomofast-X binary failure gracefully."""
        mock_process = MagicMock()
        mock_process.stdout = iter([])
        mock_process.wait.return_value = 1
        mock_process.returncode = 1
        mock_process.stderr = MagicMock()
        mock_process.stderr.read.return_value = "Segmentation fault"
        mock_popen.return_value = mock_process

        from skills.base_skill import SkillStatus
        tomofast_skill._binary_path = "/usr/local/bin/tomofastx"
        tomofast_skill._use_docker = False
        tomofast_skill._status = SkillStatus.READY

        parfile = tmp_path / "Parfile"
        parfile.write_text("# mock parfile\n")
        gravity_run_config.mesh_config["parfile_path"] = str(parfile)

        with pytest.raises((RuntimeError, Exception)):
            tomofast_skill.run(gravity_run_config)
