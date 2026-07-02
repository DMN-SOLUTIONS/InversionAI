"""Tests for BaseSkill interface and core dataclasses.

Tests cover: ValidationResult, RunConfig, InversionResult dataclasses
and the BaseSkill abstract interface contract.
"""

from dataclasses import fields
from typing import Any

import pytest

from skills.base_skill import (
    BaseSkill,
    InversionResult,
    RunConfig,
    SkillStatus,
    ValidationResult,
)


class TestValidationResult:
    """Tests for the ValidationResult dataclass."""

    def test_valid_creation(self):
        """A valid ValidationResult defaults to valid=True."""
        result = ValidationResult(valid=True)
        assert result.valid is True
        assert result.errors == []
        assert result.warnings == []

    def test_invalid_with_errors(self):
        """An invalid ValidationResult carries descriptive error messages."""
        errors = ["Missing column 'gobs'", "Negative standard deviations"]
        result = ValidationResult(valid=False, errors=errors)
        assert result.valid is False
        assert len(result.errors) == 2
        assert "Missing column 'gobs'" in result.errors

    def test_has_expected_fields(self):
        """ValidationResult has valid, errors, warnings, suggestions fields."""
        field_names = {f.name for f in fields(ValidationResult)}
        assert "valid" in field_names
        assert "errors" in field_names
        assert "warnings" in field_names
        assert "suggestions" in field_names

    def test_str_representation_valid(self):
        """String representation shows VALID status."""
        result = ValidationResult(valid=True)
        assert "VALID" in str(result)

    def test_str_representation_invalid(self):
        """String representation shows INVALID status and errors."""
        result = ValidationResult(valid=False, errors=["bad data"])
        text = str(result)
        assert "INVALID" in text
        assert "bad data" in text


class TestRunConfig:
    """Tests for the RunConfig dataclass."""

    def test_required_fields(self):
        """RunConfig must include algorithm, data_path."""
        config = RunConfig(
            algorithm="tomofast-x",
            data_path="/data/survey.csv",
        )
        assert config.algorithm == "tomofast-x"
        assert config.data_path == "/data/survey.csv"

    def test_optional_mesh_config(self):
        """RunConfig supports optional mesh configuration."""
        config = RunConfig(
            algorithm="simpeg",
            data_path="/data/survey.csv",
            mesh_config={"nx": 20, "ny": 20, "nz": 10},
        )
        assert config.mesh_config["nx"] == 20

    def test_optional_inversion_params(self):
        """RunConfig supports algorithm-specific inversion parameters."""
        config = RunConfig(
            algorithm="tomofast-x",
            data_path="/data/survey.csv",
            inversion_params={"max_iterations": 50, "target_misfit": 1.0},
        )
        assert config.inversion_params["max_iterations"] == 50
        assert config.inversion_params["target_misfit"] == 1.0

    def test_defaults(self):
        """RunConfig has sensible defaults for optional fields."""
        config = RunConfig(algorithm="tomofast-x", data_path="/data/survey.csv")
        assert config.mesh_config == {}
        assert config.inversion_params == {}
        assert config.output_dir == "./output"

    def test_to_dict(self):
        """RunConfig can be serialized to a dict."""
        config = RunConfig(algorithm="simpeg", data_path="/data/obs.csv")
        d = config.to_dict()
        assert d["algorithm"] == "simpeg"
        assert d["data_path"] == "/data/obs.csv"


class TestInversionResult:
    """Tests for the InversionResult dataclass."""

    def test_creation(self):
        """InversionResult stores completed inversion metadata."""
        result = InversionResult(
            model_path="/output/model.txt",
            mesh_path="/output/mesh.vtk",
            misfit_history=[100.0, 50.0, 25.0, 10.0, 5.0, 2.5, 1.2],
            final_misfit=1.2,
            n_iterations=7,
            runtime_seconds=120.5,
        )
        assert result.n_iterations == 7
        assert result.final_misfit == pytest.approx(1.2)
        assert result.runtime_seconds == pytest.approx(120.5)

    def test_converged_property(self):
        """Converged property returns True when misfit stabilizes."""
        result = InversionResult(
            model_path="/output/model.txt",
            mesh_path="/output/mesh.vtk",
            misfit_history=[100.0, 50.0, 10.0, 2.0, 1.05, 1.02, 1.01],
            final_misfit=1.01,
            n_iterations=7,
        )
        assert result.converged is True

    def test_not_converged(self):
        """Converged property returns False when misfit still changing."""
        result = InversionResult(
            model_path="/output/model.txt",
            mesh_path="/output/mesh.vtk",
            misfit_history=[100.0, 50.0, 25.0],
            final_misfit=25.0,
            n_iterations=3,
        )
        assert result.converged is False

    def test_summary_string(self):
        """Summary provides human-readable description."""
        result = InversionResult(
            model_path="/output/model.txt",
            mesh_path="/output/mesh.vtk",
            misfit_history=[10.0, 5.0, 2.0, 1.0],
            final_misfit=1.0,
            n_iterations=4,
            runtime_seconds=30.0,
        )
        summary = result.summary()
        assert "4 iterations" in summary
        assert "30.0s" in summary

    def test_has_expected_fields(self):
        """InversionResult dataclass has all required fields."""
        field_names = {f.name for f in fields(InversionResult)}
        expected = {
            "model_path", "mesh_path", "misfit_history",
            "final_misfit", "n_iterations", "runtime_seconds", "metadata",
        }
        assert expected.issubset(field_names)


class TestBaseSkill:
    """Tests for the BaseSkill abstract interface."""

    def test_cannot_be_instantiated(self):
        """BaseSkill is abstract and cannot be directly instantiated."""
        with pytest.raises(TypeError):
            BaseSkill()

    def test_defines_required_abstract_methods(self):
        """BaseSkill defines validate, configure, run, setup, and get_results as abstract."""
        abstract_methods = getattr(BaseSkill, "__abstractmethods__", set())
        assert "validate" in abstract_methods
        assert "configure" in abstract_methods
        assert "run" in abstract_methods
        assert "setup" in abstract_methods
        assert "get_results" in abstract_methods

    def test_defines_required_abstract_properties(self):
        """BaseSkill requires name, version, description, supported_data_types."""
        abstract_methods = getattr(BaseSkill, "__abstractmethods__", set())
        assert "name" in abstract_methods
        assert "version" in abstract_methods
        assert "description" in abstract_methods
        assert "supported_data_types" in abstract_methods

    def test_incomplete_subclass_raises(self):
        """A subclass missing methods raises TypeError on instantiation."""

        class IncompleteSkill(BaseSkill):
            @property
            def name(self):
                return "incomplete"

        with pytest.raises(TypeError):
            IncompleteSkill()

    def test_concrete_subclass_works(self):
        """A fully implemented subclass can be instantiated."""

        class ConcreteSkill(BaseSkill):
            @property
            def name(self):
                return "test_skill"

            @property
            def version(self):
                return "1.0"

            @property
            def description(self):
                return "A test skill"

            @property
            def supported_data_types(self):
                return ["gravity"]

            def setup(self):
                return True

            def validate(self, data_config):
                return ValidationResult(valid=True)

            def configure(self, user_intent, data_config):
                return RunConfig(algorithm="test", data_path="test.csv")

            def run(self, run_config, progress_callback=None):
                return InversionResult(
                    model_path="/tmp/model.txt",
                    mesh_path="/tmp/mesh.vtk",
                    final_misfit=1.0,
                    n_iterations=10,
                    runtime_seconds=5.0,
                )

            def get_results(self):
                return InversionResult(
                    model_path="/tmp/model.txt",
                    mesh_path="/tmp/mesh.vtk",
                )

        skill = ConcreteSkill()
        assert skill.name == "test_skill"
        assert skill.status == SkillStatus.SETUP_REQUIRED

    def test_initial_status(self):
        """New skill starts with SETUP_REQUIRED status."""

        class DummySkill(BaseSkill):
            @property
            def name(self): return "dummy"
            @property
            def version(self): return "1.0"
            @property
            def description(self): return "dummy"
            @property
            def supported_data_types(self): return ["gravity"]
            def setup(self): return True
            def validate(self, data_config): return ValidationResult(valid=True)
            def configure(self, user_intent, data_config):
                return RunConfig(algorithm="dummy", data_path="x")
            def run(self, run_config, progress_callback=None):
                return InversionResult(model_path="m", mesh_path="x")
            def get_results(self):
                return InversionResult(model_path="m", mesh_path="x")

        skill = DummySkill()
        assert skill.status == SkillStatus.SETUP_REQUIRED
