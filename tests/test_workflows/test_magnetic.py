"""Tests for MagneticInversionWorkflow."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from workflows import MagneticInversionWorkflow
from workflows.base_workflow import (
    WorkflowPlan,
    WorkflowResult,
    WorkflowStatus,
    WorkflowStep,
)


@pytest.fixture
def magnetic_workflow():
    """Create a MagneticInversionWorkflow instance."""
    return MagneticInversionWorkflow()


@pytest.fixture
def magnetic_csv_file(tmp_path, sample_magnetic_data):
    """Write sample magnetic data to a CSV file."""
    csv_path = tmp_path / "magnetic_data.csv"
    header = "x,y,z,tmi,std"
    data = np.column_stack([
        sample_magnetic_data["x"],
        sample_magnetic_data["y"],
        sample_magnetic_data["z"],
        sample_magnetic_data["tmi"],
        sample_magnetic_data["std"],
    ])
    np.savetxt(csv_path, data, delimiter=",", header=header, comments="")
    return csv_path


@pytest.fixture
def data_config(tmp_path, magnetic_csv_file, sample_magnetic_data):
    """Standard data config for a magnetic inversion workflow."""
    output_dir = tmp_path / "mag_workflow_output"
    output_dir.mkdir()
    return {
        "data_file": str(magnetic_csv_file),
        "output_dir": str(output_dir),
        "data_type": "magnetic",
        "inclination": sample_magnetic_data["inclination"],
        "declination": sample_magnetic_data["declination"],
        "field_strength": sample_magnetic_data["field_strength"],
    }


class TestMagneticPlanGeneration:
    """Test magnetic workflow plan generation."""

    def test_plan_returns_workflow_plan(self, magnetic_workflow, data_config):
        """plan() returns a WorkflowPlan object."""
        plan = magnetic_workflow.plan("Run magnetic inversion", data_config)
        assert isinstance(plan, WorkflowPlan)

    def test_plan_has_steps(self, magnetic_workflow, data_config):
        """Generated plan includes ordered steps."""
        plan = magnetic_workflow.plan("Run magnetic inversion", data_config)
        assert len(plan.steps) > 0

    def test_plan_includes_validation(self, magnetic_workflow, data_config):
        """Plan includes a validation step."""
        plan = magnetic_workflow.plan("Run magnetic inversion", data_config)
        step_names = [s.name.lower() for s in plan.steps]
        assert any("valid" in name for name in step_names)

    def test_plan_includes_inversion_steps(self, magnetic_workflow, data_config):
        """Plan includes inversion steps."""
        plan = magnetic_workflow.plan(
            "Run magnetic inversion with Tomofast and SimPEG", data_config
        )
        step_skills = [s.skill.lower() for s in plan.steps]
        assert any("tomofast" in s or "simpeg" in s for s in step_skills)

    def test_plan_includes_comparison(self, magnetic_workflow, data_config):
        """Plan includes a comparison step for multiple algorithms."""
        plan = magnetic_workflow.plan(
            "Compare magnetic inversion results", data_config
        )
        step_names = [s.name.lower() for s in plan.steps]
        step_skills = [s.skill.lower() for s in plan.steps]
        has_compare = (
            any("compar" in n for n in step_names) or
            any("compar" in s for s in step_skills)
        )
        assert has_compare

    def test_plan_carries_field_parameters(self, magnetic_workflow, data_config):
        """Plan metadata includes inclination/declination."""
        plan = magnetic_workflow.plan("Run magnetic inversion", data_config)
        # Field params should be in plan metadata or step params
        plan_str = str(plan)
        has_field_params = (
            "inclination" in plan_str.lower() or
            "declination" in plan_str.lower() or
            "-60" in plan_str  # The actual inclination value
        )
        assert has_field_params

    def test_plan_missing_data_file_raises(self, magnetic_workflow):
        """Plan raises for missing data_file."""
        with pytest.raises((ValueError, KeyError)):
            magnetic_workflow.plan("Run magnetic inversion", {})


class TestMagneticExecution:
    """Test magnetic workflow execution."""

    def test_execute_returns_workflow_result(self, magnetic_workflow, data_config):
        """Execute returns a WorkflowResult."""
        plan = magnetic_workflow.plan("Run magnetic inversion", data_config)
        result = magnetic_workflow.execute(plan)
        assert isinstance(result, WorkflowResult)

    def test_execute_records_steps(self, magnetic_workflow, data_config):
        """Execute records completed steps."""
        plan = magnetic_workflow.plan("Run magnetic inversion", data_config)
        result = magnetic_workflow.execute(plan)
        assert isinstance(result.steps_completed, list)


class TestMagneticErrorHandling:
    """Test magnetic workflow error scenarios."""

    def test_cancel_workflow(self, magnetic_workflow):
        """Cancellation is properly tracked."""
        magnetic_workflow.cancel()
        assert magnetic_workflow.is_cancelled() is True

    def test_workflow_properties(self, magnetic_workflow):
        """Workflow exposes required properties."""
        assert magnetic_workflow.name != ""
        assert magnetic_workflow.description != ""
        assert len(magnetic_workflow.required_skills) > 0
        assert "magnetic" in magnetic_workflow.description.lower()

    def test_workflow_status_initial(self, magnetic_workflow):
        """Initial workflow status is PLANNING."""
        assert magnetic_workflow.get_status() == WorkflowStatus.PLANNING
