"""Tests for GravityInversionWorkflow: plan generation, step sequencing, error handling."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from workflows import GravityInversionWorkflow
from workflows.base_workflow import (
    WorkflowPlan,
    WorkflowResult,
    WorkflowStatus,
    WorkflowStep,
)


@pytest.fixture
def gravity_workflow():
    """Create a GravityInversionWorkflow instance."""
    return GravityInversionWorkflow()


@pytest.fixture
def data_config(sample_csv_file, tmp_path):
    """Standard data config for a gravity inversion workflow."""
    output_dir = tmp_path / "workflow_output"
    output_dir.mkdir()
    return {
        "data_file": str(sample_csv_file),
        "output_dir": str(output_dir),
        "data_type": "gravity",
    }


class TestPlanGeneration:
    """Test workflow plan generation."""

    def test_plan_returns_workflow_plan(self, gravity_workflow, data_config):
        """plan() returns a WorkflowPlan object."""
        plan = gravity_workflow.plan("Run gravity inversion", data_config)
        assert isinstance(plan, WorkflowPlan)

    def test_plan_has_steps(self, gravity_workflow, data_config):
        """Generated plan includes ordered steps."""
        plan = gravity_workflow.plan("Run gravity inversion", data_config)
        assert len(plan.steps) > 0
        assert all(isinstance(s, WorkflowStep) for s in plan.steps)

    def test_plan_includes_validation_step(self, gravity_workflow, data_config):
        """Plan includes a validation step."""
        plan = gravity_workflow.plan("Run gravity inversion", data_config)
        step_names = [s.name.lower() for s in plan.steps]
        assert any("valid" in name for name in step_names)

    def test_plan_includes_inversion_steps(self, gravity_workflow, data_config):
        """Plan includes inversion steps for Tomofast and SimPEG."""
        plan = gravity_workflow.plan(
            "Run gravity inversion with both Tomofast and SimPEG",
            data_config,
        )
        step_skills = [s.skill.lower() for s in plan.steps]
        # Should reference at least one inversion algorithm
        assert any("tomofast" in skill or "simpeg" in skill for skill in step_skills)

    def test_plan_validation_before_inversion(self, gravity_workflow, data_config):
        """Validation step comes before inversion steps."""
        plan = gravity_workflow.plan("Run gravity inversion", data_config)
        step_names = [s.name.lower() for s in plan.steps]
        validate_idx = next(
            (i for i, name in enumerate(step_names) if "valid" in name), None
        )
        inversion_idx = next(
            (i for i, name in enumerate(step_names)
             if "inver" in name or "run" in name),
            None,
        )
        if validate_idx is not None and inversion_idx is not None:
            assert validate_idx < inversion_idx

    def test_plan_includes_comparison_step(self, gravity_workflow, data_config):
        """Plan includes a comparison step when multiple algorithms are used."""
        plan = gravity_workflow.plan(
            "Run gravity inversion comparing Tomofast and SimPEG",
            data_config,
        )
        step_names = [s.name.lower() for s in plan.steps]
        step_skills = [s.skill.lower() for s in plan.steps]
        has_compare = (
            any("compar" in name for name in step_names) or
            any("compar" in skill for skill in step_skills)
        )
        assert has_compare

    def test_plan_has_estimated_runtime(self, gravity_workflow, data_config):
        """Plan includes estimated runtime."""
        plan = gravity_workflow.plan("Run gravity inversion", data_config)
        assert plan.estimated_runtime != ""

    def test_plan_missing_data_file_raises(self, gravity_workflow):
        """Plan raises ValueError for missing data_file."""
        with pytest.raises((ValueError, KeyError)):
            gravity_workflow.plan("Run gravity inversion", {})


class TestStepSequencing:
    """Test workflow execution sequencing."""

    def test_execute_returns_workflow_result(self, gravity_workflow, data_config):
        """Workflow execute returns a WorkflowResult object."""
        plan = gravity_workflow.plan("Run gravity inversion", data_config)
        result = gravity_workflow.execute(plan)
        assert isinstance(result, WorkflowResult)

    def test_execute_records_completed_steps(self, gravity_workflow, data_config):
        """Execute records which steps completed."""
        plan = gravity_workflow.plan("Run gravity inversion", data_config)
        result = gravity_workflow.execute(plan)
        # At minimum, validation step should complete (or fail gracefully)
        assert isinstance(result.steps_completed, list)

    def test_execute_with_progress_callback(self, gravity_workflow, data_config):
        """Execute accepts a progress callback."""
        plan = gravity_workflow.plan("Run gravity inversion", data_config)
        progress_calls = []

        def callback(step_name, progress):
            progress_calls.append((step_name, progress))

        result = gravity_workflow.execute(plan, progress_callback=callback)
        assert isinstance(result, WorkflowResult)


class TestErrorHandling:
    """Test workflow error handling and recovery."""

    def test_cancel_sets_cancelled_flag(self, gravity_workflow):
        """Calling cancel() sets the cancelled flag."""
        gravity_workflow.cancel()
        assert gravity_workflow.is_cancelled() is True

    def test_workflow_properties(self, gravity_workflow):
        """Workflow exposes name, description, required_skills."""
        assert gravity_workflow.name != ""
        assert gravity_workflow.description != ""
        assert len(gravity_workflow.required_skills) > 0
        assert "tomofast" in gravity_workflow.required_skills or "simpeg" in gravity_workflow.required_skills

    def test_workflow_status_initial(self, gravity_workflow):
        """Initial workflow status is PLANNING."""
        assert gravity_workflow.get_status() == WorkflowStatus.PLANNING
