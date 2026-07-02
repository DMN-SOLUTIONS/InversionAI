"""
Base workflow module for InversionAI.

Provides abstract base class and data structures for building
pre-built recipes that chain skills together for common geophysical
inversion use cases.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """Status of a workflow execution."""

    PLANNING = "planning"
    VALIDATING = "validating"
    RUNNING = "running"
    COMPARING = "comparing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class WorkflowStep:
    """A single step in a workflow execution plan.

    Attributes:
        name: Human-readable name of the step.
        skill: The skill module to invoke (e.g., 'tomofast', 'simpeg').
        action: The specific action within the skill.
        params: Parameters to pass to the skill action.
        status: Current status of this step.
    """

    name: str
    skill: str
    action: str
    params: Dict[str, Any] = field(default_factory=dict)
    status: WorkflowStatus = WorkflowStatus.PLANNING


@dataclass
class WorkflowPlan:
    """An execution plan comprising ordered workflow steps.

    Attributes:
        steps: Ordered list of steps to execute.
        estimated_runtime: Human-readable estimated runtime string.
        metadata: Additional planning metadata.
    """

    steps: List[WorkflowStep]
    estimated_runtime: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComparisonResult:
    """Result of comparing outputs from multiple inversion algorithms.

    Attributes:
        metrics: Dictionary of comparison metrics (e.g., RMS, correlation).
        summary: Human-readable summary of comparison.
        figures: Paths to generated comparison figures.
        raw_data: Raw comparison data for further analysis.
    """

    metrics: Dict[str, float] = field(default_factory=dict)
    summary: str = ""
    figures: List[str] = field(default_factory=list)
    raw_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowResult:
    """Result of a completed workflow execution.

    Attributes:
        steps_completed: List of step names that completed successfully.
        results: Dictionary mapping step names to their output data.
        comparison: Optional comparison result if multiple algorithms were run.
        elapsed_time: Total elapsed time in seconds.
        intermediate_files: Paths to saved intermediate results.
    """

    steps_completed: List[str] = field(default_factory=list)
    results: Dict[str, Any] = field(default_factory=dict)
    comparison: Optional[ComparisonResult] = None
    elapsed_time: float = 0.0
    intermediate_files: List[str] = field(default_factory=list)


class WorkflowCancelledError(Exception):
    """Raised when a workflow is cancelled during execution."""

    pass


class BaseWorkflow(ABC):
    """Abstract base class for all InversionAI workflows.

    A workflow is a pre-built recipe that chains multiple skills together
    to accomplish a common geophysical inversion use case.

    Example:
        >>> workflow = GravityInversionWorkflow()
        >>> plan = workflow.plan("Run gravity inversion", data_config)
        >>> result = workflow.execute(plan, progress_callback=my_callback)
    """

    def __init__(self) -> None:
        self._status: WorkflowStatus = WorkflowStatus.PLANNING
        self._cancelled: bool = False
        self._start_time: Optional[float] = None
        self._current_step: Optional[str] = None
        self._intermediate_results: Dict[str, Any] = {}

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the workflow."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what this workflow does."""
        ...

    @property
    @abstractmethod
    def required_skills(self) -> List[str]:
        """List of skill module names required by this workflow."""
        ...

    @property
    @abstractmethod
    def steps(self) -> List[str]:
        """Ordered list of step names in this workflow."""
        ...

    @abstractmethod
    def plan(self, user_intent: str, data_config: Dict[str, Any]) -> WorkflowPlan:
        """Create an execution plan based on user intent and data configuration.

        Args:
            user_intent: Natural language description of what the user wants.
            data_config: Dictionary with data file paths and parameters.

        Returns:
            A WorkflowPlan with ordered steps and estimated runtime.

        Raises:
            ValueError: If data_config is missing required fields.
        """
        ...

    @abstractmethod
    def execute(
        self,
        plan: WorkflowPlan,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> WorkflowResult:
        """Execute a workflow plan.

        Args:
            plan: The WorkflowPlan to execute.
            progress_callback: Optional callback(step_name, progress_fraction).

        Returns:
            WorkflowResult with completed steps and results.

        Raises:
            RuntimeError: If a critical step fails.
            WorkflowCancelledError: If the workflow was cancelled.
        """
        ...

    def get_status(self) -> WorkflowStatus:
        """Get the current status of the workflow."""
        return self._status

    def cancel(self) -> None:
        """Request cancellation of the running workflow."""
        logger.info("Cancellation requested for workflow: %s", self.name)
        self._cancelled = True

    def is_cancelled(self) -> bool:
        """Check if cancellation has been requested."""
        return self._cancelled

    def reset(self, preserve_cancellation: bool = False) -> None:
        """Reset the workflow state for a fresh execution.

        Args:
            preserve_cancellation: If True, do not clear the cancelled flag.
        """
        self._status = WorkflowStatus.PLANNING
        if not preserve_cancellation:
            self._cancelled = False
        self._start_time = None
        self._current_step = None
        self._intermediate_results = {}

    def _set_status(self, status: WorkflowStatus) -> None:
        """Update workflow status with logging."""
        logger.info(
            "Workflow '%s' status: %s -> %s",
            self.name, self._status.value, status.value,
        )
        self._status = status

    def _save_intermediate(self, step_name: str, data: Any) -> None:
        """Save intermediate results for a completed step."""
        self._intermediate_results[step_name] = {
            "data": data,
            "timestamp": time.time(),
        }
        logger.debug("Saved intermediate result for step: %s", step_name)

    def _check_cancellation(self, step_name: str) -> None:
        """Check if cancellation was requested and raise if so.

        Raises:
            WorkflowCancelledError: If cancellation was requested.
        """
        if self._cancelled:
            logger.warning(
                "Workflow '%s' cancelled at step: %s", self.name, step_name
            )
            self._set_status(WorkflowStatus.FAILED)
            raise WorkflowCancelledError(
                f"Workflow '{self.name}' was cancelled during step: {step_name}"
            )
