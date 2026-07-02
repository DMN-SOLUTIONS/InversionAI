"""
InversionAI Workflows Module.

Pre-built recipes that chain skills together for common geophysical
inversion use cases. Each workflow orchestrates data validation,
algorithm configuration, inversion execution, and result comparison.

Available Workflows:
    - GravityInversionWorkflow: Gravity density inversion (Tomofast-x + SimPEG)
    - MagneticInversionWorkflow: Magnetic susceptibility inversion (Tomofast-x + SimPEG)
    - JointInversionWorkflow: Joint gravity+magnetic with cross-gradient coupling

Usage:
    >>> from workflows import list_workflows, GravityInversionWorkflow
    >>> workflows = list_workflows()
    >>> workflow = GravityInversionWorkflow()
    >>> plan = workflow.plan("Run gravity inversion", {"data_file": "obs.csv"})
    >>> result = workflow.execute(plan)
"""

from .base_workflow import (
    BaseWorkflow,
    ComparisonResult,
    WorkflowCancelledError,
    WorkflowPlan,
    WorkflowResult,
    WorkflowStatus,
    WorkflowStep,
)
from .gravity_inversion import GravityInversionWorkflow
from .joint_inversion import JointInversionWorkflow
from .magnetic_inversion import MagneticInversionWorkflow

__all__ = [
    # Workflow classes
    "GravityInversionWorkflow",
    "MagneticInversionWorkflow",
    "JointInversionWorkflow",
    # Base classes and data structures
    "BaseWorkflow",
    "WorkflowStep",
    "WorkflowPlan",
    "WorkflowResult",
    "WorkflowStatus",
    "ComparisonResult",
    "WorkflowCancelledError",
    # Utility functions
    "list_workflows",
]

# Registry of available workflows
_WORKFLOW_REGISTRY: dict[str, type[BaseWorkflow]] = {
    "gravity": GravityInversionWorkflow,
    "magnetic": MagneticInversionWorkflow,
    "joint": JointInversionWorkflow,
}


def list_workflows() -> list[dict[str, str]]:
    """List all available workflows with their descriptions.

    Returns:
        List of dictionaries with workflow metadata:
            - key: Short identifier for the workflow.
            - name: Human-readable name.
            - description: What the workflow does.
            - required_skills: Skills needed to run the workflow.
            - steps: Ordered list of step names.

    Example:
        >>> from workflows import list_workflows
        >>> for wf in list_workflows():
        ...     print(f"{wf['key']}: {wf['name']}")
        gravity: Gravity Inversion
        magnetic: Magnetic Inversion
        joint: Joint Gravity-Magnetic Inversion
    """
    result = []
    for key, workflow_cls in _WORKFLOW_REGISTRY.items():
        instance = workflow_cls()
        result.append({
            "key": key,
            "name": instance.name,
            "description": instance.description,
            "required_skills": instance.required_skills,
            "steps": instance.steps,
        })
    return result


def get_workflow(key: str) -> BaseWorkflow:
    """Get a workflow instance by its registry key.

    Args:
        key: Workflow identifier ('gravity', 'magnetic', or 'joint').

    Returns:
        An instance of the requested workflow.

    Raises:
        KeyError: If the key is not in the registry.
    """
    if key not in _WORKFLOW_REGISTRY:
        available = ", ".join(_WORKFLOW_REGISTRY.keys())
        raise KeyError(
            f"Unknown workflow: '{key}'. Available workflows: {available}"
        )
    return _WORKFLOW_REGISTRY[key]()
