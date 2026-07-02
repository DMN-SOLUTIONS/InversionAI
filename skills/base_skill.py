"""
Base skill module for geophysical inversion algorithms.

Defines the abstract interface that all inversion skills must implement,
along with shared data structures for validation, configuration, and results.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

import logging

logger = logging.getLogger(__name__)


class SkillStatus(Enum):
    """Status of a skill's lifecycle."""
    READY = "ready"
    SETUP_REQUIRED = "setup_required"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ValidationResult:
    """Result of validating input data and configuration.

    Attributes:
        valid: Whether the data passes all required checks.
        errors: Critical issues that prevent execution.
        warnings: Non-critical issues that may affect results.
        suggestions: Recommendations for better results.
    """
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        status = "VALID" if self.valid else "INVALID"
        parts = [f"ValidationResult({status})"]
        if self.errors:
            parts.append(f"  Errors: {self.errors}")
        if self.warnings:
            parts.append(f"  Warnings: {self.warnings}")
        if self.suggestions:
            parts.append(f"  Suggestions: {self.suggestions}")
        return "\n".join(parts)


@dataclass
class RunConfig:
    """Configuration for running an inversion.

    Attributes:
        algorithm: Name of the inversion algorithm (e.g., 'tomofast-x', 'simpeg').
        data_path: Path to the input data file(s).
        mesh_config: Mesh configuration parameters (cell sizes, extent, padding).
        inversion_params: Algorithm-specific inversion parameters.
        output_dir: Directory where results will be written.
    """
    algorithm: str
    data_path: str
    mesh_config: dict[str, Any] = field(default_factory=dict)
    inversion_params: dict[str, Any] = field(default_factory=dict)
    output_dir: str = "./output"

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to dictionary."""
        return {
            "algorithm": self.algorithm,
            "data_path": self.data_path,
            "mesh_config": self.mesh_config,
            "inversion_params": self.inversion_params,
            "output_dir": self.output_dir,
        }


@dataclass
class InversionResult:
    """Result of a completed inversion run.

    Attributes:
        model_path: Path to the output model file.
        mesh_path: Path to the output mesh file.
        misfit_history: Data misfit at each iteration.
        final_misfit: Final data misfit value.
        n_iterations: Total iterations performed.
        runtime_seconds: Wall-clock runtime in seconds.
        metadata: Additional algorithm-specific metadata.
    """
    model_path: str
    mesh_path: str
    misfit_history: list[float] = field(default_factory=list)
    final_misfit: float = 0.0
    n_iterations: int = 0
    runtime_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def converged(self) -> bool:
        """Check if the inversion converged based on misfit history."""
        if len(self.misfit_history) < 2:
            return False
        # Consider converged if last change is < 1% of initial misfit
        initial = self.misfit_history[0]
        if initial == 0:
            return True
        last_change = abs(self.misfit_history[-1] - self.misfit_history[-2])
        return (last_change / initial) < 0.01

    def summary(self) -> str:
        """Human-readable summary of the inversion result."""
        return (
            f"InversionResult: {self.n_iterations} iterations in {self.runtime_seconds:.1f}s, "
            f"final misfit={self.final_misfit:.4f}, converged={self.converged}"
        )


# Type alias for progress callbacks
ProgressCallback = Callable[[int, int, float], None]
"""Callback signature: (current_iteration, total_iterations, current_misfit)"""


class BaseSkill(ABC):
    """Abstract base class for geophysical inversion skills.

    Each skill wraps a specific inversion algorithm and provides a uniform
    interface for setup, validation, configuration, execution, and result
    retrieval. This enables the orchestrator agent to treat all algorithms
    interchangeably.

    Subclasses must implement all abstract methods and properties.
    """

    def __init__(self) -> None:
        self._status: SkillStatus = SkillStatus.SETUP_REQUIRED
        self._last_result: Optional[InversionResult] = None
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name identifying this skill."""
        ...

    @property
    @abstractmethod
    def version(self) -> str:
        """Version string of the skill implementation."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of the skill's capabilities."""
        ...

    @property
    @abstractmethod
    def supported_data_types(self) -> list[str]:
        """List of supported data types (e.g., ['gravity', 'magnetic'])."""
        ...

    @property
    def status(self) -> SkillStatus:
        """Current status of the skill."""
        return self._status

    @abstractmethod
    def setup(self) -> bool:
        """Set up the skill environment (install dependencies, check binaries).

        Returns:
            True if setup completed successfully, False otherwise.
        """
        ...

    @abstractmethod
    def validate(self, data_config: dict[str, Any]) -> ValidationResult:
        """Validate input data and configuration before running.

        Args:
            data_config: Dictionary containing paths and parameters to validate.
                Expected keys vary by skill but typically include:
                - 'data_path': Path to observation data
                - 'mesh_path': Path to mesh file (if applicable)
                - 'data_type': Type of data ('gravity' or 'magnetic')

        Returns:
            ValidationResult with status, errors, warnings, and suggestions.
        """
        ...

    @abstractmethod
    def configure(self, user_intent: str, data_config: dict[str, Any]) -> RunConfig:
        """Generate a run configuration from user intent and data.

        This method translates natural language intent into concrete
        algorithm parameters. The agent can pass the user's description
        directly and receive a fully specified RunConfig.

        Args:
            user_intent: Natural language description of what the user wants.
            data_config: Dictionary with data paths and basic parameters.

        Returns:
            RunConfig ready to be passed to run().
        """
        ...

    @abstractmethod
    def run(
        self,
        config: RunConfig,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> InversionResult:
        """Execute the inversion with the given configuration.

        Args:
            config: RunConfig specifying all parameters.
            progress_callback: Optional callback for progress updates.
                Signature: (current_iter, total_iter, current_misfit)

        Returns:
            InversionResult containing paths to outputs and metrics.

        Raises:
            RuntimeError: If the inversion fails during execution.
        """
        ...

    @abstractmethod
    def get_results(self) -> InversionResult:
        """Retrieve the results of the last completed run.

        Returns:
            InversionResult from the most recent successful run.

        Raises:
            RuntimeError: If no results are available.
        """
        ...

    def supports(self, data_type: str) -> bool:
        """Check if this skill supports a given data type.

        Args:
            data_type: Data type to check (e.g., 'gravity', 'magnetic').

        Returns:
            True if the data type is supported.
        """
        return data_type.lower() in [dt.lower() for dt in self.supported_data_types]

    def reset(self) -> None:
        """Reset the skill to ready state, clearing previous results."""
        self._last_result = None
        if self._status in (SkillStatus.COMPLETED, SkillStatus.FAILED):
            self._status = SkillStatus.READY
        self._logger.info(f"Skill '{self.name}' reset to {self._status.value}")

    def info(self) -> dict[str, Any]:
        """Return skill metadata as a dictionary (agent-readable)."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "supported_data_types": self.supported_data_types,
            "status": self._status.value,
        }
