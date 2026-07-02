"""
InversionAI Skills Module.

Each algorithm is wrapped as a 'skill' providing a uniform interface for
setup, validation, configuration, execution, and result retrieval. The
orchestrator agent can treat all algorithms interchangeably through the
BaseSkill interface.

Available Skills:
    - TomofastSkill: Parallel 3D potential field inversion (Fortran/MPI)
    - SimpegSkill: Python-native inversion with flexible regularization
    - CompareSkill: Cross-algorithm comparison and visualization

Usage:
    from skills import TomofastSkill, SimpegSkill, CompareSkill, list_available_skills

    # List all available skills
    skills = list_available_skills()

    # Use a specific skill
    tomofast = TomofastSkill(work_dir="./tomofast_output")
    tomofast.setup()
    result = tomofast.run(config)
"""

from .base_skill import (
    BaseSkill,
    InversionResult,
    ProgressCallback,
    RunConfig,
    SkillStatus,
    ValidationResult,
)
from .compare import CompareSkill
from .simpeg import SimpegSkill
from .tomofast import TomofastSkill


def list_available_skills() -> list[dict[str, str]]:
    """List all available inversion skills with their metadata.

    Returns:
        List of dictionaries with skill information:
            - name: Skill identifier
            - version: Skill version
            - description: Human-readable description
            - supported_data_types: List of supported data types
            - status: Current skill status
    """
    skills = [
        TomofastSkill(),
        SimpegSkill(),
        CompareSkill(),
    ]
    return [skill.info() for skill in skills]


__all__ = [
    "BaseSkill",
    "CompareSkill",
    "InversionResult",
    "ProgressCallback",
    "RunConfig",
    "SimpegSkill",
    "SkillStatus",
    "TomofastSkill",
    "ValidationResult",
    "list_available_skills",
]
