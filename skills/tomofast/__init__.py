"""
Tomofast-x inversion skill package.

Provides the TomofastSkill class for running parallel 3D potential field
inversions using the Tomofast-x Fortran code.
"""

from .skill import TomofastSkill

__all__ = ["TomofastSkill"]
