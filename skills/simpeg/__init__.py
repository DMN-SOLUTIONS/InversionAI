"""
SimPEG inversion skill package.

Provides the SimpegSkill class for running Python-native geophysical
inversions with flexible regularization and mesh options.
"""

from .skill import SimpegSkill

__all__ = ["SimpegSkill"]
