"""
InversionAI Data Module
========================

Handles geophysical data format detection, validation, loading, conversion,
and mesh generation for inversion workflows.

Main Classes:
    - DataLoader: Load geophysical data from various formats
    - DataValidator: Validate data files for correctness
    - FormatConverter: Convert between data formats

Utilities:
    - list_supported_formats(): Get all supported format specifications
    - detect_format(): Auto-detect file format
    - generate_mesh_from_data(): Generate inversion mesh from data extent

Example:
    >>> from data import DataLoader, DataValidator, FormatConverter, list_supported_formats
    >>> # Load data
    >>> loader = DataLoader()
    >>> dataset = loader.load("observations.csv")
    >>> # Validate
    >>> validator = DataValidator()
    >>> result = validator.validate_gravity_data("observations.csv")
    >>> # Convert
    >>> converter = FormatConverter()
    >>> converter.to_tomofast(dataset, "output/data.txt")
"""

from .converter import FormatConverter
from .formats import DataFormat, FormatSpec, detect_format, list_supported_formats
from .loader import DataLoader, GeoDataset
from .mesh_generator import MeshConfig, generate_mesh_from_data
from .validators import DataValidator, ValidationResult

__all__ = [
    "DataLoader",
    "DataValidator",
    "FormatConverter",
    "GeoDataset",
    "DataFormat",
    "FormatSpec",
    "MeshConfig",
    "ValidationResult",
    "detect_format",
    "generate_mesh_from_data",
    "list_supported_formats",
]
