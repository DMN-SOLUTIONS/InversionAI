"""Shared pytest fixtures for InversionAI test suite."""

import tempfile
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def sample_gravity_data():
    """Generate sample gravity survey data for testing.

    Returns a dict with arrays representing a small gravity survey:
    - x, y, z: station coordinates (metres)
    - gobs: observed gravity anomaly (mGal)
    - std: standard deviation / uncertainty (mGal)
    """
    np.random.seed(42)
    n_stations = 25
    x = np.linspace(0, 1000, n_stations)
    y = np.linspace(0, 1000, n_stations)
    z = np.zeros(n_stations)
    gobs = np.random.normal(loc=-5.0, scale=2.0, size=n_stations)
    std = np.full(n_stations, 0.05)

    return {
        "x": x,
        "y": y,
        "z": z,
        "gobs": gobs,
        "std": std,
        "n_stations": n_stations,
        "units": "mGal",
        "coordinate_system": "local",
    }


@pytest.fixture
def sample_magnetic_data():
    """Generate sample magnetic survey data for testing.

    Returns a dict with arrays representing a small magnetic survey:
    - x, y, z: station coordinates (metres)
    - tmi: total magnetic intensity anomaly (nT)
    - std: standard deviation / uncertainty (nT)
    - inclination, declination: field direction (degrees)
    """
    np.random.seed(123)
    n_stations = 30
    x = np.linspace(0, 2000, n_stations)
    y = np.linspace(0, 2000, n_stations)
    z = np.full(n_stations, 80.0)
    tmi = np.random.normal(loc=100.0, scale=50.0, size=n_stations)
    std = np.full(n_stations, 1.0)

    return {
        "x": x,
        "y": y,
        "z": z,
        "tmi": tmi,
        "std": std,
        "n_stations": n_stations,
        "inclination": -60.0,
        "declination": 2.0,
        "field_strength": 55000.0,
        "units": "nT",
        "coordinate_system": "local",
    }


@pytest.fixture
def temp_output_dir(tmp_path):
    """Provide a temporary directory for test outputs."""
    output_dir = tmp_path / "inversion_output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def mock_inversion_result():
    """Create a mock InversionResult for testing downstream consumers."""
    np.random.seed(99)
    nx, ny, nz = 10, 10, 5
    n_cells = nx * ny * nz

    return {
        "model": np.random.uniform(-0.5, 0.5, size=n_cells),
        "mesh": {
            "nx": nx, "ny": ny, "nz": nz,
            "dx": 100.0, "dy": 100.0, "dz": 50.0,
            "origin": (0.0, 0.0, 0.0),
        },
        "predicted_data": np.random.normal(loc=-5.0, scale=2.0, size=25),
        "misfit": 1.02,
        "iterations": 15,
        "final_phi_d": 24.5,
        "final_phi_m": 0.003,
        "converged": True,
        "runtime_seconds": 42.3,
        "algorithm": "tomofast",
    }


@pytest.fixture
def sample_csv_file(tmp_path, sample_gravity_data):
    """Write sample gravity data to a CSV file and return the path."""
    csv_path = tmp_path / "gravity_data.csv"
    header = "x,y,z,gobs,std"
    data = np.column_stack([
        sample_gravity_data["x"],
        sample_gravity_data["y"],
        sample_gravity_data["z"],
        sample_gravity_data["gobs"],
        sample_gravity_data["std"],
    ])
    np.savetxt(csv_path, data, delimiter=",", header=header, comments="")
    return csv_path


@pytest.fixture
def sample_ubc_mesh_file(tmp_path):
    """Create a sample UBC mesh file and return the path."""
    mesh_path = tmp_path / "mesh.txt"
    content = (
        "10 10 5\n"
        "0.0 0.0 0.0\n"
        "100.0 100.0 100.0 100.0 100.0 100.0 100.0 100.0 100.0 100.0\n"
        "100.0 100.0 100.0 100.0 100.0 100.0 100.0 100.0 100.0 100.0\n"
        "50.0 50.0 50.0 50.0 50.0\n"
    )
    mesh_path.write_text(content)
    return mesh_path
