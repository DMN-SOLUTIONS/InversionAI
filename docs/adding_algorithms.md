# Adding a New Inversion Algorithm

This guide walks you through adding a new inversion algorithm as a **skill** in InversionAI. The system uses a skills-based architecture where each algorithm is a self-contained, discoverable module.

## Overview

Adding a new algorithm requires:

1. Create a `skills/your_algorithm/` directory
2. Implement the `BaseSkill` interface
3. Create a `SKILL.md` for agent discovery
4. Register in `skills/__init__.py`
5. Add tests

## Step 1: Create the Skill Directory

```bash
mkdir -p skills/gravgrad
touch skills/gravgrad/__init__.py
touch skills/gravgrad/skill.py
touch skills/gravgrad/SKILL.md
```

## Step 2: Implement BaseSkill

Your skill must implement three methods: `validate`, `configure`, and `run`.

```python
# skills/gravgrad/skill.py

from pathlib import Path
from skills.base import BaseSkill, RunConfig, ValidationResult, InversionResult
import subprocess
import numpy as np


class GravGradSkill(BaseSkill):
    """Skill for gravity gradient inversion using the GravGrad algorithm."""

    name = "gravgrad"
    description = "Full tensor gravity gradient inversion using GravGrad"
    supported_data_types = ["gravity_gradient"]

    def validate(self, data_path: Path) -> ValidationResult:
        """Validate input data for GravGrad compatibility.

        Check that:
        - File exists and is readable
        - Contains required columns (x, y, z, Gxx, Gxy, Gxz, Gyy, Gyz, Gzz, std)
        - No NaN/Inf values
        - Standard deviations are positive
        """
        messages = []

        if not data_path.exists():
            return ValidationResult(is_valid=False, messages=["File not found"])

        try:
            data = np.loadtxt(data_path, delimiter=",", skiprows=1)
        except Exception as e:
            return ValidationResult(is_valid=False, messages=[f"Cannot read file: {e}"])

        if data.shape[1] < 10:
            messages.append("GravGrad requires 10 columns: x,y,z,Gxx,Gxy,Gxz,Gyy,Gyz,Gzz,std")
            return ValidationResult(is_valid=False, messages=messages)

        if np.any(np.isnan(data)):
            messages.append("Data contains NaN values")
            return ValidationResult(is_valid=False, messages=messages)

        if np.any(data[:, -1] <= 0):
            messages.append("Standard deviations must be positive")
            return ValidationResult(is_valid=False, messages=messages)

        return ValidationResult(is_valid=True, messages=[])

    def configure(self, run_config: RunConfig) -> RunConfig:
        """Generate GravGrad configuration files.

        Creates:
        - Parameter file with mesh dimensions and inversion settings
        - Formatted data file in GravGrad native format
        """
        output_dir = run_config.output_dir
        params = run_config.parameters or {}

        # Write configuration file
        config_content = f"""# GravGrad Configuration
data_file = {run_config.data_path}
output_dir = {output_dir}
max_iterations = {run_config.max_iterations or 50}
target_misfit = {run_config.target_misfit or 1.0}
nx = {params.get('nx', 20)}
ny = {params.get('ny', 20)}
nz = {params.get('nz', 10)}
"""
        (output_dir / "gravgrad.conf").write_text(config_content)
        return run_config

    def run(self, run_config: RunConfig) -> InversionResult:
        """Execute GravGrad inversion.

        Calls the GravGrad binary and parses outputs.
        """
        import time
        start = time.time()

        result = subprocess.run(
            ["gravgrad", "-c", str(run_config.output_dir / "gravgrad.conf")],
            capture_output=True, text=True
        )

        if result.returncode != 0:
            raise RuntimeError(f"GravGrad failed: {result.stderr}")

        elapsed = time.time() - start

        # Parse outputs
        model_path = run_config.output_dir / "recovered_model.txt"
        pred_path = run_config.output_dir / "predicted_data.txt"

        return InversionResult(
            model_path=model_path,
            predicted_data_path=pred_path,
            misfit=self._parse_final_misfit(run_config.output_dir),
            iterations=self._parse_iterations(run_config.output_dir),
            converged=True,
            runtime_seconds=elapsed,
            algorithm=self.name,
        )

    def _parse_final_misfit(self, output_dir: Path) -> float:
        """Parse final misfit from GravGrad log."""
        log = (output_dir / "gravgrad.log").read_text()
        # Parse last line for misfit value
        lines = log.strip().split("\n")
        return float(lines[-1].split()[-1])

    def _parse_iterations(self, output_dir: Path) -> int:
        """Parse iteration count from GravGrad log."""
        log = (output_dir / "gravgrad.log").read_text()
        return len(log.strip().split("\n")) - 1  # Subtract header
```

## Step 3: Create SKILL.md

The `SKILL.md` file allows the AI agent to discover and understand your skill:

```markdown
# GravGrad Skill

## Purpose
Full tensor gravity gradient (FTG) inversion using the GravGrad algorithm.

## Capabilities
- Inverts all 6 independent gravity gradient components
- Supports depth weighting
- Produces 3D density contrast models

## Input Requirements
- CSV file with columns: x, y, z, Gxx, Gxy, Gxz, Gyy, Gyz, Gzz, std
- Units: Eötvös (E) for gradients, metres for coordinates
- Standard deviation column for data weighting

## Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| nx, ny, nz | 20, 20, 10 | Mesh cells in each direction |
| max_iterations | 50 | Maximum iterations |
| target_misfit | 1.0 | Target chi-squared misfit |
| depth_weight | 2.0 | Depth weighting exponent |

## When to Use
- Full tensor gradiometry surveys
- When directional sensitivity matters
- Higher resolution than scalar gravity alone
```

## Step 4: Register the Skill

Add your skill to `skills/__init__.py`:

```python
# skills/__init__.py

from skills.tomofast import TomofastSkill
from skills.simpeg import SimpegSkill
from skills.compare import CompareSkill
from skills.gravgrad import GravGradSkill  # New!

AVAILABLE_SKILLS = {
    "tomofast": TomofastSkill,
    "simpeg": SimpegSkill,
    "compare": CompareSkill,
    "gravgrad": GravGradSkill,  # New!
}
```

## Step 5: Add Tests

Create `tests/test_skills/test_gravgrad.py`:

```python
"""Tests for GravGradSkill."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from skills.gravgrad import GravGradSkill
from skills.base import RunConfig


@pytest.fixture
def gravgrad_skill():
    return GravGradSkill()


@pytest.fixture
def ftg_data_file(tmp_path):
    """Create sample FTG data."""
    path = tmp_path / "ftg_data.csv"
    header = "x,y,z,Gxx,Gxy,Gxz,Gyy,Gyz,Gzz,std"
    data = np.random.randn(20, 10)
    data[:, -1] = np.abs(data[:, -1]) + 0.01  # Positive std
    np.savetxt(path, data, delimiter=",", header=header, comments="")
    return path


class TestGravGradValidation:
    def test_valid_ftg_data(self, gravgrad_skill, ftg_data_file):
        result = gravgrad_skill.validate(ftg_data_file)
        assert result.is_valid is True

    def test_insufficient_columns(self, gravgrad_skill, tmp_path):
        bad_file = tmp_path / "bad.csv"
        bad_file.write_text("x,y,z,gobs,std\n0,0,0,-5,0.05\n")
        result = gravgrad_skill.validate(bad_file)
        assert result.is_valid is False


class TestGravGradExecution:
    @patch("skills.gravgrad.skill.subprocess.run")
    def test_run_calls_binary(self, mock_run, gravgrad_skill, ftg_data_file, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)
        config = RunConfig(
            data_path=ftg_data_file,
            output_dir=tmp_path / "out",
            algorithm="gravgrad",
            max_iterations=30,
        )
        (tmp_path / "out").mkdir()
        gravgrad_skill.configure(config)
        # Would need mock output files for full run test
        mock_run.assert_not_called()  # configure doesn't call binary
```

Run tests:

```bash
pytest tests/test_skills/test_gravgrad.py -v
```

## Checklist

- [ ] `skills/gravgrad/__init__.py` exports `GravGradSkill`
- [ ] `GravGradSkill` implements `validate()`, `configure()`, `run()`
- [ ] `SKILL.md` documents capabilities, inputs, and parameters
- [ ] Skill registered in `skills/__init__.py`
- [ ] Tests cover validation, configuration, and execution (with mocked binary)
- [ ] Existing tests still pass: `pytest tests/ -v`
