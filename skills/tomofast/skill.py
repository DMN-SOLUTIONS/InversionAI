"""
Tomofast-x inversion skill implementation.

Wraps the Tomofast-x Fortran inversion code, handling binary detection,
Docker fallback, parameter file generation, execution via MPI, and
output parsing.
"""

import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

from ..base_skill import (
    BaseSkill,
    InversionResult,
    ProgressCallback,
    RunConfig,
    SkillStatus,
    ValidationResult,
)
from .output_parser import parse_stdout_progress, parse_output_model
from .parfile_builder import ParfileBuilder

logger = logging.getLogger(__name__)


class TomofastSkill(BaseSkill):
    """Skill for running Tomofast-x geophysical inversions.

    Tomofast-x is a parallel 3D potential field inversion code written in
    Fortran with MPI support. It handles gravity and magnetic data,
    including joint inversions.

    This skill manages:
    - Binary detection or Docker-based execution
    - Parameter file (Parfile) generation
    - MPI execution with progress monitoring
    - Output VTK/text file parsing
    """

    BINARY_NAME = "tomofastx"
    DOCKER_IMAGE = "ghcr.io/tomofast/tomofast-x:latest"

    def __init__(self, work_dir: str = "./tomofast_work") -> None:
        """Initialize TomofastSkill.

        Args:
            work_dir: Working directory for intermediate files and outputs.
        """
        super().__init__()
        self._work_dir = Path(work_dir)
        self._binary_path: Optional[str] = None
        self._use_docker: bool = False
        self._parfile_builder = ParfileBuilder()

    @property
    def name(self) -> str:
        return "tomofast-x"

    @property
    def version(self) -> str:
        return "2.0"

    @property
    def description(self) -> str:
        return (
            "Tomofast-x: Parallel 3D potential field inversion using MPI. "
            "Supports gravity, magnetic, and joint gravity-magnetic inversions. "
            "Written in Fortran for high performance on large-scale problems."
        )

    @property
    def supported_data_types(self) -> list[str]:
        return ["gravity", "magnetic", "joint"]

    def setup(self) -> bool:
        """Set up Tomofast-x execution environment.

        Checks for native binary first, then falls back to Docker.

        Returns:
            True if a valid execution method is available.
        """
        self._logger.info("Setting up Tomofast-x skill...")

        # Check for native binary
        binary = shutil.which(self.BINARY_NAME)
        if binary:
            self._binary_path = binary
            self._use_docker = False
            self._status = SkillStatus.READY
            self._logger.info(f"Found native binary at: {binary}")
            return True

        # Check common install locations
        common_paths = [
            "/usr/local/bin/tomofastx",
            os.path.expanduser("~/tomofast-x/tomofastx"),
            os.path.expanduser("~/bin/tomofastx"),
        ]
        for path in common_paths:
            if os.path.isfile(path) and os.access(path, os.X_OK):
                self._binary_path = path
                self._use_docker = False
                self._status = SkillStatus.READY
                self._logger.info(f"Found binary at: {path}")
                return True

        # Fall back to Docker
        if self._check_docker_available():
            self._use_docker = True
            self._status = SkillStatus.READY
            self._logger.info("Using Docker for Tomofast-x execution")
            return self._pull_docker_image()

        self._status = SkillStatus.SETUP_REQUIRED
        self._logger.error(
            "Tomofast-x binary not found and Docker not available. "
            "Please install Tomofast-x or Docker."
        )
        return False

    def _check_docker_available(self) -> bool:
        """Check if Docker is installed and running."""
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                timeout=10,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _pull_docker_image(self) -> bool:
        """Pull the Tomofast-x Docker image."""
        self._logger.info(f"Pulling Docker image: {self.DOCKER_IMAGE}")
        try:
            result = subprocess.run(
                ["docker", "pull", self.DOCKER_IMAGE],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode == 0:
                self._logger.info("Docker image pulled successfully")
                return True
            else:
                self._logger.error(f"Failed to pull Docker image: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            self._logger.error("Docker pull timed out")
            return False
        except FileNotFoundError:
            self._logger.error("Docker not found")
            return False

    def validate(self, data_config: dict[str, Any]) -> ValidationResult:
        """Validate data files and configuration for Tomofast-x.

        Args:
            data_config: Expected keys:
                - 'data_path': Path to observation data file
                - 'mesh_path': Path to mesh file
                - 'data_type': One of 'gravity', 'magnetic', 'joint'

        Returns:
            ValidationResult with detailed error/warning information.
        """
        errors: list[str] = []
        warnings: list[str] = []
        suggestions: list[str] = []

        # Check required keys
        if "data_path" not in data_config:
            errors.append("Missing required key 'data_path' in data_config")
        if "data_type" not in data_config:
            errors.append("Missing required key 'data_type' in data_config")

        if errors:
            return ValidationResult(valid=False, errors=errors, warnings=warnings, suggestions=suggestions)

        data_type = data_config["data_type"]
        if data_type not in self.supported_data_types:
            errors.append(
                f"Unsupported data type '{data_type}'. "
                f"Supported: {self.supported_data_types}"
            )
            return ValidationResult(valid=False, errors=errors, warnings=warnings, suggestions=suggestions)

        # Validate data file
        data_path = Path(data_config["data_path"])
        if not data_path.exists():
            errors.append(f"Data file not found: {data_path}")
        elif data_path.stat().st_size == 0:
            errors.append(f"Data file is empty: {data_path}")
        else:
            # Check data file format (Tomofast expects specific column format)
            validation = self._validate_data_format(data_path, data_type)
            errors.extend(validation.get("errors", []))
            warnings.extend(validation.get("warnings", []))

        # Validate mesh file if provided
        if "mesh_path" in data_config:
            mesh_path = Path(data_config["mesh_path"])
            if not mesh_path.exists():
                errors.append(f"Mesh file not found: {mesh_path}")
            elif mesh_path.stat().st_size == 0:
                errors.append(f"Mesh file is empty: {mesh_path}")

        # For joint inversion, check second data file
        if data_type == "joint":
            if "data_path_secondary" not in data_config:
                errors.append("Joint inversion requires 'data_path_secondary'")
            else:
                secondary = Path(data_config["data_path_secondary"])
                if not secondary.exists():
                    errors.append(f"Secondary data file not found: {secondary}")

        # Suggestions
        if "n_iterations" not in data_config.get("params", {}):
            suggestions.append("Consider specifying n_iterations (default: 50)")
        if "regularization_weight" not in data_config.get("params", {}):
            suggestions.append("Consider specifying regularization_weight for better control")

        # Skill readiness
        if self._status != SkillStatus.READY:
            warnings.append("Skill is not in READY state. Run setup() first.")

        valid = len(errors) == 0
        return ValidationResult(valid=valid, errors=errors, warnings=warnings, suggestions=suggestions)

    def _validate_data_format(self, data_path: Path, data_type: str) -> dict[str, list[str]]:
        """Validate the format of a data file.

        Tomofast-x expects data files with specific column layouts:
        - Gravity: x, y, z, gobs, [error]
        - Magnetic: x, y, z, Bobs, [inclination, declination, error]

        Returns:
            Dict with 'errors' and 'warnings' lists.
        """
        result: dict[str, list[str]] = {"errors": [], "warnings": []}
        try:
            with open(data_path, "r") as f:
                lines = f.readlines()

            if len(lines) < 2:
                result["errors"].append(
                    f"Data file has too few lines ({len(lines)}). "
                    "Expected header + data rows."
                )
                return result

            # Check first data line for column count
            # Skip comment lines (starting with # or !)
            data_lines = [l for l in lines if not l.strip().startswith(("#", "!"))]
            if not data_lines:
                result["errors"].append("No data lines found (all lines are comments)")
                return result

            first_data = data_lines[0].split()
            n_cols = len(first_data)

            if data_type == "gravity" and n_cols < 4:
                result["errors"].append(
                    f"Gravity data requires at least 4 columns (x, y, z, gobs), "
                    f"found {n_cols}"
                )
            elif data_type == "magnetic" and n_cols < 4:
                result["errors"].append(
                    f"Magnetic data requires at least 4 columns (x, y, z, Bobs), "
                    f"found {n_cols}"
                )

            # Check that values are numeric
            try:
                [float(v) for v in first_data[:4]]
            except ValueError:
                result["errors"].append("Non-numeric values found in data columns")

            # Warn about large datasets
            n_data = len(data_lines)
            if n_data > 100000:
                result["warnings"].append(
                    f"Large dataset ({n_data} points). Consider downsampling "
                    "for initial testing."
                )

        except IOError as e:
            result["errors"].append(f"Could not read data file: {e}")

        return result

    def configure(self, user_intent: str, data_config: dict[str, Any]) -> RunConfig:
        """Generate Tomofast-x configuration from user intent.

        Parses the user's natural language intent to determine inversion
        parameters, then generates a Parfile.

        Args:
            user_intent: Natural language description (e.g., "Run gravity
                inversion with 100 iterations and strong regularization")
            data_config: Data paths and basic parameters.

        Returns:
            RunConfig with generated Parfile path and parameters.
        """
        self._logger.info(f"Configuring Tomofast-x from intent: '{user_intent}'")

        # Extract parameters from intent
        params = self._parse_intent(user_intent, data_config)

        # Create working directory
        output_dir = Path(data_config.get("output_dir", str(self._work_dir / "output")))
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate Parfile
        parfile_path = self._work_dir / "Parfile"
        self._work_dir.mkdir(parents=True, exist_ok=True)

        self._parfile_builder.build(
            output_path=str(parfile_path),
            data_type=params["data_type"],
            data_path=data_config["data_path"],
            mesh_path=data_config.get("mesh_path", ""),
            output_dir=str(output_dir),
            n_iterations=params["n_iterations"],
            regularization_weight=params["regularization_weight"],
            depth_weighting=params["depth_weighting"],
            solver=params["solver"],
            bounds=params.get("bounds"),
            data_path_secondary=data_config.get("data_path_secondary"),
        )

        self._logger.info(f"Generated Parfile at: {parfile_path}")

        return RunConfig(
            algorithm="tomofast-x",
            data_path=data_config["data_path"],
            mesh_config={
                "mesh_path": data_config.get("mesh_path", ""),
                "parfile_path": str(parfile_path),
            },
            inversion_params=params,
            output_dir=str(output_dir),
        )

    def _parse_intent(self, user_intent: str, data_config: dict[str, Any]) -> dict[str, Any]:
        """Parse user intent string into concrete parameters.

        Uses keyword matching to extract parameters from natural language.
        Falls back to sensible defaults.

        Args:
            user_intent: Natural language description.
            data_config: Supplemental data configuration.

        Returns:
            Dictionary of parsed parameters.
        """
        intent_lower = user_intent.lower()
        params: dict[str, Any] = {}

        # Data type
        params["data_type"] = data_config.get("data_type", "gravity")
        if "joint" in intent_lower:
            params["data_type"] = "joint"
        elif "magnetic" in intent_lower:
            params["data_type"] = "magnetic"
        elif "gravity" in intent_lower:
            params["data_type"] = "gravity"

        # Iterations
        params["n_iterations"] = data_config.get("params", {}).get("n_iterations", 50)
        import re
        iter_match = re.search(r"(\d+)\s*iteration", intent_lower)
        if iter_match:
            params["n_iterations"] = int(iter_match.group(1))

        # Regularization
        params["regularization_weight"] = data_config.get("params", {}).get(
            "regularization_weight", 1.0
        )
        if "strong regularization" in intent_lower:
            params["regularization_weight"] = 10.0
        elif "weak regularization" in intent_lower:
            params["regularization_weight"] = 0.1
        elif "no regularization" in intent_lower:
            params["regularization_weight"] = 0.0

        reg_match = re.search(r"regularization[\s_]*(?:weight)?[\s:=]*([0-9.]+)", intent_lower)
        if reg_match:
            params["regularization_weight"] = float(reg_match.group(1))

        # Depth weighting
        params["depth_weighting"] = data_config.get("params", {}).get("depth_weighting", True)
        if "no depth weighting" in intent_lower:
            params["depth_weighting"] = False

        # Solver
        params["solver"] = data_config.get("params", {}).get("solver", "LSQR")
        if "lsqr" in intent_lower:
            params["solver"] = "LSQR"
        elif "cg" in intent_lower or "conjugate gradient" in intent_lower:
            params["solver"] = "CG"

        # Bounds
        params["bounds"] = data_config.get("params", {}).get("bounds")
        if "positive" in intent_lower or "positivity" in intent_lower:
            params["bounds"] = (0.0, None)

        return params

    def run(
        self,
        config: RunConfig,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> InversionResult:
        """Execute Tomofast-x inversion.

        Args:
            config: RunConfig from configure().
            progress_callback: Optional callback for progress reporting.

        Returns:
            InversionResult with output paths and metrics.

        Raises:
            RuntimeError: If execution fails.
        """
        self._status = SkillStatus.RUNNING
        self._logger.info(f"Starting Tomofast-x inversion: {config.inversion_params}")

        parfile_path = config.mesh_config.get("parfile_path", "")
        if not parfile_path or not Path(parfile_path).exists():
            self._status = SkillStatus.FAILED
            raise RuntimeError(f"Parfile not found: {parfile_path}")

        start_time = time.time()
        misfit_history: list[float] = []
        total_iterations = config.inversion_params.get("n_iterations", 50)

        try:
            cmd = self._build_command(parfile_path)
            self._logger.info(f"Executing: {' '.join(cmd)}")

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(self._work_dir),
            )

            # Stream stdout and parse progress
            assert process.stdout is not None
            for line in process.stdout:
                line = line.strip()
                if not line:
                    continue

                progress = parse_stdout_progress(line)
                if progress:
                    iteration, misfit = progress
                    misfit_history.append(misfit)
                    if progress_callback:
                        progress_callback(iteration, total_iterations, misfit)
                    self._logger.debug(
                        f"Iteration {iteration}/{total_iterations}, misfit={misfit:.6f}"
                    )

            process.wait()
            runtime = time.time() - start_time

            if process.returncode != 0:
                stderr = process.stderr.read() if process.stderr else ""
                self._status = SkillStatus.FAILED
                raise RuntimeError(
                    f"Tomofast-x exited with code {process.returncode}: {stderr}"
                )

            # Parse output files
            result = self._collect_results(config.output_dir, misfit_history, runtime)
            self._last_result = result
            self._status = SkillStatus.COMPLETED
            self._logger.info(f"Inversion completed: {result.summary()}")
            return result

        except subprocess.TimeoutExpired:
            self._status = SkillStatus.FAILED
            raise RuntimeError("Tomofast-x execution timed out")
        except OSError as e:
            self._status = SkillStatus.FAILED
            raise RuntimeError(f"Failed to execute Tomofast-x: {e}")

    def _build_command(self, parfile_path: str) -> list[str]:
        """Build the execution command."""
        if self._use_docker:
            work_dir = str(self._work_dir.resolve())
            return [
                "docker", "run", "--rm",
                "-v", f"{work_dir}:/work",
                "-w", "/work",
                self.DOCKER_IMAGE,
                "mpirun", "-np", "1", "--allow-run-as-root",
                "tomofastx", os.path.basename(parfile_path),
            ]
        else:
            binary = self._binary_path or self.BINARY_NAME
            return ["mpirun", "-np", "1", binary, parfile_path]

    def _collect_results(
        self,
        output_dir: str,
        misfit_history: list[float],
        runtime: float,
    ) -> InversionResult:
        """Collect and parse output files into InversionResult."""
        output_path = Path(output_dir)

        # Look for output model files
        model_path = ""
        mesh_path = ""

        # Tomofast-x outputs model as text or VTK
        for pattern in ["*.vtk", "*model*.txt", "*result*"]:
            matches = list(output_path.glob(pattern))
            if matches:
                model_path = str(matches[0])
                break

        # Look for mesh file
        for pattern in ["*mesh*.vtk", "*mesh*.txt"]:
            matches = list(output_path.glob(pattern))
            if matches:
                mesh_path = str(matches[0])
                break

        # If misfit history not captured from stdout, try parsing output files
        if not misfit_history:
            misfit_file = output_path / "misfit_history.txt"
            if misfit_file.exists():
                parsed = parse_output_model(str(misfit_file))
                misfit_history = parsed.get("misfit_history", [])

        final_misfit = misfit_history[-1] if misfit_history else 0.0

        return InversionResult(
            model_path=model_path,
            mesh_path=mesh_path,
            misfit_history=misfit_history,
            final_misfit=final_misfit,
            n_iterations=len(misfit_history),
            runtime_seconds=runtime,
            metadata={
                "algorithm": "tomofast-x",
                "execution_mode": "docker" if self._use_docker else "native",
                "output_dir": output_dir,
            },
        )

    def get_results(self) -> InversionResult:
        """Retrieve the last inversion result.

        Returns:
            InversionResult from the most recent run.

        Raises:
            RuntimeError: If no results are available.
        """
        if self._last_result is None:
            raise RuntimeError(
                "No results available. Run an inversion first with run()."
            )
        return self._last_result
