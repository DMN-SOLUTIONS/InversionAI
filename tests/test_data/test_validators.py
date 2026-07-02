"""Tests for data validation: valid data passes, invalid data caught with helpful messages."""

from pathlib import Path

import numpy as np
import pytest

from data.validators import DataValidator, ValidationResult


@pytest.fixture
def validator():
    """Create a DataValidator instance."""
    return DataValidator()


class TestGravityDataValidation:
    """Test gravity data validation via DataValidator."""

    def test_valid_gravity_data_passes(self, validator, tmp_path):
        """Well-formed gravity data passes all validation checks."""
        csv_path = tmp_path / "valid.csv"
        csv_path.write_text(
            "x,y,z,gobs,std\n"
            "0,0,0,-5.0,0.05\n"
            "100,0,0,-4.5,0.05\n"
            "200,0,0,-3.2,0.05\n"
        )
        result = validator.validate_gravity_data(csv_path)
        assert result.is_valid is True
        assert result.errors == []

    def test_missing_file_fails(self, validator, tmp_path):
        """Fails for non-existent file."""
        result = validator.validate_gravity_data(tmp_path / "nonexistent.csv")
        assert result.is_valid is False
        assert len(result.errors) > 0

    def test_empty_file_fails(self, validator, tmp_path):
        """Empty file is rejected with helpful message."""
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("")
        result = validator.validate_gravity_data(csv_path)
        assert result.is_valid is False
        assert len(result.errors) > 0

    def test_missing_gravity_column(self, validator, tmp_path):
        """Warns or fails when gravity data column cannot be identified by name."""
        csv_path = tmp_path / "no_gobs.csv"
        csv_path.write_text("a,b,c\n1,2,3\n")
        result = validator.validate_gravity_data(csv_path)
        # Validator may fail or produce a warning about missing columns
        assert (result.is_valid is False or
                len(result.warnings) > 0 or
                len(result.errors) > 0)

    def test_nan_values_rejected(self, validator, tmp_path):
        """NaN values in data are caught."""
        csv_path = tmp_path / "nan_data.csv"
        csv_path.write_text("x,y,z,gobs,std\n0,0,0,nan,0.05\n100,0,0,-4.0,0.05\n")
        result = validator.validate_gravity_data(csv_path)
        assert result.is_valid is False
        assert any("nan" in e.lower() or "invalid" in e.lower() or "numeric" in e.lower()
                   for e in result.errors)

    def test_negative_std_rejected(self, validator, tmp_path):
        """Negative standard deviations are rejected."""
        csv_path = tmp_path / "neg_std.csv"
        csv_path.write_text("x,y,z,gobs,std\n0,0,0,-5.0,-0.05\n100,0,0,-4.0,0.05\n")
        result = validator.validate_gravity_data(csv_path)
        assert result.is_valid is False or any(
            "negative" in w.lower() or "uncertainty" in w.lower()
            for w in result.errors + result.warnings
        )

    def test_extreme_values_flagged(self, validator, tmp_path):
        """Extreme gravity values outside reasonable range are flagged."""
        csv_path = tmp_path / "extreme.csv"
        csv_path.write_text("x,y,z,gobs,std\n0,0,0,9999.0,0.05\n")
        result = validator.validate_gravity_data(csv_path)
        # Should produce error or warning
        assert (not result.is_valid or
                len(result.warnings) > 0 or
                len(result.errors) > 0)

    def test_records_count_populated(self, validator, tmp_path):
        """Validation result reports the number of data records."""
        csv_path = tmp_path / "valid.csv"
        csv_path.write_text(
            "x,y,z,gobs,std\n"
            "0,0,0,-5.0,0.05\n"
            "100,0,0,-4.5,0.05\n"
            "200,0,0,-3.2,0.05\n"
        )
        result = validator.validate_gravity_data(csv_path)
        assert result.n_records == 3


class TestMagneticDataValidation:
    """Test magnetic data validation."""

    def test_valid_magnetic_data_passes(self, validator, tmp_path):
        """Well-formed magnetic data passes validation."""
        csv_path = tmp_path / "valid_mag.csv"
        csv_path.write_text(
            "x,y,z,tmi,std\n"
            "0,0,80,150.0,1.0\n"
            "100,0,80,120.0,1.0\n"
            "200,0,80,90.0,1.0\n"
        )
        result = validator.validate_magnetic_data(csv_path)
        assert result.is_valid is True

    def test_missing_tmi_column(self, validator, tmp_path):
        """Warns or fails when TMI column cannot be identified by name."""
        csv_path = tmp_path / "no_tmi.csv"
        csv_path.write_text("a,b,c\n1,2,3\n")
        result = validator.validate_magnetic_data(csv_path)
        # Validator may use positional fallback but should at least warn
        assert (result.is_valid is False or
                len(result.warnings) > 0 or
                len(result.errors) > 0)

    @pytest.mark.xfail(reason="Validator does not currently check for negative std in magnetic data")
    def test_negative_std_rejected(self, validator, tmp_path):
        """Negative standard deviation should be rejected or flagged."""
        csv_path = tmp_path / "neg_std.csv"
        csv_path.write_text(
            "x,y,z,tmi,std\n"
            "0,0,80,100.0,-1.0\n"
            "100,0,80,120.0,-2.0\n"
            "200,0,80,90.0,-0.5\n"
        )
        result = validator.validate_magnetic_data(csv_path)
        has_issue = (
            result.is_valid is False or
            any("negative" in m.lower() or "uncertainty" in m.lower()
                for m in result.errors + result.warnings)
        )
        assert has_issue

    def test_extreme_values_flagged(self, validator, tmp_path):
        """Extremely large TMI values are flagged."""
        csv_path = tmp_path / "extreme.csv"
        csv_path.write_text("x,y,z,tmi,std\n0,0,80,999999.0,1.0\n")
        result = validator.validate_magnetic_data(csv_path)
        # Should produce at least a warning
        assert (not result.is_valid or
                len(result.warnings) > 0 or
                len(result.errors) > 0)


class TestMeshValidation:
    """Test mesh file validation."""

    def test_valid_ubc_mesh_passes(self, validator, sample_ubc_mesh_file):
        """Valid UBC mesh file passes validation."""
        result = validator.validate_mesh(sample_ubc_mesh_file)
        assert result.is_valid is True

    def test_missing_mesh_file(self, validator, tmp_path):
        """Missing mesh file gives clear error."""
        result = validator.validate_mesh(tmp_path / "nonexistent.txt")
        assert result.is_valid is False
        assert len(result.errors) > 0

    def test_malformed_mesh(self, validator, tmp_path):
        """Malformed mesh file is caught."""
        mesh_path = tmp_path / "bad_mesh.txt"
        mesh_path.write_text("not a valid mesh\ngarbage data\n")
        result = validator.validate_mesh(mesh_path)
        assert result.is_valid is False


class TestValidationResultDataclass:
    """Test the ValidationResult dataclass from data.validators."""

    def test_default_is_valid(self):
        """Default ValidationResult is valid."""
        result = ValidationResult()
        assert result.is_valid is True

    def test_add_error_invalidates(self):
        """add_error() marks result as invalid."""
        result = ValidationResult()
        result.add_error("Something went wrong")
        assert result.is_valid is False
        assert "Something went wrong" in result.errors

    def test_add_warning_keeps_valid(self):
        """add_warning() does not invalidate the result."""
        result = ValidationResult()
        result.add_warning("Minor issue")
        assert result.is_valid is True
        assert "Minor issue" in result.warnings

    def test_str_representation(self):
        """String representation includes status and messages."""
        result = ValidationResult()
        result.add_error("Bad data")
        text = str(result)
        assert "INVALID" in text
        assert "Bad data" in text
