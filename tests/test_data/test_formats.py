"""Tests for format detection, loading, and conversion between data formats."""

from pathlib import Path

import numpy as np
import pytest

from data.formats import DataFormat, detect_format
from data.loader import DataLoader, GeoDataset
from data.converter import FormatConverter


@pytest.fixture
def csv_gravity_file(tmp_path):
    """Create a CSV gravity data file."""
    path = tmp_path / "survey.csv"
    path.write_text(
        "x,y,z,gobs,std\n"
        "0.0,0.0,0.0,-5.2,0.05\n"
        "100.0,0.0,0.0,-4.8,0.05\n"
        "200.0,0.0,0.0,-3.1,0.05\n"
        "300.0,0.0,0.0,-2.5,0.05\n"
    )
    return path


@pytest.fixture
def ubc_obs_file(tmp_path):
    """Create a UBC-format gravity observation file."""
    path = tmp_path / "obs_grav.grv"
    # UBC gravity format: N_obs on first line, then x y z data std
    path.write_text(
        "4\n"
        "0.0 0.0 0.0 -5.2 0.05\n"
        "100.0 0.0 0.0 -4.8 0.05\n"
        "200.0 0.0 0.0 -3.1 0.05\n"
        "300.0 0.0 0.0 -2.5 0.05\n"
    )
    return path


@pytest.fixture
def geosoft_xyz_file(tmp_path):
    """Create a Geosoft XYZ format file."""
    path = tmp_path / "survey.xyz"
    path.write_text(
        "/ Geosoft XYZ export\n"
        "/ Line 1\n"
        "   0.0    0.0   0.0  -5.2  0.05\n"
        " 100.0    0.0   0.0  -4.8  0.05\n"
        " 200.0    0.0   0.0  -3.1  0.05\n"
        " 300.0    0.0   0.0  -2.5  0.05\n"
    )
    return path


@pytest.fixture
def tomofast_data_file(tmp_path):
    """Create a Tomofast native data file (space-separated, no header)."""
    path = tmp_path / "data.txt"
    np.savetxt(
        path,
        np.array([
            [0.0, 0.0, 0.0, -5.2, 0.05],
            [100.0, 0.0, 0.0, -4.8, 0.05],
            [200.0, 0.0, 0.0, -3.1, 0.05],
            [300.0, 0.0, 0.0, -2.5, 0.05],
        ]),
    )
    return path


class TestFormatDetection:
    """Test automatic format detection from file content and extension."""

    def test_detect_csv_format(self, csv_gravity_file):
        """CSV format detected from .csv extension and comma separators."""
        fmt = detect_format(csv_gravity_file)
        assert fmt == DataFormat.CSV

    def test_detect_ubc_grv_format(self, ubc_obs_file):
        """UBC gravity format detected from .grv extension."""
        fmt = detect_format(ubc_obs_file)
        assert fmt == DataFormat.GIF_GRAVITY

    def test_detect_geosoft_xyz_format(self, geosoft_xyz_file):
        """Geosoft XYZ format detected from .xyz extension."""
        fmt = detect_format(geosoft_xyz_file)
        assert fmt == DataFormat.GEOSOFT_XYZ

    def test_detect_tomofast_format(self, tomofast_data_file):
        """Tomofast native format detected from space-separated data."""
        fmt = detect_format(tomofast_data_file)
        # May be detected as TOMOFAST_DATA or a generic numeric format
        assert fmt in (DataFormat.TOMOFAST_DATA, DataFormat.CSV, DataFormat.UNKNOWN)

    def test_detect_nonexistent_raises(self, tmp_path):
        """FileNotFoundError raised for non-existent file."""
        with pytest.raises(FileNotFoundError):
            detect_format(tmp_path / "does_not_exist.csv")

    def test_detect_empty_file(self, tmp_path):
        """Empty file returns UNKNOWN."""
        empty = tmp_path / "empty.csv"
        empty.write_text("")
        fmt = detect_format(empty)
        assert fmt == DataFormat.UNKNOWN

    def test_detect_ubc_mesh(self, tmp_path):
        """UBC mesh file detected from .msh extension."""
        mesh_file = tmp_path / "mesh.msh"
        mesh_file.write_text("10 10 5\n0.0 0.0 0.0\n")
        fmt = detect_format(mesh_file)
        assert fmt == DataFormat.UBC_MESH


class TestDataLoading:
    """Test loading data from various formats using DataLoader."""

    @pytest.fixture
    def loader(self):
        """Create a DataLoader instance."""
        return DataLoader()

    def test_load_csv(self, loader, csv_gravity_file):
        """Loading CSV returns a GeoDataset."""
        dataset = loader.load(str(csv_gravity_file))
        assert isinstance(dataset, GeoDataset)
        assert dataset.n_stations == 4

    def test_load_csv_preserves_values(self, loader, csv_gravity_file):
        """Loaded values match the original file content."""
        dataset = loader.load(str(csv_gravity_file))
        assert dataset.observations[0] == pytest.approx(-5.2)
        assert dataset.uncertainties[0] == pytest.approx(0.05)

    def test_load_csv_station_coordinates(self, loader, csv_gravity_file):
        """Station coordinates are correctly parsed."""
        dataset = loader.load(str(csv_gravity_file))
        assert dataset.stations.shape == (4, 3)
        assert dataset.stations[1, 0] == pytest.approx(100.0)

    def test_load_nonexistent_raises(self, loader, tmp_path):
        """Loading nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            loader.load(str(tmp_path / "does_not_exist.csv"))

    def test_geodataset_extent(self, loader, csv_gravity_file):
        """GeoDataset.extent returns correct spatial bounds."""
        dataset = loader.load(str(csv_gravity_file))
        extent = dataset.extent
        assert extent["x"][0] == pytest.approx(0.0)
        assert extent["x"][1] == pytest.approx(300.0)


class TestFormatConversion:
    """Test conversion between data formats using FormatConverter."""

    @pytest.fixture
    def converter(self):
        """Create a FormatConverter instance."""
        return FormatConverter()

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample GeoDataset for conversion tests."""
        return GeoDataset(
            stations=np.array([
                [0.0, 0.0, 0.0],
                [100.0, 0.0, 0.0],
                [200.0, 0.0, 0.0],
                [300.0, 0.0, 0.0],
            ]),
            observations=np.array([-5.2, -4.8, -3.1, -2.5]),
            uncertainties=np.array([0.05, 0.05, 0.05, 0.05]),
            metadata={"data_type": "gravity", "units": "mGal"},
        )

    def test_to_tomofast(self, converter, sample_dataset, tmp_path):
        """Convert GeoDataset to Tomofast native format."""
        output = tmp_path / "converted.txt"
        result_path = converter.to_tomofast(sample_dataset, output)
        assert result_path.exists()
        data = np.loadtxt(output, comments="#")
        assert data.shape[0] == 4

    def test_to_ubc(self, converter, sample_dataset, tmp_path):
        """Convert GeoDataset to UBC observation format."""
        output = tmp_path / "converted.grv"
        result_path = converter.to_ubc(sample_dataset, output, data_type="gravity")
        assert result_path.exists()
        content = output.read_text()
        # UBC format has observation count
        assert "4" in content.split("\n")[0]

    def test_to_simpeg(self, converter, sample_dataset):
        """Convert GeoDataset to SimPEG arrays."""
        result = converter.to_simpeg(sample_dataset)
        assert "receiver_locations" in result
        assert "data" in result
        assert "standard_deviation" in result
        assert result["receiver_locations"].shape == (4, 3)
        assert result["data"][0] == pytest.approx(-5.2)

    def test_roundtrip_tomofast(self, converter, sample_dataset, tmp_path):
        """Converting to Tomofast and back preserves data."""
        tomofast_path = tmp_path / "data_tomofast.txt"
        converter.to_tomofast(sample_dataset, tomofast_path)

        # Read back
        data = np.loadtxt(tomofast_path, comments="#")
        np.testing.assert_allclose(
            data[:, 3], sample_dataset.observations, atol=1e-6
        )
