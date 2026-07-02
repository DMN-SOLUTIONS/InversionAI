# Simple Magnetic Demo Dataset

## Description

Synthetic magnetic data for testing and demonstration of magnetic inversion workflows.

## Source Model

- **Geometry:** 200m × 200m × 200m cube (rectangular prism)
- **Location:** Centered at (500, 500) m, depth range 300–500 m below surface
- **Susceptibility contrast:** 0.05 SI relative to background (0 SI)
- **Forward model:** Analytic solution for a uniformly magnetized rectangular prism

## Inducing Field Parameters

- **Field strength:** 55,000 nT
- **Inclination:** -60° (Southern Hemisphere, e.g., Western Australia)
- **Declination:** 0°
- **Magnetization:** Purely induced (no remanence)

## Survey Configuration

- **Grid:** 10 × 10 regular grid = 100 stations
- **Station spacing:** 100 m
- **Coordinate range:** 50–950 m in both X and Y
- **Elevation:** All stations at z = 0 m (flat surface)
- **Component:** Total Magnetic Intensity (TMI) anomaly

## Data Properties

- **Units:** nanoTesla (nT)
- **Noise:** Gaussian noise with σ = 3% of maximum signal amplitude
- **Uncertainty:** Absolute value of noise + 1% of max signal (floor)
- **Anomaly range:** ~-40 to ~+8 nT (asymmetric due to inclination)

## Files

| File | Description |
|------|-------------|
| `observations.csv` | Station locations and magnetic observations (CSV) |
| `mesh.txt` | UBC-format tensor mesh (20×20×10 cells, 50m spacing) |
| `true_model.txt` | True susceptibility model (SI) for verification |

## Mesh Details

- **Dimensions:** 20 × 20 × 10 = 4,000 cells
- **Cell size:** 50 m × 50 m × 50 m (uniform)
- **Extent:** 1000 m × 1000 m × 500 m
- **Origin:** (0, 0, 0)

## Notes

The TMI anomaly is asymmetric due to the -60° inclination angle. The negative
lobe appears over the northern edge of the body, while a weaker positive lobe
appears to the south. This is characteristic of induced magnetization in the
Southern Hemisphere.

## Usage

```python
from data import DataLoader, DataValidator

# Validate
validator = DataValidator()
result = validator.validate_magnetic_data("observations.csv")
print(result)

# Load
loader = DataLoader()
dataset = loader.load("observations.csv")
print(f"Stations: {dataset.n_stations}")
print(f"TMI range: {dataset.observations.min():.2f} to {dataset.observations.max():.2f} nT")
```

## Reference

Forward modeling uses the analytic solution for the magnetic field of a uniformly
magnetized rectangular prism projected onto the Earth's field direction (TMI
approximation).
