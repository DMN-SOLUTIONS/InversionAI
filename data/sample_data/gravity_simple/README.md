# Simple Gravity Demo Dataset

## Description

Synthetic gravity data for testing and demonstration of gravity inversion workflows.

## Source Model

- **Geometry:** 200m × 200m × 200m cube (rectangular prism)
- **Location:** Centered at (500, 500) m, depth range 300–500 m below surface
- **Density contrast:** 0.5 g/cm³ (500 kg/m³) relative to background (0 g/cm³)
- **Forward model:** Analytic solution for a rectangular prism (Blakely, 1996)

## Survey Configuration

- **Grid:** 10 × 10 regular grid = 100 stations
- **Station spacing:** 100 m
- **Coordinate range:** 50–950 m in both X and Y
- **Elevation:** All stations at z = 0 m (flat surface)
- **Component:** Vertical gravity anomaly (gz)

## Data Properties

- **Units:** milliGals (mGal)
- **Noise:** Gaussian noise with σ = 2% of maximum signal amplitude
- **Uncertainty:** Absolute value of noise + 1% of max signal (floor)
- **Anomaly range:** ~0.02 to ~0.16 mGal

## Files

| File | Description |
|------|-------------|
| `observations.csv` | Station locations and gravity observations (CSV) |
| `mesh.txt` | UBC-format tensor mesh (20×20×10 cells, 50m spacing) |
| `true_model.txt` | True density model (g/cm³) for verification |

## Mesh Details

- **Dimensions:** 20 × 20 × 10 = 4,000 cells
- **Cell size:** 50 m × 50 m × 50 m (uniform)
- **Extent:** 1000 m × 1000 m × 500 m
- **Origin:** (0, 0, 0)

## Usage

```python
from data import DataLoader, DataValidator

# Validate
validator = DataValidator()
result = validator.validate_gravity_data("observations.csv")
print(result)

# Load
loader = DataLoader()
dataset = loader.load("observations.csv")
print(f"Stations: {dataset.n_stations}")
print(f"Anomaly range: {dataset.observations.min():.4f} to {dataset.observations.max():.4f} mGal")
```

## Reference

Forward modeling uses the analytic solution for the gravitational attraction of a
right rectangular prism (e.g., Nagy, 1966; Blakely, 1996, "Potential Theory in
Gravity and Magnetic Applications").
