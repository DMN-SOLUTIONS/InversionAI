# Quick Start Guide

Get InversionAI running in under 5 minutes.

## Prerequisites

You need **one** of the following:

- **Docker** (recommended) — Docker Desktop installed and running
- **Python 3.11+** with pip

## Option A: Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-org/InversionAI.git
cd InversionAI

# Start all services
docker compose up

# Open your browser
open http://localhost:8501
```

That's it. Docker handles all dependencies including Tomofast-X compilation.

## Option B: Local Installation

```bash
# Clone the repository
git clone https://github.com/your-org/InversionAI.git
cd InversionAI

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch the UI
streamlit run ui/app.py
```

Open http://localhost:8501 in your browser.

> **Note:** For Tomofast-X inversions on macOS (ARM), you still need Docker.
> SimPEG inversions work natively on all platforms.

## Run the Demo

1. **Load Demo Data** — Click the "Load Demo Data" button in the sidebar. This loads a synthetic gravity dataset over a buried density anomaly.

2. **Use Defaults** — Click "Use Defaults" to auto-configure both Tomofast-X and SimPEG with sensible inversion parameters (30 iterations, target misfit = 1.0).

3. **Start Inversion** — Click "Start Inversion". The system will:
   - Validate your input data
   - Run Tomofast-X inversion
   - Run SimPEG inversion
   - Compare the results

## What You'll See

Once the inversion completes, the UI displays:

| Panel | Description |
|-------|-------------|
| **Convergence Plot** | Misfit vs. iteration for each algorithm |
| **Recovered Models** | 3D density/susceptibility models side-by-side |
| **Difference Map** | Cell-by-cell difference between algorithms |
| **Metrics Table** | RMSE, correlation, runtime comparison |
| **Data Fit** | Observed vs. predicted data for each algorithm |

### Interpreting Results

- **Misfit ≈ 1.0** means the model fits the data within the noise level
- **High correlation** between algorithms suggests the result is robust
- **Low RMSE** in the difference map confirms structural agreement

## Next Steps

- Try your own data: see [Data Formats](data_formats.md) for supported formats
- Add a new algorithm: see [Adding Algorithms](adding_algorithms.md)
- Understand the architecture: see [Architecture](architecture.md)
