# Tomofast-x Inversion Skill

## What is Tomofast-x?

Tomofast-x is a high-performance, open-source 3D potential field inversion code written in Fortran with MPI parallelization. It is designed for large-scale gravity and magnetic inversions and supports joint inversion of both data types simultaneously.

**Repository:** https://github.com/TOMOFAST/Tomofast-x

## When to Use Tomofast-x

| Scenario | Recommendation |
|----------|---------------|
| Large-scale problems (>1M cells) | ✅ Excellent - MPI scales well |
| Fast turnaround needed | ✅ Fortran performance |
| Joint gravity-magnetic inversion | ✅ Native support |
| Custom regularization schemes | ⚠️ Limited - use SimPEG instead |
| IRLS / compact body inversion | ⚠️ Limited support |
| Python-native workflow | ❌ External binary execution |
| Rapid prototyping | ❌ Requires compilation or Docker |

## Supported Data Types

- **Gravity** (`gravity`): Bouguer anomaly, free-air anomaly
- **Magnetic** (`magnetic`): Total magnetic intensity (TMI)
- **Joint** (`joint`): Simultaneous gravity + magnetic inversion with structural coupling

## Input Data Format

### Observation Data
Text file with columns: `x y z value [error]`
- Coordinates in meters
- Gravity in mGal
- Magnetic in nT
- Optional error column for data weighting

### Mesh File
Regular 3D grid specification:
```
nx ny nz
x0 y0 z0
dx dy dz
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_iterations` | 50 | Number of inversion iterations |
| `regularization_weight` | 1.0 | Strength of model regularization |
| `depth_weighting` | True | Apply depth weighting to counteract decay |
| `depth_exponent` | 2.0 | Exponent for depth weighting kernel |
| `solver` | "LSQR" | Linear solver: LSQR or CG |
| `bounds` | None | Optional (min, max) model bounds |

## Solver Options

- **LSQR**: Least-squares QR decomposition. Generally faster convergence, good for well-posed problems.
- **CG**: Conjugate Gradient. Lower memory usage, better for very large problems.

## Typical Runtime

| Problem Size | Processors | Approximate Runtime |
|-------------|-----------|-------------------|
| 100K cells | 1 | 2-5 minutes |
| 500K cells | 1 | 10-30 minutes |
| 1M cells | 4 | 15-45 minutes |
| 5M cells | 8+ | 1-3 hours |

## Output Files

- `model_gravity.vtk` / `model_magnetic.vtk`: Inverted model in VTK format
- `misfit_history.txt`: Data misfit per iteration
- `predicted_data.txt`: Forward-modeled data from recovered model

## Execution Modes

1. **Native binary**: Fastest. Requires Fortran compiler + MPI on Linux.
2. **Docker**: Portable. Works on macOS/Windows without compilation.

## Agent Instructions

When configuring Tomofast-x:
1. Always validate data files first (`validate()`)
2. For initial runs, use default parameters
3. If misfit doesn't decrease, try reducing `regularization_weight`
4. For positive-only density models (e.g., ore bodies), set `bounds=(0.0, None)`
5. For joint inversions, ensure both datasets cover the same spatial extent
6. Monitor convergence - if flat, increase iterations or adjust regularization
