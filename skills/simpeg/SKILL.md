# SimPEG Inversion Skill

## What is SimPEG?

SimPEG (Simulation and Parameter Estimation in Geophysics) is an open-source Python framework for geophysical inversions. It provides modular components for mesh generation, forward modeling, regularization, and optimization.

**Repository:** https://github.com/simpeg/simpeg
**Documentation:** https://docs.simpeg.xyz/

## When to Use SimPEG

| Scenario | Recommendation |
|----------|---------------|
| Python-native workflow | ✅ Excellent - pure Python, easy to customize |
| IRLS / compact body inversion | ✅ Strong support for sparse/blocky models |
| Custom regularization | ✅ Highly flexible (Tikhonov, sparse, custom) |
| Adaptive meshing (TreeMesh) | ✅ Octree refinement near targets |
| Very large problems (>5M cells) | ⚠️ Memory-intensive; consider Tomofast-x |
| MPI parallelization | ❌ Not supported; single-process |
| Joint inversion | ⚠️ Possible but requires custom setup |
| Maximum speed | ❌ Python overhead; Tomofast-x is faster |

## Supported Data Types

- **Gravity** (`gravity`): Gz component (vertical gravity anomaly)
- **Magnetic** (`magnetic`): Total Magnetic Intensity (TMI)

## Mesh Types

### TensorMesh (Regular Grid)
- Uniform core cells with geometric padding
- Predictable memory usage
- Best for regional studies with uniform data coverage

### TreeMesh (Octree)
- Adaptive refinement near data and targets
- Fewer total cells for equivalent resolution
- Best for focused studies with variable resolution needs

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_iterations` | 20 | Maximum Gauss-Newton iterations |
| `alpha_s` | 1e-4 | Smallness regularization weight |
| `alpha_x` | 1.0 | Smoothness in x-direction |
| `alpha_y` | 1.0 | Smoothness in y-direction |
| `alpha_z` | 1.0 | Smoothness in z-direction |
| `chi_factor` | 1.0 | Target misfit (1.0 = fit to noise level) |
| `optimization` | "Gauss-Newton" | "Gauss-Newton" or "IRLS" |
| `upper_bound` | None | Upper bound on model values |
| `lower_bound` | None | Lower bound on model values |

## Optimization Methods

### Gauss-Newton (default)
- Standard L2 inversion
- Smooth models
- Fast convergence (10-20 iterations typical)
- Good starting point for any problem

### IRLS (Iteratively Reweighted Least Squares)
- Produces compact/blocky models
- Better for discrete geological bodies
- Requires more iterations (40-60)
- Two-phase: initial L2 + IRLS re-weighting

## Comparison with Tomofast-x

| Feature | SimPEG | Tomofast-x |
|---------|--------|------------|
| Language | Python | Fortran |
| Parallelism | Single-process | MPI |
| Speed (1M cells) | ~30 min | ~10 min |
| Memory | Higher | Lower |
| Customization | Excellent | Limited |
| IRLS support | ✅ Native | ⚠️ Limited |
| Joint inversion | Custom code | ✅ Native |
| Mesh flexibility | TreeMesh + Tensor | Regular grid |
| Output format | UBC, NumPy | VTK, text |

## Typical Runtime

| Problem Size | Mesh Type | Iterations | Approximate Runtime |
|-------------|-----------|-----------|-------------------|
| 50K cells | TensorMesh | 20 | 2-5 minutes |
| 200K cells | TensorMesh | 20 | 10-20 minutes |
| 500K cells | TreeMesh | 20 | 20-45 minutes |
| 1M cells | TreeMesh | 40 (IRLS) | 1-3 hours |

## Output Files

- `recovered_model.txt`: Model values (one per cell)
- `mesh.ubc`: UBC-format mesh file
- Compatible with UBC-GIF visualization tools

## Agent Instructions

When configuring SimPEG:
1. Always validate data format first
2. Start with TensorMesh for simplicity; switch to TreeMesh for large areas
3. Use Gauss-Newton first, then try IRLS if compact bodies expected
4. If chi_factor=1.0 target not reached, the data may have underestimated uncertainties
5. For gravity, consider lower_bound=0 for positive density contrasts
6. For magnetic susceptibility, lower_bound=0 (no remanence assumption)
7. Cell size should be ≤ half the data spacing for good resolution
