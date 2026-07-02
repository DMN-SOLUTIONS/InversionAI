# Comparison Skill

## Overview

The Compare skill evaluates and contrasts inversion results from different algorithms (e.g., Tomofast-x vs SimPEG). It produces quantitative metrics and interactive visualizations to help determine which algorithm performs better for a given dataset.

## When to Use

- After running the same dataset through two different inversion algorithms
- To evaluate the effect of different parameters on the same algorithm
- For benchmarking and validation studies
- To communicate results in publications or reports

## Methodology

### Model Normalization

Before comparison, models are interpolated onto a common regular grid:

1. **Bounding box intersection**: Find the overlapping domain of both models
2. **Resolution selection**: Use the coarser of the two model resolutions (conservative)
3. **Interpolation**: Linear interpolation with nearest-neighbor fallback for gaps
4. **Alignment**: Both models sampled at identical grid points

### Metrics

| Metric | Description | Range | Best Value |
|--------|-------------|-------|-----------|
| `model_correlation` | Pearson correlation coefficient | [-1, 1] | 1.0 |
| `structural_similarity_index` | SSIM (luminance, contrast, structure) | [-1, 1] | 1.0 |
| `depth_weighted_rms` | RMS difference weighted by depth | [0, ∞) | 0.0 |
| `convergence_rate` | Average misfit reduction per iteration | [0, 1] | Higher = faster |
| `misfit_reduction_rate` | Total fractional misfit reduction | [0, 1] | Higher = better |
| `runtime_comparison` | Relative execution time | (0, ∞) | Depends on context |

### Depth-Weighted RMS

The depth-weighted RMS emphasizes differences in the shallow subsurface where geophysical resolution is highest. The weighting function is:

```
w(z) = 1 / (1 + depth/total_depth)
```

This gives double weight to surface cells compared to the deepest cells.

### Structural Similarity Index (SSIM)

Adapted from image processing, SSIM compares:
- **Luminance**: Mean model values
- **Contrast**: Variance of model values
- **Structure**: Covariance between models

SSIM is more perceptually meaningful than simple RMS because it captures structural patterns.

## Visualizations Generated

1. **Convergence curves**: Log-scale misfit vs iteration for both algorithms
2. **Model histograms**: Distribution of recovered model values
3. **XY cross-section**: Horizontal slice at mid-depth
4. **XZ cross-section**: Vertical slice along x-direction
5. **YZ cross-section**: Vertical slice along y-direction
6. **Depth slices**: Horizontal slices at user-specified depths
7. **Difference map**: Maximum absolute difference projected along each axis

All plots are generated using Plotly for interactivity (HTML) or static export (PNG).

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `depth_slices` | [50, 100, 200, 500] | Depths for slice comparison (m) |
| `generate_plots` | True | Whether to generate visualization plots |
| `plot_format` | "html" | Output format: "html" (interactive) or "png" |
| `label_a` | "Model A" | Display label for first model |
| `label_b` | "Model B" | Display label for second model |

## Interpreting Results

### Correlation > 0.9
Models are highly similar. Differences are likely in fine details or amplitude.

### Correlation 0.7 - 0.9
Models agree on large-scale structure but differ in details. Check depth-weighted RMS to see where differences concentrate.

### Correlation < 0.7
Significant structural differences. Consider:
- Are the algorithms converging to different local minima?
- Is one algorithm better regularized?
- Do the depth slices show where agreement breaks down?

### Convergence Analysis
- If one algorithm converges faster, it may be more efficient for the problem size
- If final misfits differ significantly, the lower-misfit result fits data better
- Stalled convergence suggests regularization issues

## Agent Instructions

1. Always run both inversions to completion before comparing
2. Use the same data and similar mesh resolutions for fair comparison
3. Report correlation and SSIM as primary similarity metrics
4. Use depth-weighted RMS to identify where models diverge
5. Include convergence plots to show efficiency differences
6. Highlight runtime differences for practical recommendations
