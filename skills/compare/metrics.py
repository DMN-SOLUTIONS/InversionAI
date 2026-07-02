"""
Comparison metrics for inversion model evaluation.

Provides quantitative measures to compare two inversion models including
spatial correlation, structural similarity, depth-weighted differences,
and convergence analysis.
"""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def model_correlation(model_a: "np.ndarray", model_b: "np.ndarray") -> float:
    """Compute Pearson correlation coefficient between two models.

    Measures the linear relationship between model values. A value of 1.0
    indicates perfectly correlated models, 0.0 indicates no correlation.

    Args:
        model_a: First model values (flattened).
        model_b: Second model values (flattened).

    Returns:
        Pearson correlation coefficient in range [-1, 1].
    """
    if len(model_a) != len(model_b):
        min_len = min(len(model_a), len(model_b))
        model_a = model_a[:min_len]
        model_b = model_b[:min_len]

    if len(model_a) == 0:
        return 0.0

    # Handle constant models
    std_a = np.std(model_a)
    std_b = np.std(model_b)
    if std_a == 0 or std_b == 0:
        logger.warning("One or both models are constant; correlation undefined")
        return 0.0

    correlation = np.corrcoef(model_a, model_b)[0, 1]
    return float(correlation)


def structural_similarity_index(
    model_a: "np.ndarray",
    model_b: "np.ndarray",
    data_range: float | None = None,
) -> float:
    """Compute Structural Similarity Index (SSIM) between two models.

    SSIM considers luminance, contrast, and structure. Adapted for 1D
    model comparison (applied to sorted or windowed values).

    Values range from -1 to 1, where 1 indicates identical structure.

    Args:
        model_a: First model values.
        model_b: Second model values.
        data_range: Dynamic range of data. If None, computed from data.

    Returns:
        SSIM value in range [-1, 1].
    """
    if len(model_a) != len(model_b):
        min_len = min(len(model_a), len(model_b))
        model_a = model_a[:min_len]
        model_b = model_b[:min_len]

    if len(model_a) == 0:
        return 0.0

    # SSIM constants (standard values)
    if data_range is None:
        data_range = max(
            np.max(model_a) - np.min(model_a),
            np.max(model_b) - np.min(model_b),
        )

    if data_range == 0:
        return 1.0 if np.allclose(model_a, model_b) else 0.0

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    mu_a = np.mean(model_a)
    mu_b = np.mean(model_b)
    sigma_a = np.std(model_a)
    sigma_b = np.std(model_b)
    sigma_ab = np.mean((model_a - mu_a) * (model_b - mu_b))

    # SSIM formula
    numerator = (2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)
    denominator = (mu_a**2 + mu_b**2 + C1) * (sigma_a**2 + sigma_b**2 + C2)

    ssim = numerator / denominator
    return float(ssim)


def depth_weighted_rms(
    model_a: "np.ndarray",
    model_b: "np.ndarray",
    common_grid: dict[str, Any],
) -> dict[str, float]:
    """Compute depth-weighted RMS difference between models.

    Applies depth weighting to emphasize differences at different depth
    levels. Useful because geophysical resolution decreases with depth.

    Args:
        model_a: First model values on common grid.
        model_b: Second model values on common grid.
        common_grid: Grid specification with 'z' coordinates.

    Returns:
        Dictionary with:
            - 'total_rms': Overall RMS difference
            - 'shallow_rms': RMS in upper third
            - 'middle_rms': RMS in middle third
            - 'deep_rms': RMS in lower third
            - 'weighted_rms': Depth-weighted RMS
    """
    if len(model_a) != len(model_b):
        min_len = min(len(model_a), len(model_b))
        model_a = model_a[:min_len]
        model_b = model_b[:min_len]

    diff = model_a - model_b
    total_rms = float(np.sqrt(np.mean(diff**2)))

    # Get z-coordinates for each cell
    z_values = common_grid.get("z", np.array([0.0]))
    nx = len(common_grid.get("x", [1]))
    ny = len(common_grid.get("y", [1]))
    nz = len(z_values)

    if nz <= 1:
        return {
            "total_rms": total_rms,
            "shallow_rms": total_rms,
            "middle_rms": total_rms,
            "deep_rms": total_rms,
            "weighted_rms": total_rms,
        }

    # Split into depth thirds
    z_range = z_values[-1] - z_values[0]
    z_third = z_range / 3.0

    # Reshape to 3D if possible
    n_total = nx * ny * nz
    if len(diff) == n_total:
        diff_3d = diff.reshape(nx, ny, nz)

        shallow_idx = nz // 3
        middle_idx = 2 * nz // 3

        shallow_rms = float(np.sqrt(np.mean(diff_3d[:, :, middle_idx:]**2)))
        middle_rms = float(np.sqrt(np.mean(diff_3d[:, :, shallow_idx:middle_idx]**2)))
        deep_rms = float(np.sqrt(np.mean(diff_3d[:, :, :shallow_idx]**2)))

        # Depth weighting: emphasize shallow (where resolution is better)
        weights = np.zeros(nz)
        for i in range(nz):
            depth = z_values[-1] - z_values[i]  # Depth from surface
            weights[i] = 1.0 / (1.0 + depth / z_range) if z_range > 0 else 1.0

        weighted_diff = diff_3d * weights[np.newaxis, np.newaxis, :]
        weighted_rms = float(np.sqrt(np.mean(weighted_diff**2)))
    else:
        # Cannot reshape; compute flat statistics
        shallow_rms = total_rms
        middle_rms = total_rms
        deep_rms = total_rms
        weighted_rms = total_rms

    return {
        "total_rms": total_rms,
        "shallow_rms": shallow_rms,
        "middle_rms": middle_rms,
        "deep_rms": deep_rms,
        "weighted_rms": weighted_rms,
    }


def convergence_rate(misfit_history: list[float]) -> float:
    """Compute the average convergence rate from misfit history.

    Defined as the average fractional reduction in misfit per iteration.

    Args:
        misfit_history: List of misfit values per iteration.

    Returns:
        Average convergence rate (positive = converging).
        Returns 0.0 if insufficient data.
    """
    if len(misfit_history) < 2:
        return 0.0

    rates = []
    for i in range(1, len(misfit_history)):
        if misfit_history[i - 1] > 0:
            rate = (misfit_history[i - 1] - misfit_history[i]) / misfit_history[i - 1]
            rates.append(rate)

    return float(np.mean(rates)) if rates else 0.0


def misfit_reduction_rate(misfit_history: list[float]) -> float:
    """Compute total misfit reduction as a fraction of initial misfit.

    Args:
        misfit_history: List of misfit values per iteration.

    Returns:
        Fractional reduction: (initial - final) / initial.
        Returns 0.0 if insufficient data.
    """
    if len(misfit_history) < 2:
        return 0.0

    initial = misfit_history[0]
    final = misfit_history[-1]

    if initial == 0:
        return 0.0

    return float((initial - final) / initial)


def runtime_comparison(
    runtime_a: float,
    runtime_b: float,
) -> dict[str, float]:
    """Compare runtimes of two algorithms.

    Args:
        runtime_a: Runtime of first algorithm (seconds).
        runtime_b: Runtime of second algorithm (seconds).

    Returns:
        Dictionary with:
            - 'runtime_a': First runtime
            - 'runtime_b': Second runtime
            - 'speedup': How much faster A is than B (>1 means A is faster)
            - 'ratio': runtime_b / runtime_a
    """
    if runtime_a <= 0 or runtime_b <= 0:
        return {
            "runtime_a": runtime_a,
            "runtime_b": runtime_b,
            "speedup": 0.0,
            "ratio": 0.0,
        }

    return {
        "runtime_a": runtime_a,
        "runtime_b": runtime_b,
        "speedup": runtime_b / runtime_a,
        "ratio": runtime_b / runtime_a,
    }


def model_difference_statistics(
    model_a: "np.ndarray",
    model_b: "np.ndarray",
) -> dict[str, float]:
    """Compute comprehensive difference statistics.

    Args:
        model_a: First model values.
        model_b: Second model values.

    Returns:
        Dictionary with various difference statistics.
    """
    if len(model_a) != len(model_b):
        min_len = min(len(model_a), len(model_b))
        model_a = model_a[:min_len]
        model_b = model_b[:min_len]

    diff = model_a - model_b
    abs_diff = np.abs(diff)

    return {
        "mean_difference": float(np.mean(diff)),
        "median_difference": float(np.median(diff)),
        "max_absolute_difference": float(np.max(abs_diff)),
        "rms_difference": float(np.sqrt(np.mean(diff**2))),
        "normalized_rms": float(
            np.sqrt(np.mean(diff**2)) / max(np.std(model_a), np.std(model_b), 1e-10)
        ),
        "percent_similar_5pct": float(
            np.sum(abs_diff < 0.05 * max(np.max(abs_diff), 1e-10)) / len(diff) * 100
        ),
    }
