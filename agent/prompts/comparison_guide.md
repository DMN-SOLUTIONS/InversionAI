# Comparison Results Interpretation Guide

Use this guide when explaining comparison results between Tomofast-x and SimPEG inversions to users.

## Key Metrics

### RMS Difference
- **What it measures:** Root-mean-square difference between the two recovered models, normalized by the model range.
- **Good agreement:** RMS < 5% of model range
- **Moderate agreement:** RMS 5-15% of model range
- **Poor agreement:** RMS > 15% of model range
- **Interpretation:** Lower is better. High RMS may indicate different convergence behavior or sensitivity to regularization choices.

### Correlation Coefficient
- **What it measures:** Linear correlation between model values at corresponding cells.
- **Good agreement:** r > 0.90
- **Moderate agreement:** 0.70 < r < 0.90
- **Poor agreement:** r < 0.70
- **Interpretation:** High correlation means structural similarity — the same features appear in both models even if amplitudes differ.

### Structural Similarity Index (SSIM)
- **What it measures:** Perceptual similarity considering luminance, contrast, and structure.
- **Good agreement:** SSIM > 0.85
- **Moderate agreement:** 0.60 < SSIM < 0.85
- **Poor agreement:** SSIM < 0.60
- **Interpretation:** SSIM captures whether the spatial patterns match, not just individual cell values.

### Maximum Difference Location
- **What it measures:** Where in the model the two algorithms disagree the most.
- **Interpretation:** Disagreements at depth are common (less data constraint). Disagreements at shallow depths or near strong anomalies may indicate algorithmic sensitivity.

---

## When Results Agree Well

**Tell the user:**
- Both algorithms recovered consistent features — this increases confidence in the result.
- The recovered structures likely represent real geological features.
- Minor amplitude differences are normal and expected due to different regularization approaches.

**Suggest:**
- Proceed with interpretation using either model.
- Use the Tomofast-x result for its better resolution (if larger mesh was used).
- Report both results if publishing.

---

## When Results Differ Moderately

**Tell the user:**
- The major structures are consistent, but details differ.
- This is common when regularization strength or type differs between codes.
- Features present in both models are more reliable than those in only one.

**Suggest:**
- Focus interpretation on features present in both models.
- Consider adjusting regularization to make them more comparable.
- Run sensitivity tests on the differing features.
- Check if the misfit achieved is similar — one may have converged better.

---

## When Results Differ Significantly

**Tell the user:**
- Significant differences suggest the problem may be poorly constrained or highly sensitive to algorithmic choices.
- This doesn't mean both are wrong — but neither can be fully trusted without additional information.
- The data itself may be insufficient to resolve the structure uniquely.

**Investigate:**
- Are both inversions fitting the data equally well (similar misfit)?
- Is one algorithm over-fitting (low misfit but complex model)?
- Could different depth weighting be causing the discrepancy?
- Is the mesh adequate for both algorithms?
- Are there data quality issues that affect one code differently?

**Suggest:**
- Add geological constraints (reference model, bounds) to reduce non-uniqueness.
- Try different regularization strengths and compare stability.
- Consider acquiring additional data if possible.
- Report the range of models as an uncertainty estimate.

---

## Common Causes of Disagreement

| Cause | Symptom | Fix |
|-------|---------|-----|
| Different regularization strength | Amplitude differs, structure similar | Match beta/alpha values |
| Different depth weighting | Shallow vs deep feature placement | Use same depth weighting exponent |
| Mesh resolution mismatch | One model is smoother | Use comparable cell sizes |
| Convergence issues | One has higher misfit | Increase iterations |
| Different starting models | Systematic offset | Use same reference model |
| Non-uniqueness | Both fit data well but look different | Add constraints or accept uncertainty |

---

## Reporting Template

When presenting comparison results, structure the explanation as:

1. **Summary:** "Both algorithms agree well / moderately / poorly on the recovered model."
2. **Metrics:** Present RMS, correlation, SSIM with interpretation.
3. **Consistent features:** "Both models show [describe features] — these are likely real."
4. **Differences:** "The models differ primarily in [location/depth/amplitude] — this may be due to [cause]."
5. **Recommendation:** Specific next steps based on the agreement level.
