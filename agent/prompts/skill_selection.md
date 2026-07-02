# Skill Selection Guide

Use this guide to choose the appropriate inversion algorithm based on the user's needs.

## Tomofast-x

**Best for:**
- Large models (>100k cells) where speed matters
- Production-quality inversions with proven convergence
- Joint gravity and magnetic inversion (unique capability)
- MPI-parallel execution on multi-core systems or clusters
- Reproducible results with well-tested Fortran numerics

**Characteristics:**
- Language: Fortran 2008 with MPI
- Parallelism: Full MPI support for distributed computation
- Mesh: Structured rectilinear grids
- Input format: Custom text-based parameter files
- Speed: ~5-10x faster than SimPEG for equivalent problems

**When to recommend:**
- User has large datasets or high-resolution meshes
- User needs joint inversion
- User wants fast turnaround
- User is running in production/batch mode

---

## SimPEG

**Best for:**
- Flexibility and custom regularization strategies
- Prototyping new inversion approaches
- Educational use and learning about inversions
- Problems requiring unstructured or OcTree meshes
- Integration with other Python scientific workflows

**Characteristics:**
- Language: Python with NumPy/SciPy
- Parallelism: Limited (single-node, some operations use NumPy threading)
- Mesh: Tensor, OcTree, unstructured
- Input format: Python API (programmatic)
- Speed: Moderate — sufficient for small-to-medium problems

**When to recommend:**
- User wants to experiment with parameters
- User needs custom objective functions
- User is learning about inversions
- Problem is small enough that speed isn't critical
- User needs OcTree mesh refinement

---

## Both (Comparison Mode)

**When to recommend running both algorithms:**
- User wants to validate results (features consistent across codes are more reliable)
- User is uncertain which algorithm suits their problem
- User is publishing results and wants to demonstrate robustness
- Results from one algorithm seem unexpected
- First time working with a new dataset

**Comparison workflow:**
1. Run Tomofast-x (faster — provides quick baseline)
2. Run SimPEG with equivalent parameters
3. Compute difference metrics
4. Identify consistent vs. divergent features
5. Report confidence levels based on agreement

---

## Decision Matrix

| Factor | Choose Tomofast-x | Choose SimPEG | Choose Both |
|--------|-------------------|---------------|-------------|
| Model size >100k cells | ✓ | | |
| Joint inversion needed | ✓ | | |
| Speed is critical | ✓ | | |
| Custom regularization | | ✓ | |
| OcTree mesh needed | | ✓ | |
| Learning/prototyping | | ✓ | |
| Validation needed | | | ✓ |
| Publishing results | | | ✓ |
| Unexpected results | | | ✓ |
