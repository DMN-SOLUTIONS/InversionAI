# Architecture Overview

InversionAI uses a **skills-based agent architecture** inspired by the Darkmine/Anthropic pattern, where specialized capabilities are modular, discoverable, and orchestrated by an AI agent.

## The Skills Pattern

A **skill** is a self-contained module that encapsulates a specific capability (e.g., running a Tomofast-X inversion). Each skill:

- Implements a standard interface (`BaseSkill`)
- Declares its capabilities via a `SKILL.md` file
- Handles its own validation, configuration, and execution
- Is independently testable

This pattern enables:
- **Extensibility** — new algorithms added without touching core code
- **Discovery** — the agent reads `SKILL.md` files to understand what's available
- **Composability** — skills are combined into workflows
- **Isolation** — a failing skill doesn't crash the system

### Skill Interface

```python
class BaseSkill(ABC):
    name: str
    description: str

    @abstractmethod
    def validate(self, data_path: Path) -> ValidationResult: ...

    @abstractmethod
    def configure(self, run_config: RunConfig) -> RunConfig: ...

    @abstractmethod
    def run(self, run_config: RunConfig) -> InversionResult: ...
```

---

## How Skills Are Discovered and Orchestrated

### Discovery

On startup, the system scans the `skills/` directory for `SKILL.md` files:

```
skills/
├── __init__.py          # Registry of available skills
├── base.py              # BaseSkill ABC + dataclasses
├── tomofast/
│   ├── __init__.py
│   ├── skill.py         # TomofastSkill implementation
│   └── SKILL.md         # Agent-readable capability description
├── simpeg/
│   ├── __init__.py
│   ├── skill.py         # SimpegSkill implementation
│   └── SKILL.md
└── compare/
    ├── __init__.py
    ├── skill.py         # CompareSkill implementation
    └── SKILL.md
```

The agent reads each `SKILL.md` to understand:
- What data types the skill supports
- What parameters it accepts
- When to use it vs. alternatives

### Orchestration

The AI agent selects and sequences skills based on the user's request:

1. **Parse intent** — What does the user want? (gravity inversion, comparison, etc.)
2. **Select skills** — Which skills are needed? (validate → invert → compare)
3. **Plan workflow** — What order? What parameters?
4. **Execute** — Run each skill, passing outputs forward
5. **Report** — Present results to the user

---

## Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                          USER REQUEST                                │
│         "Run gravity inversion with Tomofast and SimPEG"            │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        AI AGENT                                      │
│  • Parses intent                                                     │
│  • Reads SKILL.md files                                              │
│  • Generates execution plan                                          │
└────────────┬──────────────────────────────────┬─────────────────────┘
             │                                  │
             ▼                                  ▼
┌────────────────────────┐       ┌────────────────────────────┐
│   DATA VALIDATION      │       │   DATA VALIDATION          │
│   (TomofastSkill)      │       │   (SimpegSkill)            │
└────────────┬───────────┘       └──────────────┬─────────────┘
             │ ✓ valid                           │ ✓ valid
             ▼                                   ▼
┌────────────────────────┐       ┌────────────────────────────┐
│   CONFIGURE            │       │   CONFIGURE                │
│   • Generate Parfile   │       │   • Build TensorMesh       │
│   • Format data        │       │   • Set regularization     │
└────────────┬───────────┘       └──────────────┬─────────────┘
             │                                   │
             ▼                                   ▼
┌────────────────────────┐       ┌────────────────────────────┐
│   RUN INVERSION        │       │   RUN INVERSION            │
│   • Call binary        │       │   • SimPEG optimization    │
│   • Monitor progress   │       │   • In-process execution   │
└────────────┬───────────┘       └──────────────┬─────────────┘
             │                                   │
             │    InversionResult A              │  InversionResult B
             └──────────────┬───────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      COMPARE SKILL                                   │
│  • Normalize models                                                  │
│  • Compute RMSE, correlation, structural similarity                  │
│  • Generate difference map                                           │
│  • Create comparison report                                          │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      RESULTS TO USER                                 │
│  • Convergence plots                                                 │
│  • 3D model visualizations                                           │
│  • Metrics summary table                                             │
│  • Recommendations                                                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Agent Decision Flow

The agent follows a ReAct (Reasoning + Acting) loop:

```
THINK → What do I need to do next?
  │
  ├── User wants gravity inversion
  ├── Two algorithms requested: tomofast, simpeg
  ├── Need to validate data first
  │
ACT → Call skill.validate(data_path)
  │
OBSERVE → ValidationResult(is_valid=True)
  │
THINK → Data is valid, proceed with inversions
  │
ACT → Call tomofast_skill.configure(config)
  │     Call tomofast_skill.run(config)
  │
OBSERVE → InversionResult(converged=True, misfit=1.02)
  │
THINK → Tomofast done, now run SimPEG
  │
ACT → Call simpeg_skill.configure(config)
  │     Call simpeg_skill.run(config)
  │
OBSERVE → InversionResult(converged=True, misfit=1.05)
  │
THINK → Both done, compare results
  │
ACT → Call compare_skill.compare(result_a, result_b)
  │
OBSERVE → Metrics(rmse=0.03, correlation=0.95)
  │
RESPOND → Present results to user
```

### Error Recovery

If a skill fails, the agent:
1. Logs the error with context
2. Determines if the workflow can continue (other algorithms may still run)
3. Reports partial results if available
4. Suggests remediation steps to the user

---

## Comparison Methodology

When multiple algorithms produce results for the same dataset, the `CompareSkill` evaluates agreement:

### Normalization

Before comparison, models are normalized to account for different parameterizations:

- **Min-max scaling** to [0, 1] for visual comparison
- **Z-score normalization** for statistical metrics
- **Spatial alignment** to ensure cell-to-cell correspondence

### Metrics

| Metric | Description | Ideal Value |
|--------|-------------|-------------|
| RMSE | Root mean square difference between models | 0 |
| Correlation | Pearson correlation coefficient | 1.0 |
| Structural Similarity | SSIM-inspired metric for spatial structure | 1.0 |
| Misfit Difference | |φ_d(A) - φ_d(B)| | 0 |
| Runtime Ratio | Time(A) / Time(B) | — |

### Interpretation

- **High correlation + low RMSE** → Algorithms agree, result is robust
- **High correlation + high RMSE** → Same structure, different amplitudes (scaling issue)
- **Low correlation** → Algorithms found different solutions; review regularization or data fit

### When Results Disagree

The agent considers:
1. Did both algorithms converge? (Check misfit ≈ 1.0)
2. Is one over-fitting? (misfit << 1.0 suggests noise fitting)
3. Are regularization parameters comparable?
4. Would a different mesh resolution help?
