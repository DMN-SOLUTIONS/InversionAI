# 🧲 InversionAI

A skills-based geophysical inversion framework that makes running and comparing inversion algorithms accessible to non-experts.

![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776ab?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.45-ff4b4b?logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ed?logo=docker&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-Agents-1c3c3c?logo=langchain&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🌟 Overview

InversionAI bridges the gap between powerful geophysical inversion codes and researchers who want to use them without deep expertise in each tool's configuration. It wraps **Tomofast-x** and **SimPEG** behind a unified, AI-guided interface powered by LLM agents with a modular "skills" architecture.

Instead of learning each algorithm's parameter file format, data conventions, and execution workflow, you describe your problem in natural language and InversionAI handles the rest — configuring, running, and comparing results across multiple inversion engines.

![InversionAI Architecture](./docs/images/architecture.png)

---

## ✨ Features

### 🧩 Skills Architecture

Each capability is encapsulated as a **skill** — a self-contained, composable unit that an agent can invoke:

- **Data Validation Skill** — Checks input data formats, dimensions, and physical plausibility
- **Configuration Skill** — Generates algorithm-specific parameter files from high-level descriptions
- **Execution Skill** — Runs inversions with progress monitoring and error recovery
- **Visualization Skill** — Produces 3D models, cross-sections, and convergence plots
- **Comparison Skill** — Side-by-side analysis of results from different algorithms

### 🔄 Comparison Workflow

Run the same inversion problem through both Tomofast-x and SimPEG, then compare:

- Model misfit and data fit metrics
- Recovered model differences (RMS, structural similarity)
- Convergence behaviour
- Computational performance

### 🎯 Guided Configuration

The LLM agent walks you through setup step-by-step:

1. Upload or describe your survey data
2. Define the mesh/discretization
3. Set inversion parameters (regularization, stopping criteria)
4. Choose one or both algorithms
5. Review generated configs before execution

---

## 🚀 Quickstart with Docker

```bash
# Clone the repository
git clone https://github.com/your-org/InversionAI.git
cd InversionAI

# Copy environment template and add your API key
cp .env.example .env
# Edit .env with your ANTHROPIC_API_KEY or OPENAI_API_KEY

# Build and run
docker compose up --build

# Open in browser
open http://localhost:8501
```

The Docker image includes a pre-built Tomofast-x binary — no Fortran compiler needed on your host machine.

---

## 📁 Project Structure

```
InversionAI/
├── ui/                        # Streamlit application
│   ├── app.py                 # Main entry point
│   ├── pages/                 # Multi-page Streamlit views
│   └── components/            # Reusable UI widgets
├── agents/                    # LangGraph agent definitions
│   ├── orchestrator.py        # Top-level agent router
│   ├── inversion_agent.py     # Inversion execution agent
│   └── comparison_agent.py    # Cross-algorithm comparison agent
├── skills/                    # Composable skill modules
│   ├── data_validation.py
│   ├── config_generation.py
│   ├── execution.py
│   ├── visualization.py
│   └── comparison.py
├── engines/                   # Algorithm-specific adapters
│   ├── tomofast/              # Tomofast-x wrapper
│   └── simpeg/                # SimPEG wrapper
├── data/                      # Sample datasets
├── output/                    # Inversion results (gitignored)
├── docs/                      # Documentation and images
├── .streamlit/                # Streamlit configuration
│   └── config.toml
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## ⚙️ Supported Algorithms

| Algorithm | Type | Physics | Method | Status |
|-----------|------|---------|--------|--------|
| **Tomofast-x** | Deterministic | Gravity, Magnetics | Parallel iterative (LSQR, PCGLS) | ✅ Supported |
| **SimPEG** | Deterministic | Gravity, Magnetics, DC/IP, EM | Gauss-Newton, IRLS | ✅ Supported |

### Planned

| Algorithm | Type | Physics | Status |
|-----------|------|---------|--------|
| **GemPy** | Probabilistic | Structural geology | 🔜 Planned |
| **pyGIMLi** | Deterministic | ERT, seismic | 🔜 Planned |

---

## 🛠️ Local Development (without Docker)

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your keys

# Run the app
streamlit run ui/app.py
```

> **Note:** Running Tomofast-x locally requires `gfortran` and `OpenMPI`. On macOS ARM, use the Docker approach instead.

---

## 🤝 Contributing

Contributions are welcome! Whether it's adding a new inversion engine, improving the UI, or writing documentation.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-engine`)
3. Write tests for new skills or adapters
4. Commit your changes (`git commit -m 'Add new engine adapter'`)
5. Push to the branch (`git push origin feature/new-engine`)
6. Open a Pull Request

### Adding a New Skill

Skills follow a simple interface:

```python
class MySkill:
    name: str = "my_skill"
    description: str = "What this skill does"

    def validate_inputs(self, inputs: dict) -> bool: ...
    def execute(self, inputs: dict) -> dict: ...
```

### Adding a New Engine

Engine adapters implement the `InversionEngine` protocol:

```python
class MyEngine:
    def generate_config(self, params: dict) -> Path: ...
    def run(self, config: Path) -> InversionResult: ...
    def parse_results(self, output_dir: Path) -> Model: ...
```

---

## 📝 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Tomofast-x](https://github.com/TOMOFAST/Tomofast-x) — Parallel geophysical inversion code
- [SimPEG](https://simpeg.xyz/) — Simulation and Parameter Estimation in Geophysics
- [LangGraph](https://github.com/langchain-ai/langgraph) — Agent orchestration framework
- [Anthropic Claude](https://www.anthropic.com/) — LLM backbone

---

**Made with 🧲 by the InversionAI Team**

*Making geophysical inversion accessible, one skill at a time.*
