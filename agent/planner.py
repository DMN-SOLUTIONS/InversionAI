"""
WorkflowPlanner - Analyzes user intent and suggests optimal workflows.

Uses LLM to interpret ambiguous requests and map them to concrete
inversion workflows with appropriate algorithms and parameters.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class UserIntent:
    """Parsed user intent from natural language input."""

    task_type: str  # "inversion", "validation", "comparison", "visualization", "explanation"
    data_type: Optional[str] = None  # "gravity", "magnetic", "joint"
    algorithms: list[str] = field(default_factory=list)
    comparison_requested: bool = False
    parameters: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0


@dataclass
class WorkflowSuggestion:
    """A suggested workflow based on user intent and available data."""

    workflow_name: str
    steps: list[str]
    algorithm: str
    estimated_runtime: str
    explanation: str
    alternatives: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)


_RUNTIME_ESTIMATES = {
    "tomofast-x": {
        "small": "~30 seconds",
        "medium": "~2-5 minutes",
        "large": "~10-30 minutes",
        "xlarge": "~1-3 hours",
    },
    "simpeg": {
        "small": "~1-2 minutes",
        "medium": "~5-15 minutes",
        "large": "~30-90 minutes",
        "xlarge": "~3-8 hours",
    },
}


class WorkflowPlanner:
    """
    Interprets user messages to determine intent and suggests
    appropriate inversion workflows.
    """

    def __init__(self) -> None:
        self._llm = None
        logger.info("WorkflowPlanner initialized.")

    def _get_llm(self):
        """Lazy-load LLM for intent analysis of ambiguous requests."""
        if self._llm is not None:
            return self._llm

        provider = os.getenv("INVERSION_AI_LLM_PROVIDER", "anthropic").lower()
        model = os.getenv("INVERSION_AI_LLM_MODEL")
        temperature = float(os.getenv("INVERSION_AI_LLM_TEMPERATURE", "0.1"))

        if provider == "openai":
            from langchain_openai import ChatOpenAI
            model = model or "gpt-4o"
            self._llm = ChatOpenAI(model=model, temperature=temperature)
        elif provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            model = model or "claude-sonnet-4-20250514"
            self._llm = ChatAnthropic(model=model, temperature=temperature)
        else:
            raise ValueError(f"Unsupported LLM provider: '{provider}'")

        return self._llm

    async def analyze_intent(self, message: str) -> UserIntent:
        """
        Analyze a user message to determine their intent.

        First tries pattern-based matching. Falls back to LLM for ambiguous messages.
        """
        logger.debug("Analyzing intent for: %s", message[:80])

        intent = self._pattern_analysis(message)
        if intent.confidence >= 0.8:
            logger.info("Intent resolved via pattern matching: %s", intent.task_type)
            return intent

        try:
            intent = await self._llm_analysis(message)
            logger.info("Intent resolved via LLM: %s", intent.task_type)
            return intent
        except Exception as e:
            logger.warning("LLM analysis failed: %s. Using pattern result.", str(e))
            return intent

    def _pattern_analysis(self, message: str) -> UserIntent:
        """Fast pattern-based intent analysis for common requests."""
        lower = message.lower()

        task_type = "unknown"
        if any(w in lower for w in ["validate", "check", "verify"]):
            task_type = "validation"
        elif any(w in lower for w in ["compare", "comparison", "difference", "vs"]):
            task_type = "comparison"
        elif any(w in lower for w in ["invert", "inversion", "run", "execute", "compute"]):
            task_type = "inversion"
        elif any(w in lower for w in ["visualize", "plot", "show", "display", "view"]):
            task_type = "visualization"
        elif any(w in lower for w in ["explain", "what is", "help", "describe", "meaning"]):
            task_type = "explanation"
        elif any(w in lower for w in ["list", "available", "skills", "workflows"]):
            task_type = "listing"

        data_type = None
        if any(w in lower for w in ["gravity", "grav", "gz"]):
            data_type = "gravity"
        elif any(w in lower for w in ["magnetic", "mag", "tmi"]):
            data_type = "magnetic"
        elif any(w in lower for w in ["joint", "both gravity and magnetic"]):
            data_type = "joint"

        algorithms = []
        if any(w in lower for w in ["tomofast", "tomofast-x", "tomofastx"]):
            algorithms.append("tomofast-x")
        if any(w in lower for w in ["simpeg", "sim-peg"]):
            algorithms.append("simpeg")

        comparison_requested = task_type == "comparison" or len(algorithms) > 1

        confidence = 0.5
        if task_type != "unknown":
            confidence += 0.3
        if data_type or algorithms:
            confidence += 0.2

        return UserIntent(
            task_type=task_type,
            data_type=data_type,
            algorithms=algorithms,
            comparison_requested=comparison_requested,
            confidence=min(confidence, 1.0),
        )

    async def _llm_analysis(self, message: str) -> UserIntent:
        """Use LLM to analyze ambiguous user intent."""
        llm = self._get_llm()

        prompt = (
            "Analyze this user message about geophysical inversion and extract their intent.\n"
            "Return a JSON object with these fields:\n"
            '- task_type: one of "inversion", "validation", "comparison", "visualization", "explanation", "listing"\n'
            '- data_type: one of "gravity", "magnetic", "joint", or null\n'
            '- algorithms: list (options: "tomofast-x", "simpeg"), empty if none\n'
            "- comparison_requested: boolean\n"
            "- parameters: dict of any specific parameters mentioned\n"
            "- confidence: float 0-1\n\n"
            f'User message: "{message}"\n\n'
            "Respond ONLY with the JSON object."
        )

        response = await llm.ainvoke(prompt)
        content = response.content.strip()

        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1])

        parsed = json.loads(content)

        return UserIntent(
            task_type=parsed.get("task_type", "unknown"),
            data_type=parsed.get("data_type"),
            algorithms=parsed.get("algorithms", []),
            comparison_requested=parsed.get("comparison_requested", False),
            parameters=parsed.get("parameters", {}),
            confidence=parsed.get("confidence", 0.7),
        )

    def suggest_workflow(
        self, intent: UserIntent, available_data: Optional[dict[str, Any]] = None
    ) -> WorkflowSuggestion:
        """Suggest an optimal workflow based on user intent and available data."""
        available_data = available_data or {}

        if intent.algorithms:
            algorithm = intent.algorithms[0]
        elif intent.data_type == "joint":
            algorithm = "tomofast-x"
        else:
            algorithm = "tomofast-x"

        if intent.task_type == "inversion":
            steps = self._inversion_steps(intent, algorithm)
            workflow_name = f"{algorithm}_inversion"
        elif intent.task_type == "comparison":
            steps = self._comparison_steps()
            workflow_name = "algorithm_comparison"
            algorithm = "both"
        elif intent.task_type == "validation":
            steps = ["Load and inspect data", "Check format and dimensions",
                     "Validate coordinates", "Report quality metrics"]
            workflow_name = "data_validation"
        elif intent.task_type == "visualization":
            steps = ["Load results", "Generate 3D visualization",
                     "Create cross-sections", "Export figures"]
            workflow_name = "result_visualization"
        else:
            steps = ["Interpret request", "Provide explanation or guidance"]
            workflow_name = "assistance"

        data_size = available_data.get("num_cells", 50000)
        estimated_runtime = self.estimate_runtime(algorithm, data_size)

        alternatives = []
        if algorithm == "tomofast-x" and intent.task_type == "inversion":
            alternatives.append("Use SimPEG for more flexibility in regularization.")
        elif algorithm == "simpeg" and intent.task_type == "inversion":
            alternatives.append("Use Tomofast-x for faster computation on large models.")
        if not intent.comparison_requested and intent.task_type == "inversion":
            alternatives.append("Run both algorithms and compare results.")

        explanation = self._build_explanation(algorithm)

        return WorkflowSuggestion(
            workflow_name=workflow_name,
            steps=steps,
            algorithm=algorithm,
            estimated_runtime=estimated_runtime,
            explanation=explanation,
            alternatives=alternatives,
            parameters=intent.parameters,
        )

    def _inversion_steps(self, intent: UserIntent, algorithm: str) -> list[str]:
        steps = [
            "Validate input data files",
            "Generate mesh/model discretization",
            f"Configure {algorithm} parameters",
            f"Run {algorithm} inversion",
            "Monitor convergence",
            "Load and summarize results",
        ]
        if intent.data_type == "joint":
            steps.insert(3, "Configure joint inversion coupling")
        return steps

    def _comparison_steps(self) -> list[str]:
        return [
            "Validate input data files",
            "Run Tomofast-x inversion",
            "Run SimPEG inversion",
            "Compute difference metrics (RMS, correlation, structural similarity)",
            "Generate comparison visualizations",
            "Summarize findings and recommendations",
        ]

    def _build_explanation(self, algorithm: str) -> str:
        if algorithm == "both":
            return (
                "Running both Tomofast-x and SimPEG allows comparison. "
                "Consistent features across algorithms are more likely real geological structures."
            )
        elif algorithm == "tomofast-x":
            return (
                "Tomofast-x is recommended for speed and efficiency with large models. "
                "It supports MPI parallelism and joint gravity-magnetic inversion."
            )
        elif algorithm == "simpeg":
            return (
                "SimPEG is recommended for flexibility — custom regularization, "
                "multiple mesh types, and ideal for prototyping."
            )
        return "Workflow selected based on your request."

    def estimate_runtime(self, workflow_or_algorithm: str, data_size: int) -> str:
        """Estimate runtime for a given algorithm and data size (number of model cells)."""
        if data_size < 10_000:
            size_cat = "small"
        elif data_size < 100_000:
            size_cat = "medium"
        elif data_size < 1_000_000:
            size_cat = "large"
        else:
            size_cat = "xlarge"

        algo = workflow_or_algorithm.lower().replace("-", "").replace("_", "")
        if "tomofast" in algo:
            key = "tomofast-x"
        elif "simpeg" in algo:
            key = "simpeg"
        elif "both" in algo or "comparison" in algo:
            tf = _RUNTIME_ESTIMATES["tomofast-x"].get(size_cat, "unknown")
            sp = _RUNTIME_ESTIMATES["simpeg"].get(size_cat, "unknown")
            return f"Tomofast-x: {tf}, SimPEG: {sp} (sequential)"
        else:
            return "Unable to estimate — unknown algorithm."

        return _RUNTIME_ESTIMATES.get(key, {}).get(size_cat, "Unable to estimate.")
