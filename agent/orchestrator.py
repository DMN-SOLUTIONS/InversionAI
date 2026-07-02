"""
InversionOrchestrator - ReAct agent for geophysical inversion workflows.

Uses LangChain/LangGraph to create a ReAct agent that interprets user intent,
validates data, runs inversions, and explains results.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from agent.planner import WorkflowPlanner
from agent.tools import (
    compare_inversions,
    explain_parameter,
    get_run_status,
    list_available_skills,
    run_inversion,
    validate_data_file,
)

load_dotenv()

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"


@dataclass
class AgentResponse:
    """Response from the orchestrator agent."""

    message: str
    action_taken: Optional[str] = None
    results: dict[str, Any] = field(default_factory=dict)
    suggestions: list[str] = field(default_factory=list)


def _load_system_prompt() -> str:
    """Load the system prompt from the prompts directory."""
    system_prompt_path = PROMPTS_DIR / "system.md"
    if system_prompt_path.exists():
        return system_prompt_path.read_text(encoding="utf-8")
    logger.warning("System prompt not found at %s, using default.", system_prompt_path)
    return (
        "You are InversionAI, a geophysical inversion assistant. "
        "Help users run gravity and magnetic inversions using Tomofast-x and SimPEG."
    )


def _get_llm():
    """
    Create an LLM instance based on environment configuration.

    Supports INVERSION_AI_LLM_PROVIDER=anthropic (default) or openai.
    """
    provider = os.getenv("INVERSION_AI_LLM_PROVIDER", "anthropic").lower()
    model = os.getenv("INVERSION_AI_LLM_MODEL")
    temperature = float(os.getenv("INVERSION_AI_LLM_TEMPERATURE", "0.3"))

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        model = model or "gpt-4o"
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required.")
        return ChatOpenAI(model=model, temperature=temperature, api_key=api_key)

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        model = model or "claude-sonnet-4-20250514"
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required.")
        return ChatAnthropic(model=model, temperature=temperature, api_key=api_key)

    else:
        raise ValueError(
            f"Unsupported LLM provider: '{provider}'. "
            "Set INVERSION_AI_LLM_PROVIDER to 'anthropic' or 'openai'."
        )


class InversionOrchestrator:
    """
    Main orchestrator using a LangGraph ReAct agent to process
    user requests related to geophysical inversion.
    """

    def __init__(self) -> None:
        """Initialize the orchestrator with LLM, tools, and conversation state."""
        logger.info("Initializing InversionOrchestrator...")

        self._llm = _get_llm()
        self._system_prompt = _load_system_prompt()
        self._planner = WorkflowPlanner()

        self._tools = [
            list_available_skills,
            validate_data_file,
            run_inversion,
            compare_inversions,
            get_run_status,
            explain_parameter,
        ]

        self._agent = create_react_agent(
            model=self._llm,
            tools=self._tools,
            prompt=self._system_prompt,
        )

        self._conversation_history: list[HumanMessage | AIMessage | SystemMessage] = []

        logger.info(
            "InversionOrchestrator initialized with provider=%s, %d tools.",
            os.getenv("INVERSION_AI_LLM_PROVIDER", "anthropic"),
            len(self._tools),
        )

    @property
    def planner(self) -> WorkflowPlanner:
        """Access the workflow planner for intent analysis."""
        return self._planner

    async def process_request(
        self, user_message: str, context: Optional[dict[str, Any]] = None
    ) -> AgentResponse:
        """
        Process a user request through the ReAct agent.

        Args:
            user_message: The user's natural language request.
            context: Optional context dict (e.g., current data paths, active runs).

        Returns:
            AgentResponse with message, actions taken, results, and suggestions.
        """
        context = context or {}
        logger.info("Processing request: %s", user_message[:100])

        enriched_message = user_message
        if context:
            context_str = "\n".join(f"- {k}: {v}" for k, v in context.items())
            enriched_message = f"{user_message}\n\n[Current context]\n{context_str}"

        self._conversation_history.append(HumanMessage(content=enriched_message))

        try:
            result = await self._agent.ainvoke(
                {"messages": self._conversation_history}
            )

            ai_messages = [
                msg for msg in result.get("messages", []) if isinstance(msg, AIMessage)
            ]
            final_message = ai_messages[-1].content if ai_messages else "No response generated."

            self._conversation_history.append(AIMessage(content=final_message))

            actions_taken = self._extract_actions(result.get("messages", []))
            suggestions = self._generate_suggestions(user_message, actions_taken)

            response = AgentResponse(
                message=final_message,
                action_taken=actions_taken if actions_taken else None,
                results=self._extract_results(result.get("messages", [])),
                suggestions=suggestions,
            )
            logger.info("Request processed. Action: %s", response.action_taken)
            return response

        except Exception as e:
            logger.error("Error processing request: %s", str(e), exc_info=True)
            return AgentResponse(
                message=f"I encountered an error processing your request: {str(e)}",
                action_taken="error",
                results={"error": str(e)},
                suggestions=[
                    "Try rephrasing your request.",
                    "Check that your data files are accessible.",
                    "Verify your API key configuration.",
                ],
            )

    def _extract_actions(self, messages: list) -> str:
        """Extract a summary of tool calls made during the agent's reasoning."""
        tool_calls = []
        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append(tc.get("name", "unknown_tool"))
        return ", ".join(tool_calls) if tool_calls else ""

    def _extract_results(self, messages: list) -> dict[str, Any]:
        """Extract structured results from tool messages."""
        results: dict[str, Any] = {}
        for msg in messages:
            if hasattr(msg, "name") and hasattr(msg, "content") and msg.name:
                results[msg.name] = msg.content
        return results

    def _generate_suggestions(self, user_message: str, actions: str) -> list[str]:
        """Generate follow-up suggestions based on the interaction."""
        suggestions = []
        lower_msg = user_message.lower()

        if "validate" in actions or "validate" in lower_msg:
            suggestions.append("Run the inversion now that data is validated.")
        if "run_inversion" in actions:
            suggestions.append("Compare results with another algorithm.")
            suggestions.append("Visualize the inversion results.")
        if "compare" in actions:
            suggestions.append("Export comparison report as PDF.")
            suggestions.append("Try adjusting regularization parameters.")

        if not suggestions:
            suggestions = [
                "Validate your data files before running an inversion.",
                "Compare Tomofast-x and SimPEG results for validation.",
                "Ask me to explain any parameter or result.",
            ]
        return suggestions

    def reset_conversation(self) -> None:
        """Clear conversation history to start a fresh session."""
        self._conversation_history.clear()
        logger.info("Conversation history cleared.")

    @property
    def conversation_length(self) -> int:
        """Number of messages in the current conversation."""
        return len(self._conversation_history)
