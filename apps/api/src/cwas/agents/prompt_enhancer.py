"""Agent 1 - Prompt Enhancer: turns the raw request into a structured Brief."""

from typing import Any

from langgraph.types import interrupt

from cwas.agents import prompts
from cwas.agents.deps import AgentDeps
from cwas.config import Settings
from cwas.graph.state import RunState
from cwas.models import Brief


async def prompt_enhancer(state: RunState, deps: AgentDeps) -> dict[str, Any]:
    parts = [f"<user_request>\n{state['user_prompt']}\n</user_request>"]
    if clarifications := state.get("clarifications"):
        parts.append("<clarifications>\n" + "\n\n".join(clarifications) + "\n</clarifications>")

    brief = await deps.llm.structured(
        tier="sonnet",
        system=prompts.PROMPT_ENHANCER,
        prompt="\n\n".join(parts),
        schema=Brief,
    )
    return {"brief": brief}


def needs_clarification(state: RunState, settings: Settings) -> bool:
    brief = state["brief"]
    return (
        bool(brief.clarifying_question)
        and brief.confidence < settings.clarification_threshold
        and state.get("clarification_rounds", 0) < settings.max_clarification_rounds
    )


async def clarify(state: RunState) -> dict[str, Any]:
    """Pause the run until the user answers the brief's clarifying question."""
    question = state["brief"].clarifying_question
    answer = interrupt({"question": question})
    return {
        "clarifications": [*state.get("clarifications", []), f"Q: {question}\nA: {answer}"],
        "clarification_rounds": state.get("clarification_rounds", 0) + 1,
    }
