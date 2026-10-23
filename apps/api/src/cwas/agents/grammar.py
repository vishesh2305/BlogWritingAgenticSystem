"""Agent 10 - Grammar & Readability: deterministic fixes, then readability and AI-likeness
scores of the final text. Runs last."""

from typing import Any

from cwas.agents.deps import AgentDeps
from cwas.graph.state import RunState
from cwas.models import Draft
from cwas.tools.ai_score import ai_score
from cwas.tools.grammar import GrammarError, apply_corrections, readability


async def grammar_agent(state: RunState, deps: AgentDeps) -> dict[str, Any]:
    warnings: list[str] = []
    drafts: list[Draft] = []
    for draft in state["drafts"]:
        text = draft.markdown
        if deps.grammar is not None:
            try:
                text = apply_corrections(text, await deps.grammar.check(text))
            except GrammarError as exc:
                warnings.append(f"LanguageTool unavailable; grammar not auto-corrected: {exc}")
        drafts.append(
            draft.model_copy(
                update={
                    "markdown": text,
                    "readability": readability(text),
                    "ai_score": ai_score(text),
                }
            )
        )
    return {"drafts": drafts, "warnings": warnings}
