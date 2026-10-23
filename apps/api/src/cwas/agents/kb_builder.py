"""Agent 3 - Knowledge Base Builder: chunks, embeds and stores the scraped pages, and pulls out
the numeric facts later agents may cite."""

from typing import Any

from cwas.agents.deps import AgentDeps
from cwas.graph.state import RunState
from cwas.kb.facts import extract_facts


async def kb_builder(state: RunState, deps: AgentDeps) -> dict[str, Any]:
    sources = state["sources"]
    return {
        "kb_chunks": await deps.kb.ingest(state["run_id"], sources),
        "facts": extract_facts(sources, deps.settings.max_facts_per_page),
    }
