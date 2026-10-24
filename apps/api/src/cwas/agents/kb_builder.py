"""Agent 3 - Knowledge Base Builder: chunks, embeds and stores the scraped pages, and pulls out
the numeric facts later agents may cite."""

from typing import Any

from cwas.agents.deps import AgentDeps
from cwas.graph.state import RunState
from cwas.kb.facts import extract_facts


async def kb_builder(state: RunState, deps: AgentDeps) -> dict[str, Any]:
    sources = state["sources"]
    chunks = await deps.kb.ingest(state["run_id"], sources)
    warnings = []
    if deps.kb.uses_fallback(state["run_id"]):
        warnings.append(
            "embedding service failed (e.g. rate limit); this run's knowledge base used the "
            "offline hashing embedder, so retrieval is less precise"
        )
    return {
        "kb_chunks": chunks,
        "facts": extract_facts(sources, deps.settings.max_facts_per_page),
        "warnings": warnings,
    }
