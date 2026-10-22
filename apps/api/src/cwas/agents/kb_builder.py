"""Agent 3 - Knowledge Base Builder: chunks, embeds and stores the scraped pages."""

from typing import Any

from cwas.agents.deps import AgentDeps
from cwas.graph.state import RunState


async def kb_builder(state: RunState, deps: AgentDeps) -> dict[str, Any]:
    return {"kb_chunks": await deps.kb.ingest(state["run_id"], state["sources"])}
