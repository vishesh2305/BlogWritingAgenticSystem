"""Orchestrator step: pick the delivered draft and render it with its sources."""

from typing import Any

from cwas.graph.state import RunState
from cwas.models import Draft


def render_article(draft: Draft) -> str:
    article = draft.markdown.rstrip() + "\n"
    if not draft.citations:
        return article
    lines = [
        f"{c.number}. [{c.title.replace(']', r'\]')}]({c.url})"
        for c in sorted(draft.citations, key=lambda c: c.number)
    ]
    return article + "\n## Sources\n\n" + "\n".join(lines) + "\n"


async def finalize(state: RunState) -> dict[str, Any]:
    # Single variant until the Benchmark Judge (Phase 5) chooses between several.
    return {"final_article": render_article(state["drafts"][0])}
