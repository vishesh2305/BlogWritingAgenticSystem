"""Agent 5 - Writer: drafts a cited article grounded in the knowledge base."""

import re
from html import escape
from typing import Any

from cwas.agents import prompts
from cwas.agents.deps import AgentDeps
from cwas.graph.state import RunState
from cwas.models import Brief, Citation, Draft

_CITATION = re.compile(r"\s?\[(\d+)\]")


class WriterError(Exception):
    pass


def _brief_block(brief: Brief) -> str:
    return (
        "<brief>\n"
        f"{brief.refined_prompt}\n"
        f"Audience: {brief.audience}\n"
        f"Tone: {brief.tone}\n"
        f"Format: {brief.content_format}\n"
        f"Primary keyword: {brief.primary_keyword}\n"
        f"Secondary keywords: {', '.join(brief.secondary_keywords)}\n"
        f"Target length: about {brief.target_word_count} words\n"
        "</brief>"
    )


async def writer(state: RunState, deps: AgentDeps) -> dict[str, Any]:
    brief = state["brief"]
    query = " ".join([brief.topic, brief.primary_keyword, *brief.secondary_keywords])
    hits = await deps.kb.retrieve(state["run_id"], query, k=deps.settings.retrieval_k)
    if not hits:
        raise WriterError("knowledge base is empty; nothing to ground the article in")

    numbers: dict[str, int] = {}
    titles: dict[str, str] = {}
    blocks: list[str] = []
    for hit in hits:
        url = hit.chunk.source_url
        number = numbers.setdefault(url, len(numbers) + 1)
        titles[url] = hit.chunk.source_title
        blocks.append(
            f'<source id="{number}" url="{escape(url)}" title="{escape(hit.chunk.source_title)}">\n'
            f"{hit.chunk.text}\n</source>"
        )

    prompt = (
        f"{_brief_block(brief)}\n\n"
        "<untrusted_sources>\n" + "\n".join(blocks) + "\n</untrusted_sources>\n\n"
        "Write the article described in the brief, citing the sources above as [n]."
    )
    markdown = await deps.llm.text(tier="opus", system=prompts.WRITER, prompt=prompt, effort="high")

    url_by_number = {n: url for url, n in numbers.items()}
    used: set[int] = set()

    def keep_valid(match: re.Match[str]) -> str:
        number = int(match.group(1))
        if number not in url_by_number:
            return ""
        used.add(number)
        return match.group(0)

    cleaned = _CITATION.sub(keep_valid, markdown).strip()
    citations = [
        Citation(number=n, url=url_by_number[n], title=titles[url_by_number[n]])
        for n in sorted(used)
    ]
    return {"drafts": [Draft(variant_id="v1", markdown=cleaned, citations=citations)]}
