"""Agent 2 - Web Research & Scraper: finds and fetches fact sources and ranking competitors."""

import asyncio
from typing import Any

from cwas.agents.deps import AgentDeps
from cwas.graph.state import RunState
from cwas.models import ScrapedPage, SearchResult, SourceType
from cwas.tools.search import SearchError


class ResearchError(Exception):
    pass


async def researcher(state: RunState, deps: AgentDeps) -> dict[str, Any]:
    settings = deps.settings
    brief = state["brief"]
    queries = brief.search_queries[: settings.max_search_queries] or [brief.topic]
    warnings: list[str] = []

    async def search(query: str, max_results: int) -> list[SearchResult]:
        try:
            return await deps.search.search(query, max_results)
        except SearchError as exc:
            warnings.append(f"search failed for {query!r}: {exc}")
            return []

    *per_query, ranking = await asyncio.gather(
        *(search(q, settings.results_per_query) for q in queries),
        search(brief.primary_keyword, settings.competitor_results),
    )

    # Pages that rank for the primary keyword are the competitors we must beat.
    competitors = list(dict.fromkeys(r.url for r in ranking))
    sources = [
        url
        for url in dict.fromkeys(r.url for results in per_query for r in results)
        if url not in competitors
    ][: settings.max_sources]

    semaphore = asyncio.Semaphore(settings.scrape_concurrency)

    async def fetch(url: str, source_type: SourceType) -> ScrapedPage | None:
        async with semaphore:
            return await deps.scraper.fetch(url, source_type)

    fetched = await asyncio.gather(
        *(fetch(url, "competitor") for url in competitors),
        *(fetch(url, "source") for url in sources),
    )
    pages = [page for page in fetched if page is not None]
    if not pages:
        raise ResearchError(f"no web pages could be collected for {brief.topic!r}")
    if not any(page.source_type == "competitor" for page in pages):
        warnings.append(
            f"no competitor page could be collected for {brief.primary_keyword!r}; "
            "the article can't be compared with the blogs that rank for it"
        )
    return {"sources": pages, "warnings": warnings}
