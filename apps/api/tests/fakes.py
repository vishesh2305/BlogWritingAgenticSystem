"""Test doubles for the agent dependencies."""

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel

from cwas.agents.deps import AgentDeps
from cwas.config import Settings
from cwas.kb.knowledge_base import KnowledgeBase
from cwas.kb.store import InMemoryVectorStore
from cwas.models import Brief, ScrapedPage, SearchResult, SourceType
from cwas.tools.embed import HashingEmbedder
from cwas.tools.grammar import GrammarMatch
from cwas.tools.search import SearchError


class FakeLLM:
    """Returns scripted outputs. Lists are consumed in order; the last item repeats."""

    def __init__(
        self,
        structured: dict[type[BaseModel], list[BaseModel]] | None = None,
        texts: list[str] | None = None,
    ) -> None:
        self._structured = structured or {}
        self._texts = texts or ["Draft text."]
        self.calls: list[dict[str, Any]] = []

    async def structured(self, *, tier: str, system: str, prompt: str, schema: type, effort: str = "medium") -> Any:
        self.calls.append({"kind": "structured", "tier": tier, "system": system, "prompt": prompt, "schema": schema})
        queue = self._structured[schema]
        return queue.pop(0) if len(queue) > 1 else queue[0]

    async def text(
        self, *, tier: str, system: str, prompt: str, effort: str = "medium", max_tokens: int = 16000
    ) -> str:
        self.calls.append({"kind": "text", "tier": tier, "system": system, "prompt": prompt})
        return self._texts.pop(0) if len(self._texts) > 1 else self._texts[0]


class FakeSearch:
    def __init__(self, results: dict[str, list[str]], failing: set[str] | None = None) -> None:
        self._results = results
        self._failing = failing or set()
        self.queries: list[str] = []

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        self.queries.append(query)
        if query in self._failing:
            raise SearchError(f"failed {query}")
        urls = self._results.get(query, [])[:max_results]
        return [SearchResult(url=u, title=u) for u in urls]


class FakeScraper:
    def __init__(self, pages: dict[str, str]) -> None:
        self._pages = pages
        self.fetched: list[str] = []

    async def fetch(self, url: str, source_type: SourceType = "source") -> ScrapedPage | None:
        self.fetched.append(url)
        if url not in self._pages:
            return None
        return ScrapedPage(
            url=url,
            title=f"Title of {url}",
            text=self._pages[url],
            source_type=source_type,
            fetched_at=datetime.now(UTC),
        )


class FakeGrammar:
    def __init__(self, matches: Callable[[str], list[GrammarMatch]] | Exception) -> None:
        self._matches = matches

    async def check(self, text: str) -> list[GrammarMatch]:
        if isinstance(self._matches, Exception):
            raise self._matches
        return self._matches(text)


def make_brief(**overrides: Any) -> Brief:
    values: dict[str, Any] = {
        "refined_prompt": "Write a beginner guide to home solar panels.",
        "topic": "home solar panels",
        "audience": "homeowners",
        "tone": "friendly",
        "content_format": "how-to guide",
        "primary_keyword": "home solar panels",
        "secondary_keywords": ["solar cost"],
        "search_queries": ["how do solar panels work", "solar panel cost"],
        "confidence": 0.9,
    }
    values.update(overrides)
    return Brief(**values)


def make_settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


def make_deps(
    llm: FakeLLM | None = None,
    search: FakeSearch | None = None,
    scraper: FakeScraper | None = None,
    grammar: FakeGrammar | None = None,
    **settings: Any,
) -> AgentDeps:
    return AgentDeps(
        llm=llm or FakeLLM(),
        search=search or FakeSearch({}),
        scraper=scraper or FakeScraper({}),
        kb=KnowledgeBase(HashingEmbedder(dimensions=256), InMemoryVectorStore(), max_chars=400),
        grammar=grammar,
        settings=make_settings(**settings),
    )
