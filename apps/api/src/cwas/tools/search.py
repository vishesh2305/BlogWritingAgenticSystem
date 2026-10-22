"""Web search: Tavily (API key) or DuckDuckGo (free, no key)."""

import asyncio
from collections.abc import Callable
from typing import Any, Protocol

import httpx

from cwas.models import SearchResult


class SearchError(Exception):
    pass


class SearchClient(Protocol):
    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]: ...


class TavilySearch:
    def __init__(
        self, client: httpx.AsyncClient, api_key: str, base_url: str = "https://api.tavily.com"
    ) -> None:
        self._client = client
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        try:
            response = await self._client.post(
                f"{self._base_url}/search",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={"query": query, "max_results": max_results, "search_depth": "advanced"},
                timeout=30.0,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise SearchError(f"Tavily search failed for {query!r}: {exc}") from exc

        return [
            SearchResult(
                url=item["url"],
                title=item.get("title", ""),
                snippet=item.get("content", ""),
                score=float(item.get("score", 0.0)),
            )
            for item in response.json().get("results", [])
        ]


DuckDuckGoBackend = Callable[[str, int], list[dict[str, Any]]]


def _ddgs_text(query: str, max_results: int) -> list[dict[str, Any]]:
    from ddgs import DDGS

    return DDGS().text(query, max_results=max_results)


class DuckDuckGoSearch:
    """Keyless search. Rate-limited by DuckDuckGo, so prefer Tavily for heavy use."""

    def __init__(self, backend: DuckDuckGoBackend = _ddgs_text) -> None:
        self._backend = backend

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        try:
            rows = await asyncio.to_thread(self._backend, query, max_results)
        except Exception as exc:  # the ddgs library raises a variety of error types
            raise SearchError(f"DuckDuckGo search failed for {query!r}: {exc}") from exc
        return [
            SearchResult(url=row["href"], title=row.get("title", ""), snippet=row.get("body", ""))
            for row in rows
            if row.get("href")
        ]
