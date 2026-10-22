"""Web search via the Tavily API."""

from typing import Protocol

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
