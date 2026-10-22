import json

import httpx
import pytest
import respx

from cwas.tools.search import DuckDuckGoSearch, SearchError, TavilySearch

TAVILY_URL = "https://api.tavily.com/search"


@respx.mock
async def test_search_maps_results_and_sends_query() -> None:
    route = respx.post(TAVILY_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {"title": "A", "url": "https://a.com", "content": "alpha", "score": 0.9},
                    {"title": "B", "url": "https://b.com", "content": "beta", "score": 0.5},
                ]
            },
        )
    )
    async with httpx.AsyncClient() as client:
        results = await TavilySearch(client, api_key="tvly-test").search("python agents", 2)

    assert [r.url for r in results] == ["https://a.com", "https://b.com"]
    assert results[0].title == "A"
    assert results[0].snippet == "alpha"
    sent = route.calls.last.request
    assert sent.headers["Authorization"] == "Bearer tvly-test"
    body = json.loads(sent.content)
    assert body["query"] == "python agents"
    assert body["max_results"] == 2


@respx.mock
async def test_search_returns_empty_list_when_no_results() -> None:
    respx.post(TAVILY_URL).mock(return_value=httpx.Response(200, json={"results": []}))
    async with httpx.AsyncClient() as client:
        assert await TavilySearch(client, api_key="k").search("q") == []


@respx.mock
async def test_search_raises_search_error_on_http_failure() -> None:
    respx.post(TAVILY_URL).mock(return_value=httpx.Response(500, text="boom"))
    async with httpx.AsyncClient() as client:
        with pytest.raises(SearchError):
            await TavilySearch(client, api_key="k").search("q")


@respx.mock
async def test_search_raises_search_error_on_network_failure() -> None:
    respx.post(TAVILY_URL).mock(side_effect=httpx.ConnectError("down"))
    async with httpx.AsyncClient() as client:
        with pytest.raises(SearchError):
            await TavilySearch(client, api_key="k").search("q")


async def test_duckduckgo_maps_results_and_passes_limit() -> None:
    seen: dict[str, object] = {}

    def backend(query: str, max_results: int) -> list[dict[str, str]]:
        seen.update(query=query, max_results=max_results)
        return [
            {"title": "A", "href": "https://a.com", "body": "alpha"},
            {"title": "B", "href": "https://b.com", "body": "beta"},
        ]

    results = await DuckDuckGoSearch(backend).search("solar", 2)

    assert seen == {"query": "solar", "max_results": 2}
    assert [(r.url, r.title, r.snippet) for r in results] == [
        ("https://a.com", "A", "alpha"),
        ("https://b.com", "B", "beta"),
    ]


async def test_duckduckgo_skips_results_without_a_url() -> None:
    def backend(query: str, max_results: int) -> list[dict[str, str]]:
        return [{"title": "no link", "body": "x"}, {"title": "ok", "href": "https://ok.com"}]

    assert [r.url for r in await DuckDuckGoSearch(backend).search("q")] == ["https://ok.com"]


async def test_duckduckgo_failure_raises_search_error() -> None:
    def backend(query: str, max_results: int) -> list[dict[str, str]]:
        raise RuntimeError("rate limited")

    with pytest.raises(SearchError, match="rate limited"):
        await DuckDuckGoSearch(backend).search("q")
