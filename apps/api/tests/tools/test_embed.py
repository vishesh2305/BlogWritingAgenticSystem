import json
import math

import httpx
import pytest
import respx

from cwas.tools.embed import EmbeddingError, GeminiEmbedder, HashingEmbedder, VoyageEmbedder

VOYAGE_URL = "https://api.voyageai.com/v1/embeddings"


def voyage_reply(request: httpx.Request) -> httpx.Response:
    texts = json.loads(request.content)["input"]
    # Return out of order to prove results are re-sorted by index.
    data = [{"index": i, "embedding": [float(len(t)), float(i)]} for i, t in enumerate(texts)]
    return httpx.Response(200, json={"data": list(reversed(data))})


@respx.mock
async def test_voyage_returns_vectors_in_input_order() -> None:
    route = respx.post(VOYAGE_URL).mock(side_effect=voyage_reply)
    async with httpx.AsyncClient() as client:
        vectors = await VoyageEmbedder(client, api_key="pa-key").embed(["a", "bbb"], "query")

    assert vectors == [[1.0, 0.0], [3.0, 1.0]]
    body = json.loads(route.calls.last.request.content)
    assert body["input_type"] == "query"
    assert body["model"] == "voyage-3.5"
    assert route.calls.last.request.headers["Authorization"] == "Bearer pa-key"


@respx.mock
async def test_voyage_splits_large_inputs_into_batches() -> None:
    route = respx.post(VOYAGE_URL).mock(side_effect=voyage_reply)
    texts = [f"t{i}" for i in range(5)]
    async with httpx.AsyncClient() as client:
        vectors = await VoyageEmbedder(client, api_key="k", batch_size=2).embed(texts, "document")

    assert route.call_count == 3
    assert len(vectors) == 5


async def test_voyage_empty_input_makes_no_request() -> None:
    async with httpx.AsyncClient() as client:
        assert await VoyageEmbedder(client, api_key="k").embed([], "document") == []


@respx.mock
async def test_voyage_raises_embedding_error_on_failure() -> None:
    respx.post(VOYAGE_URL).mock(return_value=httpx.Response(429, text="slow down"))
    async with httpx.AsyncClient() as client:
        with pytest.raises(EmbeddingError):
            await VoyageEmbedder(client, api_key="k").embed(["x"], "document")


async def test_hashing_embedder_is_deterministic_and_normalised() -> None:
    embedder = HashingEmbedder(dimensions=64)
    [a, b] = await embedder.embed(["solar panels power", "solar panels power"], "document")
    assert a == b
    assert len(a) == 64
    assert math.isclose(math.sqrt(sum(x * x for x in a)), 1.0, rel_tol=1e-9)


async def test_hashing_embedder_ranks_overlapping_text_closer() -> None:
    embedder = HashingEmbedder(dimensions=256)
    [q, near, far] = await embedder.embed(
        ["solar panel efficiency", "solar panel efficiency ratings", "chocolate cake recipe"],
        "document",
    )

    def cos(x: list[float], y: list[float]) -> float:
        return sum(i * j for i, j in zip(x, y, strict=True))

    assert cos(q, near) > cos(q, far)


async def test_hashing_embedder_handles_empty_text() -> None:
    [v] = await HashingEmbedder(dimensions=8).embed([""], "query")
    assert v == [0.0] * 8


GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:batchEmbedContents"
)


def gemini_reply(request: httpx.Request) -> httpx.Response:
    texts = [r["content"]["parts"][0]["text"] for r in json.loads(request.content)["requests"]]
    return httpx.Response(200, json={"embeddings": [{"values": [float(len(t))]} for t in texts]})


@respx.mock
async def test_gemini_embeds_documents_with_a_title_prefix_and_key_header() -> None:
    route = respx.post(GEMINI_URL).mock(side_effect=gemini_reply)
    async with httpx.AsyncClient() as client:
        vectors = await GeminiEmbedder(client, api_key="AIza-k").embed(["ab", "c"], "document")

    request = route.calls.last.request
    assert request.headers["x-goog-api-key"] == "AIza-k"
    assert "AIza-k" not in str(request.url)
    [first, _] = json.loads(request.content)["requests"]
    assert first == {
        "model": "models/gemini-embedding-2",
        "content": {"parts": [{"text": "title: none | text: ab"}]},
        "outputDimensionality": 768,
    }
    assert vectors == [
        [float(len("title: none | text: ab"))],
        [float(len("title: none | text: c"))],
    ]


@respx.mock
async def test_gemini_embeds_queries_with_a_search_prefix() -> None:
    route = respx.post(GEMINI_URL).mock(side_effect=gemini_reply)
    async with httpx.AsyncClient() as client:
        await GeminiEmbedder(client, api_key="k", dimensions=1536).embed(["solar cost"], "query")

    [request] = json.loads(route.calls.last.request.content)["requests"]
    assert request["content"]["parts"][0]["text"] == "task: search result | query: solar cost"
    assert request["outputDimensionality"] == 1536


@respx.mock
async def test_gemini_splits_large_inputs_into_batches() -> None:
    route = respx.post(GEMINI_URL).mock(side_effect=gemini_reply)
    async with httpx.AsyncClient() as client:
        vectors = await GeminiEmbedder(client, api_key="k", batch_size=2).embed(
            [f"t{i}" for i in range(5)], "document"
        )

    assert route.call_count == 3
    assert len(vectors) == 5


async def test_gemini_empty_input_makes_no_request() -> None:
    async with httpx.AsyncClient() as client:
        assert await GeminiEmbedder(client, api_key="k").embed([], "query") == []


@respx.mock
async def test_gemini_http_error_raises_embedding_error_without_the_key() -> None:
    respx.post(GEMINI_URL).mock(return_value=httpx.Response(429, json={"error": "quota"}))
    async with httpx.AsyncClient() as client:
        with pytest.raises(EmbeddingError, match="Gemini") as caught:
            await GeminiEmbedder(client, api_key="AIza-secret").embed(["a"], "query")

    assert "AIza-secret" not in str(caught.value)


@respx.mock
async def test_gemini_wrong_vector_count_raises_embedding_error() -> None:
    respx.post(GEMINI_URL).mock(return_value=httpx.Response(200, json={"embeddings": []}))
    async with httpx.AsyncClient() as client:
        with pytest.raises(EmbeddingError, match="1 texts"):
            await GeminiEmbedder(client, api_key="k").embed(["a"], "query")


def quota_exceeded(delay: str) -> httpx.Response:
    """Gemini's real free-tier 429: no Retry-After header, the delay is in RetryInfo."""
    return httpx.Response(
        429,
        json={
            "error": {
                "code": 429,
                "message": "Embed content requests per minute limit exceeded on Free Tier.",
                "details": [
                    {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": delay}
                ],
            }
        },
    )


class Sleeps(list[float]):
    async def __call__(self, seconds: float) -> None:
        self.append(seconds)


@respx.mock
async def test_gemini_waits_out_a_free_tier_rate_limit_and_retries() -> None:
    route = respx.post(GEMINI_URL).mock(side_effect=[quota_exceeded("3s"), gemini_reply])
    sleeps = Sleeps()
    async with httpx.AsyncClient() as client:
        vectors = await GeminiEmbedder(client, api_key="k", sleep=sleeps).embed(["a"], "query")

    assert sleeps == [3.0]
    assert route.call_count == 2
    assert len(vectors) == 1


@respx.mock
async def test_gemini_gives_up_after_max_retries() -> None:
    route = respx.post(GEMINI_URL).mock(return_value=quota_exceeded("2s"))
    sleeps = Sleeps()
    async with httpx.AsyncClient() as client:
        embedder = GeminiEmbedder(client, api_key="k", max_retries=2, sleep=sleeps)
        with pytest.raises(EmbeddingError, match="429"):
            await embedder.embed(["a"], "query")

    assert sleeps == [2.0, 2.0]
    assert route.call_count == 3


@respx.mock
async def test_gemini_does_not_wait_out_a_long_rate_limit() -> None:
    respx.post(GEMINI_URL).mock(return_value=quota_exceeded("3600s"))
    sleeps = Sleeps()
    async with httpx.AsyncClient() as client:
        with pytest.raises(EmbeddingError):
            await GeminiEmbedder(client, api_key="k", sleep=sleeps).embed(["a"], "query")

    assert sleeps == []
