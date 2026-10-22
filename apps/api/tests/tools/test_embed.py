import json
import math

import httpx
import pytest
import respx

from cwas.tools.embed import EmbeddingError, HashingEmbedder, VoyageEmbedder

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
