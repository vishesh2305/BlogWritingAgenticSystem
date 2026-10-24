from datetime import UTC, datetime

import pytest

from cwas.kb.knowledge_base import KnowledgeBase
from cwas.kb.store import InMemoryVectorStore
from cwas.models import ScrapedPage
from cwas.tools.embed import EmbeddingError, HashingEmbedder, InputType


def page(url: str, text: str, source_type: str = "source") -> ScrapedPage:
    return ScrapedPage(
        url=url,
        title=url,
        text=text,
        source_type=source_type,  # type: ignore[arg-type]
        fetched_at=datetime.now(UTC),
    )


def make_kb() -> KnowledgeBase:
    return KnowledgeBase(HashingEmbedder(dimensions=512), InMemoryVectorStore(), max_chars=300)


async def test_ingest_chunks_and_stores_every_page() -> None:
    kb = make_kb()
    long_text = "\n\n".join(f"Paragraph {i} about heat pumps and insulation." for i in range(20))

    stored = await kb.ingest(
        "run", [page("https://a.com", long_text), page("https://b.com", "Short.")]
    )

    assert stored > 2


async def test_retrieve_returns_the_most_relevant_page_first() -> None:
    kb = make_kb()
    await kb.ingest(
        "run",
        [
            page("https://bake.com", "Chocolate cake needs flour, sugar, cocoa and eggs."),
            page("https://sun.com", "Solar panel efficiency depends on photovoltaic cell quality."),
        ],
    )

    hits = await kb.retrieve("run", "photovoltaic solar panel efficiency", k=1)

    assert hits[0].chunk.source_url == "https://sun.com"


async def test_retrieve_can_be_limited_to_competitor_pages() -> None:
    kb = make_kb()
    await kb.ingest(
        "run",
        [
            page("https://fact.com", "Solar facts."),
            page("https://rival.com", "Solar blog post.", source_type="competitor"),
        ],
    )

    hits = await kb.retrieve("run", "solar", k=5, source_type="competitor")

    assert {h.chunk.source_url for h in hits} == {"https://rival.com"}


async def test_ingest_of_no_pages_stores_nothing() -> None:
    assert await make_kb().ingest("run", []) == 0


class FailingEmbedder:
    async def embed(self, texts: list[str], input_type: InputType) -> list[list[float]]:
        raise EmbeddingError("Gemini embedding failed: 429 Too Many Requests")


SOLAR = "Rooftop solar panels convert sunlight into electricity for homes."
HEAT = "Heat pumps move warmth from outside air into the house in winter."


async def test_a_failing_embedder_falls_back_to_the_offline_one_for_that_collection() -> None:
    kb = KnowledgeBase(FailingEmbedder(), InMemoryVectorStore(), fallback=HashingEmbedder(256))

    stored = await kb.ingest("run", [page("https://s.com", SOLAR), page("https://h.com", HEAT)])
    hits = await kb.retrieve("run", "solar panels sunlight", k=1)

    assert stored == 2
    assert kb.uses_fallback("run")
    assert not kb.uses_fallback("other-run")
    assert hits[0].chunk.source_url == "https://s.com"

    await kb.drop("run")
    assert not kb.uses_fallback("run")


async def test_without_a_fallback_embedding_errors_propagate() -> None:
    kb = KnowledgeBase(FailingEmbedder(), InMemoryVectorStore())

    with pytest.raises(EmbeddingError):
        await kb.ingest("run", [page("https://s.com", SOLAR)])
