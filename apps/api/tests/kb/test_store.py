from cwas.kb.store import InMemoryVectorStore
from cwas.models import Chunk


def chunk(cid: str, text: str, url: str = "https://a.com", source_type: str = "source") -> Chunk:
    return Chunk(
        id=cid,
        text=text,
        source_url=url,
        source_title="T",
        source_type=source_type,  # type: ignore[arg-type]
        index=0,
    )


async def test_search_returns_top_k_sorted_by_cosine_similarity() -> None:
    store = InMemoryVectorStore()
    await store.add(
        "run1",
        [chunk("a", "A"), chunk("b", "B"), chunk("c", "C")],
        [[1.0, 0.0], [0.7, 0.7], [0.0, 1.0]],
    )

    hits = await store.search("run1", [1.0, 0.0], k=2)

    assert [h.chunk.id for h in hits] == ["a", "b"]
    assert hits[0].score > hits[1].score


async def test_search_empty_collection_returns_nothing() -> None:
    assert await InMemoryVectorStore().search("missing", [1.0], k=3) == []


async def test_collections_are_isolated() -> None:
    store = InMemoryVectorStore()
    await store.add("run1", [chunk("a", "A")], [[1.0]])
    assert await store.search("run2", [1.0], k=5) == []


async def test_add_dedupes_identical_text_from_same_url() -> None:
    store = InMemoryVectorStore()
    added = await store.add("r", [chunk("a", "same"), chunk("b", "same")], [[1.0], [1.0]])
    again = await store.add("r", [chunk("c", "same")], [[1.0]])

    assert added == 1
    assert again == 0
    assert len(await store.search("r", [1.0], k=10)) == 1


async def test_search_can_filter_by_source_type() -> None:
    store = InMemoryVectorStore()
    await store.add(
        "r",
        [
            chunk("s", "fact", source_type="source"),
            chunk("c", "blog", "https://b.com", "competitor"),
        ],
        [[1.0, 0.0], [1.0, 0.0]],
    )

    hits = await store.search("r", [1.0, 0.0], k=5, source_type="competitor")

    assert [h.chunk.id for h in hits] == ["c"]


async def test_zero_vectors_do_not_crash_search() -> None:
    store = InMemoryVectorStore()
    await store.add("r", [chunk("z", "zero")], [[0.0, 0.0]])
    hits = await store.search("r", [1.0, 0.0], k=1)
    assert hits[0].score == 0.0


async def test_drop_removes_collection() -> None:
    store = InMemoryVectorStore()
    await store.add("r", [chunk("a", "A")], [[1.0]])
    await store.drop("r")
    assert await store.search("r", [1.0], k=1) == []
