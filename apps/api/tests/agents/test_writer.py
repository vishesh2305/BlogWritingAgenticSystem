from datetime import UTC, datetime

import pytest

from cwas.agents.writer import WriterError, writer
from cwas.models import ScrapedPage
from tests.fakes import FakeLLM, make_brief, make_deps


def page(url: str, text: str) -> ScrapedPage:
    return ScrapedPage(url=url, title=f"T {url}", text=text, fetched_at=datetime.now(UTC))


async def seeded_deps(llm: FakeLLM):  # type: ignore[no-untyped-def]
    deps = make_deps(llm=llm)
    await deps.kb.ingest(
        "r",
        [
            page("https://a.com", "Home solar panels cost about 2.50 dollars per watt installed."),
            page("https://b.com", "Home solar panels last 25 years with minimal maintenance."),
        ],
    )
    return deps


async def test_prompt_contains_brief_and_untrusted_sources_and_uses_opus() -> None:
    llm = FakeLLM(texts=["# Guide\n\nPanels cost money [1]."])
    deps = await seeded_deps(llm)

    await writer({"run_id": "r", "brief": make_brief()}, deps)

    call = llm.calls[0]
    assert call["tier"] == "opus"
    assert "Write a beginner guide to home solar panels." in call["prompt"]
    assert "<untrusted_sources>" in call["prompt"]
    assert "2.50 dollars per watt" in call["prompt"]
    assert 'url="https://a.com"' in call["prompt"]


async def test_draft_keeps_valid_citations_and_strips_invented_ones() -> None:
    llm = FakeLLM(texts=["Cost is low [1]. Life is long [2]. Fake claim [9]."])
    deps = await seeded_deps(llm)

    update = await writer({"run_id": "r", "brief": make_brief()}, deps)

    [draft] = update["drafts"]
    assert draft.variant_id == "v1"
    assert "[9]" not in draft.markdown
    assert "Fake claim." in draft.markdown
    assert {c.number for c in draft.citations} == {1, 2}
    assert {c.url for c in draft.citations} == {"https://a.com", "https://b.com"}


async def test_unused_sources_are_not_listed_as_citations() -> None:
    llm = FakeLLM(texts=["Only one claim [2]."])
    deps = await seeded_deps(llm)

    update = await writer({"run_id": "r", "brief": make_brief()}, deps)

    assert [c.number for c in update["drafts"][0].citations] == [2]


async def test_chunks_from_the_same_page_share_one_source_number() -> None:
    llm = FakeLLM(texts=["x [1]"])
    deps = make_deps(llm=llm)
    long_text = "\n\n".join(f"Solar fact number {i} about home solar panels." for i in range(30))
    await deps.kb.ingest("r", [page("https://one.com", long_text)])

    await writer({"run_id": "r", "brief": make_brief()}, deps)

    prompt = llm.calls[0]["prompt"]
    assert prompt.count('id="1"') >= 2
    assert 'id="2"' not in prompt


async def test_raises_when_knowledge_base_is_empty() -> None:
    with pytest.raises(WriterError):
        await writer({"run_id": "empty", "brief": make_brief()}, make_deps())


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Cheap [n1]. Durable [n2].", "Cheap [1]. Durable [2]."),
        ("Cheap [source 1].", "Cheap [1]."),
        ("Cheap [Source 2].", "Cheap [2]."),
        ("Cheap [1, 2].", "Cheap [1][2]."),
        ("Cheap [n1, n9].", "Cheap [1]."),
        # gpt-oss (the Groq fallback) cites in its own format; seen in a live run.
        ("Cheap【1†source】. Durable【2†L5-L9】.", "Cheap [1]. Durable [2]."),
        ("Cheap【1†source】【2†source】.", "Cheap [1][2]."),
        ("Cheap【source】. Durable【9†source】.", "Cheap. Durable."),
    ],
)
async def test_citation_variants_from_small_models_are_normalised(raw: str, expected: str) -> None:
    deps = await seeded_deps(FakeLLM(texts=[raw]))

    update = await writer({"run_id": "r", "brief": make_brief()}, deps)

    assert update["drafts"][0].markdown == expected


async def test_markdown_links_and_checkboxes_are_not_treated_as_citations() -> None:
    text = "See [the guide](https://x.io) and [x] done [1]."
    deps = await seeded_deps(FakeLLM(texts=[text]))

    update = await writer({"run_id": "r", "brief": make_brief()}, deps)

    assert update["drafts"][0].markdown == text
