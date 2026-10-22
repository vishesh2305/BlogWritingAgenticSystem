from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from cwas.agents.researcher import ResearchError
from cwas.graph.builder import build_graph
from cwas.graph.finalize import render_article
from cwas.models import Brief, Citation, Draft
from tests.fakes import FakeLLM, FakeScraper, FakeSearch, make_brief, make_deps

PAGES = {
    "https://facts.org/a": "Home solar panels cost about 2.50 dollars per watt installed.",
    "https://rival.com/blog": "Our blog: home solar panels are a great investment for families.",
}
SEARCH = {
    "how do solar panels work": ["https://facts.org/a"],
    "solar panel cost": ["https://facts.org/a"],
    "home solar panels": ["https://rival.com/blog"],
}
ARTICLE = "# Home Solar Panels\n\nPanels cost about 2.50 dollars per watt [1]."


def deps_with(llm: FakeLLM) -> Any:
    return make_deps(llm=llm, search=FakeSearch(SEARCH), scraper=FakeScraper(PAGES))


async def run_to_end(
    graph: Any, payload: Any, thread: str
) -> tuple[list[dict[str, Any]], list[Any]]:
    events: list[dict[str, Any]] = []
    interrupts: list[Any] = []
    config = {"configurable": {"thread_id": thread}}
    async for mode, chunk in graph.astream(payload, config, stream_mode=["custom", "updates"]):
        if mode == "custom":
            events.append(chunk)
        elif "__interrupt__" in chunk:
            interrupts.extend(chunk["__interrupt__"])
    return events, interrupts


async def test_happy_path_produces_cited_article_and_agent_events() -> None:
    llm = FakeLLM(structured={Brief: [make_brief()]}, texts=[ARTICLE])
    graph = build_graph(deps_with(llm), checkpointer=InMemorySaver())

    events, interrupts = await run_to_end(graph, {"run_id": "r1", "user_prompt": "solar"}, "t1")

    assert interrupts == []
    state = (await graph.aget_state({"configurable": {"thread_id": "t1"}})).values
    assert "2.50 dollars per watt [1]" in state["final_article"]
    assert "## Sources" in state["final_article"]
    assert "https://facts.org/a" in state["final_article"]

    finished = [e["agent"] for e in events if e["type"] == "agent_finished"]
    assert finished == ["prompt_enhancer", "researcher", "kb_builder", "writer", "grammar"]
    started = [e["agent"] for e in events if e["type"] == "agent_started"]
    assert started == finished
    assert all(e["duration_ms"] >= 0 for e in events if e["type"] == "agent_finished")


async def test_ambiguous_prompt_pauses_for_a_question_then_resumes() -> None:
    unclear = make_brief(confidence=0.2, clarifying_question="Which country are you in?")
    llm = FakeLLM(structured={Brief: [unclear, make_brief()]}, texts=[ARTICLE])
    graph = build_graph(deps_with(llm), checkpointer=InMemorySaver())

    _, interrupts = await run_to_end(graph, {"run_id": "r2", "user_prompt": "solar?"}, "t2")

    assert [i.value for i in interrupts] == [{"question": "Which country are you in?"}]

    _, more = await run_to_end(graph, Command(resume="United Kingdom"), "t2")

    assert more == []
    state = (await graph.aget_state({"configurable": {"thread_id": "t2"}})).values
    assert state["final_article"]
    second_enhancer_prompt = [c for c in llm.calls if c["kind"] == "structured"][1]["prompt"]
    assert "United Kingdom" in second_enhancer_prompt


async def test_agent_failure_emits_failed_event_and_raises() -> None:
    llm = FakeLLM(structured={Brief: [make_brief()]})
    deps = make_deps(llm=llm, search=FakeSearch({}), scraper=FakeScraper({}))
    graph = build_graph(deps, checkpointer=InMemorySaver())
    events: list[dict[str, Any]] = []

    with pytest.raises(ResearchError):
        async for mode, chunk in graph.astream(
            {"run_id": "r3", "user_prompt": "x"},
            {"configurable": {"thread_id": "t3"}},
            stream_mode=["custom", "updates"],
        ):
            if mode == "custom":
                events.append(chunk)

    assert events[-1]["type"] == "agent_failed"
    assert events[-1]["agent"] == "researcher"


def test_render_article_appends_numbered_sources() -> None:
    draft = Draft(
        variant_id="v1",
        markdown="Body [2] and [1].",
        citations=[
            Citation(number=2, url="https://b.com", title="B"),
            Citation(number=1, url="https://a.com", title="A"),
        ],
    )

    article = render_article(draft)

    assert article.startswith("Body [2] and [1].")
    assert article.endswith("## Sources\n\n1. [A](https://a.com)\n2. [B](https://b.com)\n")


def test_render_article_without_citations_has_no_sources_section() -> None:
    assert render_article(Draft(variant_id="v1", markdown="Body.")) == "Body.\n"
