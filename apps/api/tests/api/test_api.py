import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from langgraph.checkpoint.memory import InMemorySaver

from cwas.api.app import create_app
from cwas.graph.builder import build_graph
from cwas.models import Brief
from cwas.runs import RunManager
from tests.fakes import FakeLLM, FakeScraper, FakeSearch, make_brief, make_deps

PAGES = {"https://facts.org/a": "Home solar panels cost about 2.50 dollars per watt installed."}
SEARCH = {"how do solar panels work": ["https://facts.org/a"]}
ARTICLE = "# Solar\n\nPanels cost 2.50 dollars per watt [1]."


def manager_for(
    llm: FakeLLM, search: FakeSearch | None = None, cleaned: list[str] | None = None
) -> RunManager:
    deps = make_deps(llm=llm, search=search or FakeSearch(SEARCH), scraper=FakeScraper(PAGES))

    async def cleanup(run_id: str) -> None:
        if cleaned is not None:
            cleaned.append(run_id)

    return RunManager(build_graph(deps, checkpointer=InMemorySaver()), cleanup=cleanup)


@pytest.fixture
def happy_llm() -> FakeLLM:
    return FakeLLM(structured={Brief: [make_brief()]}, texts=[ARTICLE])


async def client_for(manager: RunManager) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=create_app(manager=manager))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
async def client(happy_llm: FakeLLM) -> AsyncIterator[httpx.AsyncClient]:
    async for c in client_for(manager_for(happy_llm)):
        yield c


async def wait_for_status(client: httpx.AsyncClient, run_id: str, *statuses: str) -> dict[str, Any]:
    for _ in range(300):
        body: dict[str, Any] = (await client.get(f"/runs/{run_id}")).json()
        if body["status"] in statuses:
            return body
        await asyncio.sleep(0.01)
    raise AssertionError(f"run never reached {statuses}: {body}")


async def test_health(client: httpx.AsyncClient) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_create_run_completes_with_cited_article(client: httpx.AsyncClient) -> None:
    created = await client.post("/runs", json={"prompt": "Write about home solar panels"})

    assert created.status_code == 202
    run = await wait_for_status(client, created.json()["run_id"], "completed", "failed")
    assert run["status"] == "completed"
    assert "2.50 dollars per watt [1]" in run["article"]
    assert run["brief"]["primary_keyword"] == "home solar panels"
    assert run["error"] is None
    assert 0 <= run["ai_score"]["score"] < 10
    assert set(run["ai_score"]["signals"]) >= {"stock_phrases", "uniform_rhythm"}


@pytest.mark.parametrize("payload", [{"prompt": ""}, {"prompt": "  "}, {}, {"prompt": "x" * 20001}])
async def test_create_run_rejects_invalid_prompts(
    client: httpx.AsyncClient, payload: dict[str, str]
) -> None:
    assert (await client.post("/runs", json=payload)).status_code == 422


@pytest.mark.parametrize(
    ("method", "path"),
    [("GET", "/runs/nope"), ("GET", "/runs/nope/stream"), ("POST", "/runs/nope/answer")],
)
async def test_unknown_run_is_404(client: httpx.AsyncClient, method: str, path: str) -> None:
    kwargs = {"json": {"answer": "x"}} if method == "POST" else {}
    assert (await client.request(method, path, **kwargs)).status_code == 404


async def read_sse(client: httpx.AsyncClient, run_id: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    async with client.stream("GET", f"/runs/{run_id}/stream") as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        async for line in response.aiter_lines():
            if line.startswith("data:"):
                events.append(json.loads(line.removeprefix("data:").strip()))
                if events[-1]["type"] in ("done", "error"):
                    break
    return events


async def test_stream_replays_and_ends_with_done(client: httpx.AsyncClient) -> None:
    run_id = (await client.post("/runs", json={"prompt": "solar panels"})).json()["run_id"]

    events = await asyncio.wait_for(read_sse(client, run_id), timeout=5)

    types = [e["type"] for e in events]
    assert types[0] == "run_started"
    assert types[-1] == "done"
    finished = [e["agent"] for e in events if e["type"] == "agent_finished"]
    assert finished == ["prompt_enhancer", "researcher", "kb_builder", "writer", "grammar"]
    assert [e["seq"] for e in events] == list(range(len(events)))


async def test_clarifying_question_then_answer_completes_the_run() -> None:
    unclear = make_brief(confidence=0.1, clarifying_question="Which country?")
    llm = FakeLLM(structured={Brief: [unclear, make_brief()]}, texts=[ARTICLE])
    async for client in client_for(manager_for(llm)):
        run_id = (await client.post("/runs", json={"prompt": "solar?"})).json()["run_id"]

        waiting = await wait_for_status(client, run_id, "waiting_for_input")
        assert waiting["question"] == "Which country?"

        answered = await client.post(f"/runs/{run_id}/answer", json={"answer": "India"})
        assert answered.status_code == 202

        done = await wait_for_status(client, run_id, "completed", "failed")
        assert done["status"] == "completed"
        assert done["question"] is None
        assert "India" in llm.calls[1]["prompt"]


async def test_answer_when_not_waiting_is_409(client: httpx.AsyncClient) -> None:
    run_id = (await client.post("/runs", json={"prompt": "solar"})).json()["run_id"]
    await wait_for_status(client, run_id, "completed")

    response = await client.post(f"/runs/{run_id}/answer", json={"answer": "hi"})

    assert response.status_code == 409


async def test_failed_run_reports_error_and_stream_ends_with_error() -> None:
    llm = FakeLLM(structured={Brief: [make_brief()]})
    cleaned: list[str] = []
    async for client in client_for(manager_for(llm, search=FakeSearch({}), cleaned=cleaned)):
        run_id = (await client.post("/runs", json={"prompt": "solar"})).json()["run_id"]

        run = await wait_for_status(client, run_id, "failed", "completed")
        events = await asyncio.wait_for(read_sse(client, run_id), timeout=5)

        assert run["status"] == "failed"
        assert "no web pages" in run["error"]
        assert events[-1]["type"] == "error"
        assert cleaned == [run_id]


async def test_cleanup_runs_after_completion_but_not_while_waiting() -> None:
    unclear = make_brief(confidence=0.1, clarifying_question="Which country?")
    llm = FakeLLM(structured={Brief: [unclear, make_brief()]}, texts=[ARTICLE])
    cleaned: list[str] = []
    async for client in client_for(manager_for(llm, cleaned=cleaned)):
        run_id = (await client.post("/runs", json={"prompt": "solar"})).json()["run_id"]
        await wait_for_status(client, run_id, "waiting_for_input")
        assert cleaned == []

        await client.post(f"/runs/{run_id}/answer", json={"answer": "UK"})
        await wait_for_status(client, run_id, "completed")
        assert cleaned == [run_id]
