"""HTTP API: start runs, follow their agent timeline over SSE, answer clarifying questions."""

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, StringConstraints
from sse_starlette import EventSourceResponse

from cwas.config import Settings
from cwas.runs import Run, RunManager, RunNotWaitingError, RunStatus

Prompt = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=20000)]
Answer = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]


class RunCreate(BaseModel):
    prompt: Prompt


class RunCreated(BaseModel):
    run_id: str


class AnswerIn(BaseModel):
    answer: Answer


class RunView(BaseModel):
    id: str
    status: RunStatus
    prompt: str
    question: str | None
    article: str | None
    brief: dict[str, Any] | None
    warnings: list[str]
    error: str | None


def create_app(manager: RunManager | None = None, settings: Settings | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if manager is not None:
            app.state.runs = manager
            yield
            return
        from cwas.wiring import runtime

        async with runtime(settings or Settings()) as built:
            app.state.runs = built
            yield

    app = FastAPI(title="Content Writing Multi-Agent System", lifespan=lifespan)
    if manager is not None:
        app.state.runs = manager  # available even when the lifespan is not run (e.g. ASGI tests)

    def runs(request: Request) -> RunManager:
        return request.app.state.runs  # type: ignore[no-any-return]

    def find(request: Request, run_id: str) -> Run:
        run = runs(request).get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")
        return run

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/runs", status_code=202)
    async def create_run(body: RunCreate, request: Request) -> RunCreated:
        run = await runs(request).start(body.prompt)
        return RunCreated(run_id=run.id)

    @app.get("/runs/{run_id}")
    async def get_run(run_id: str, request: Request) -> RunView:
        run = find(request, run_id)
        return RunView(
            id=run.id,
            status=run.status,
            prompt=run.prompt,
            question=run.question,
            article=run.article,
            brief=run.brief,
            warnings=run.warnings,
            error=run.error,
        )

    @app.get("/runs/{run_id}/stream")
    async def stream_run(run_id: str, request: Request) -> EventSourceResponse:
        run = find(request, run_id)

        async def events() -> AsyncIterator[dict[str, Any]]:
            async for event in run.stream():
                yield {"id": str(event["seq"]), "data": json.dumps(event)}

        return EventSourceResponse(events())

    @app.post("/runs/{run_id}/answer", status_code=202)
    async def answer_run(run_id: str, body: AnswerIn, request: Request) -> dict[str, str]:
        run = find(request, run_id)
        try:
            await runs(request).answer(run, body.answer)
        except RunNotWaitingError as exc:
            raise HTTPException(status_code=409, detail="run is not waiting for input") from exc
        return {"status": "running"}

    return app
