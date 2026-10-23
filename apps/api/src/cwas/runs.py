"""Run lifecycle: drives graph executions in the background and fans out their events."""

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from langgraph.types import Command

from cwas.usage import UsageMeter, metering

logger = logging.getLogger(__name__)

RunStatus = Literal["running", "waiting_for_input", "completed", "failed"]
TERMINAL_EVENTS = frozenset({"done", "error"})

Cleanup = Callable[[str], Awaitable[None]]


class RunNotWaitingError(Exception):
    pass


@dataclass
class Run:
    id: str
    prompt: str
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    status: RunStatus = "running"
    question: str | None = None
    article: str | None = None
    brief: dict[str, Any] | None = None
    ai_score: dict[str, Any] | None = None
    meter: UsageMeter = field(default_factory=UsageMeter, repr=False)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    _changed: asyncio.Condition = field(default_factory=asyncio.Condition, repr=False)

    async def publish(self, event: dict[str, Any]) -> None:
        async with self._changed:
            self.events.append({**event, "seq": len(self.events)})
            self._changed.notify_all()

    async def stream(self) -> AsyncIterator[dict[str, Any]]:
        """Replay every event so far, then follow live events until the run ends."""
        position = 0
        while True:
            async with self._changed:
                while position >= len(self.events):
                    await self._changed.wait()
                batch = self.events[position:]
            position += len(batch)
            for event in batch:
                yield event
                if event["type"] in TERMINAL_EVENTS:
                    return


class RunManager:
    def __init__(self, graph: Any, cleanup: Cleanup | None = None) -> None:
        self._graph = graph
        self._cleanup = cleanup
        self._runs: dict[str, Run] = {}
        self._tasks: set[asyncio.Task[None]] = set()

    def get(self, run_id: str) -> Run | None:
        return self._runs.get(run_id)

    async def start(self, prompt: str) -> Run:
        run = Run(id=uuid.uuid4().hex, prompt=prompt)
        self._runs[run.id] = run
        await run.publish({"type": "run_started"})
        self._spawn(run, {"run_id": run.id, "user_prompt": prompt})
        return run

    async def answer(self, run: Run, answer: str) -> None:
        if run.status != "waiting_for_input":
            raise RunNotWaitingError(run.id)
        run.status = "running"
        run.question = None
        await run.publish({"type": "answer_received"})
        self._spawn(run, Command(resume=answer))

    def _spawn(self, run: Run, payload: Any) -> None:
        task = asyncio.create_task(self._drive(run, payload))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _drive(self, run: Run, payload: Any) -> None:
        with metering(run.meter):
            await self._drive_metered(run, payload)

    async def _drive_metered(self, run: Run, payload: Any) -> None:
        config = {"configurable": {"thread_id": run.id}}
        try:
            async for mode, chunk in self._graph.astream(
                payload, config, stream_mode=["custom", "updates"]
            ):
                if mode == "custom":
                    await run.publish(chunk)
                elif "__interrupt__" in chunk:
                    run.question = chunk["__interrupt__"][0].value["question"]
                    run.status = "waiting_for_input"
                    await run.publish({"type": "question", "question": run.question})
                    return

            state = (await self._graph.aget_state(config)).values
            run.article = state["final_article"]
            run.brief = state["brief"].model_dump()
            delivered = state["drafts"][0].ai_score
            run.ai_score = delivered.model_dump() if delivered else None
            run.warnings = state.get("warnings", [])
            run.status = "completed"
            await run.publish({"type": "done"})
        except Exception as exc:
            logger.exception("run %s failed", run.id)
            run.status = "failed"
            run.error = str(exc)
            await run.publish({"type": "error", "message": str(exc)})
        if self._cleanup:
            await self._cleanup(run.id)
