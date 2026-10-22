"""Agent lifecycle events, streamed through LangGraph's custom stream channel."""

from time import perf_counter
from typing import Any, Protocol

from langgraph.config import get_stream_writer

from cwas.graph.state import RunState


class Node(Protocol):
    """A graph node; LangGraph's typing requires the parameter to be named `state`."""

    async def __call__(self, state: RunState) -> dict[str, Any]: ...


def instrument(agent: str, fn: Node) -> Node:
    async def node(state: RunState) -> dict[str, Any]:
        emit = get_stream_writer()
        emit({"type": "agent_started", "agent": agent})
        started = perf_counter()
        try:
            update = await fn(state)
        except Exception as exc:
            emit({"type": "agent_failed", "agent": agent, "error": str(exc)})
            raise
        emit(
            {
                "type": "agent_finished",
                "agent": agent,
                "duration_ms": round((perf_counter() - started) * 1000),
            }
        )
        return update

    return node
