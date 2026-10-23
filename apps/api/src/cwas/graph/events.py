"""Agent lifecycle events, streamed through LangGraph's custom stream channel."""

from time import perf_counter
from typing import Any, Protocol

from langgraph.config import get_stream_writer

from cwas.graph.state import RunState
from cwas.usage import agent_scope, current_meter


class Node(Protocol):
    """A graph node; LangGraph's typing requires the parameter to be named `state`."""

    async def __call__(self, state: RunState) -> dict[str, Any]: ...


def instrument(agent: str, fn: Node) -> Node:
    async def node(state: RunState) -> dict[str, Any]:
        emit = get_stream_writer()
        emit({"type": "agent_started", "agent": agent})
        started = perf_counter()
        tokens_before = _agent_tokens(agent)
        try:
            with agent_scope(agent):
                update = await fn(state)
        except Exception as exc:
            emit({"type": "agent_failed", "agent": agent, "error": str(exc)})
            raise
        emit(
            {
                "type": "agent_finished",
                "agent": agent,
                "duration_ms": round((perf_counter() - started) * 1000),
                "tokens": _agent_tokens(agent) - tokens_before,
            }
        )
        return update

    return node


def _agent_tokens(agent: str) -> int:
    meter = current_meter()
    usage = meter.by_agent.get(agent) if meter else None
    return usage.input_tokens + usage.output_tokens if usage else 0
