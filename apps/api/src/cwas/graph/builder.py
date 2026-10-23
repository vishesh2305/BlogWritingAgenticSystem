"""Orchestrator: wires the agents into a LangGraph state machine."""

from collections.abc import Awaitable, Callable
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from cwas.agents.deps import AgentDeps
from cwas.agents.grammar import grammar_agent
from cwas.agents.humanizer import humanizer
from cwas.agents.kb_builder import kb_builder
from cwas.agents.prompt_enhancer import clarify, needs_clarification, prompt_enhancer
from cwas.agents.researcher import researcher
from cwas.agents.writer import writer
from cwas.graph.events import Node, instrument
from cwas.graph.finalize import finalize
from cwas.graph.state import RunState

AgentFn = Callable[[RunState, AgentDeps], Awaitable[dict[str, Any]]]

# The linear part of the slice, in execution order after the prompt enhancer.
PIPELINE: list[tuple[str, AgentFn]] = [
    ("researcher", researcher),
    ("kb_builder", kb_builder),
    ("writer", writer),
    ("humanizer", humanizer),
    ("grammar", grammar_agent),
]


def build_graph(
    deps: AgentDeps, checkpointer: BaseCheckpointSaver[Any] | None = None
) -> CompiledStateGraph[Any, Any, Any, Any]:
    def bind(name: str, fn: AgentFn) -> Node:
        async def call(state: RunState) -> dict[str, Any]:
            return await fn(state, deps)

        return instrument(name, call)

    graph = StateGraph(RunState)
    graph.add_node("prompt_enhancer", bind("prompt_enhancer", prompt_enhancer))
    graph.add_node("clarify", clarify)
    for name, fn in PIPELINE:
        graph.add_node(name, bind(name, fn))
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "prompt_enhancer")
    graph.add_conditional_edges(
        "prompt_enhancer",
        lambda state: "clarify" if needs_clarification(state, deps.settings) else "researcher",
        ["clarify", "researcher"],
    )
    graph.add_edge("clarify", "prompt_enhancer")
    names = [name for name, _ in PIPELINE]
    for current, following in zip(names, names[1:], strict=False):
        graph.add_edge(current, following)
    graph.add_edge(names[-1], "finalize")
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer)
