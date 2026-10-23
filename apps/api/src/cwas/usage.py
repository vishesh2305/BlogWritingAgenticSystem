"""Per-run token accounting. LLM backends call `record_usage`; the run and the agent being
executed are carried in context variables, so no agent has to pass a meter around."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

from cwas.models import Usage

_meter: ContextVar["UsageMeter | None"] = ContextVar("usage_meter", default=None)
_agent: ContextVar[str | None] = ContextVar("usage_agent", default=None)


@dataclass
class UsageMeter:
    total: Usage = field(default_factory=Usage)
    by_agent: dict[str, Usage] = field(default_factory=dict)

    def add(self, agent: str | None, input_tokens: int, output_tokens: int) -> None:
        targets = [self.total]
        if agent is not None:
            targets.append(self.by_agent.setdefault(agent, Usage()))
        for usage in targets:
            usage.calls += 1
            usage.input_tokens += input_tokens
            usage.output_tokens += output_tokens


def current_meter() -> UsageMeter | None:
    return _meter.get()


def record_usage(input_tokens: int, output_tokens: int) -> None:
    meter = _meter.get()
    if meter is not None:
        meter.add(_agent.get(), input_tokens, output_tokens)


@contextmanager
def metering(meter: UsageMeter) -> Iterator[UsageMeter]:
    token = _meter.set(meter)
    try:
        yield meter
    finally:
        _meter.reset(token)


@contextmanager
def agent_scope(agent: str) -> Iterator[None]:
    token = _agent.set(agent)
    try:
        yield
    finally:
        _agent.reset(token)
