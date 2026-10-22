from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from cwas.llm import FALLBACK_BETA, AnthropicLLM, LLMError, LLMRefusalError

MODELS = {"opus": "claude-opus-5-5", "sonnet": "claude-sonnet-5-5", "haiku": "claude-haiku-5-5"}


class Answer(BaseModel):
    value: str


def response(**overrides: Any) -> SimpleNamespace:
    fields: dict[str, Any] = {
        "stop_reason": "end_turn",
        "stop_details": None,
        "parsed_output": Answer(value="42"),
        "content": [
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text="Hello "),
            SimpleNamespace(type="text", text="world"),
        ],
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


class FakeMessages:
    def __init__(self, reply: SimpleNamespace) -> None:
        self.reply = reply
        self.calls: list[dict[str, Any]] = []

    async def parse(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return self.reply

    async def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return self.reply


def make_llm(reply: SimpleNamespace) -> tuple[AnthropicLLM, FakeMessages]:
    messages = FakeMessages(reply)
    client = SimpleNamespace(beta=SimpleNamespace(messages=messages))
    return AnthropicLLM(client, MODELS), messages


async def test_structured_returns_parsed_output_and_sends_schema() -> None:
    llm, messages = make_llm(response())

    result = await llm.structured(
        tier="sonnet", system="Be precise.", prompt="Q?", schema=Answer, effort="low"
    )

    assert result == Answer(value="42")
    call = messages.calls[0]
    assert call["model"] == "claude-sonnet-5-5"
    assert call["system"] == "Be precise."
    assert call["messages"] == [{"role": "user", "content": "Q?"}]
    assert call["output_format"] is Answer
    assert call["output_config"] == {"effort": "low"}


@pytest.mark.parametrize("tier", ["opus", "sonnet"])
async def test_opus_and_sonnet_requests_enable_server_side_fallbacks(tier: str) -> None:
    llm, messages = make_llm(response())
    await llm.structured(tier=tier, system="s", prompt="p", schema=Answer)  # type: ignore[arg-type]
    assert messages.calls[0]["fallbacks"] == "default"
    assert messages.calls[0]["betas"] == [FALLBACK_BETA]


async def test_haiku_requests_have_no_fallbacks() -> None:
    llm, messages = make_llm(response())
    await llm.structured(tier="haiku", system="s", prompt="p", schema=Answer)
    assert "fallbacks" not in messages.calls[0]
    assert "betas" not in messages.calls[0]


async def test_refusal_raises_refusal_error_with_category() -> None:
    reply = response(
        stop_reason="refusal",
        stop_details=SimpleNamespace(category="cyber", explanation="no"),
        parsed_output=None,
    )
    llm, _ = make_llm(reply)
    with pytest.raises(LLMRefusalError, match="cyber"):
        await llm.structured(tier="opus", system="s", prompt="p", schema=Answer)


async def test_truncated_output_raises_llm_error() -> None:
    llm, _ = make_llm(response(stop_reason="max_tokens"))
    with pytest.raises(LLMError, match="max_tokens"):
        await llm.text(tier="opus", system="s", prompt="p")


async def test_missing_parsed_output_raises_llm_error() -> None:
    llm, _ = make_llm(response(parsed_output=None))
    with pytest.raises(LLMError):
        await llm.structured(tier="opus", system="s", prompt="p", schema=Answer)


async def test_text_joins_text_blocks_and_skips_thinking() -> None:
    llm, messages = make_llm(response())

    text = await llm.text(tier="opus", system="Write.", prompt="Go", effort="high", max_tokens=9000)

    assert text == "Hello world"
    call = messages.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["max_tokens"] == 9000
    assert call["output_config"] == {"effort": "high"}
