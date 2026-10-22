import json
from typing import Any

import httpx
import pytest
import respx
from pydantic import BaseModel

from cwas.llm import LLMError
from cwas.ollama_llm import OllamaLLM

CHAT_URL = "http://localhost:11434/api/chat"


class Answer(BaseModel):
    value: str
    score: int


def reply(content: str, done_reason: str = "stop", **message: Any) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "message": {"role": "assistant", "content": content, **message},
            "done": True,
            "done_reason": done_reason,
        },
    )


def sent(route: respx.Route, call: int = -1) -> dict[str, Any]:
    return json.loads(route.calls[call].request.content)  # type: ignore[no-any-return]


def make_llm(client: httpx.AsyncClient, **kwargs: Any) -> OllamaLLM:
    return OllamaLLM(client, model="qwen3-vl:8b-instruct", num_ctx=16384, **kwargs)


@respx.mock
async def test_structured_sends_schema_and_returns_validated_model() -> None:
    route = respx.post(CHAT_URL).mock(return_value=reply('{"value": "Paris", "score": 9}'))

    async with httpx.AsyncClient() as client:
        result = await make_llm(client).structured(
            tier="sonnet", system="Be precise.", prompt="Capital?", schema=Answer
        )

    assert result == Answer(value="Paris", score=9)
    body = sent(route)
    assert body["model"] == "qwen3-vl:8b-instruct"
    assert body["stream"] is False
    assert body["messages"] == [
        {"role": "system", "content": "Be precise."},
        {"role": "user", "content": "Capital?"},
    ]
    assert body["format"] == Answer.model_json_schema()
    assert body["options"]["num_ctx"] == 16384


@respx.mock
async def test_every_tier_uses_the_configured_local_model() -> None:
    route = respx.post(CHAT_URL).mock(return_value=reply("text"))

    async with httpx.AsyncClient() as client:
        llm = make_llm(client)
        for tier in ("opus", "sonnet", "haiku"):
            await llm.text(tier=tier, system="s", prompt="p")  # type: ignore[arg-type]

    assert {sent(route, i)["model"] for i in range(3)} == {"qwen3-vl:8b-instruct"}


@respx.mock
async def test_structured_retries_with_the_validation_error_then_succeeds() -> None:
    route = respx.post(CHAT_URL).mock(
        side_effect=[reply('{"value": "Paris"}'), reply('{"value": "Paris", "score": 9}')]
    )

    async with httpx.AsyncClient() as client:
        result = await make_llm(client).structured(
            tier="sonnet", system="s", prompt="Capital?", schema=Answer
        )

    assert result.score == 9
    retry_messages = sent(route, 1)["messages"]
    assert retry_messages[2] == {"role": "assistant", "content": '{"value": "Paris"}'}
    assert retry_messages[3]["role"] == "user"
    assert "score" in retry_messages[3]["content"]


@respx.mock
async def test_structured_gives_up_after_max_attempts() -> None:
    route = respx.post(CHAT_URL).mock(return_value=reply("not json at all"))

    async with httpx.AsyncClient() as client:
        with pytest.raises(LLMError, match="Answer"):
            await make_llm(client, max_attempts=3).structured(
                tier="sonnet", system="s", prompt="p", schema=Answer
            )

    assert route.call_count == 3


@respx.mock
async def test_text_returns_content_without_inline_thinking() -> None:
    respx.post(CHAT_URL).mock(
        return_value=reply("<think>\nplanning the essay\n</think>\n\n# Title\n\nBody.")
    )

    async with httpx.AsyncClient() as client:
        text = await make_llm(client).text(tier="opus", system="Write.", prompt="Go")

    assert text == "# Title\n\nBody."


@respx.mock
async def test_text_ignores_separate_thinking_field() -> None:
    respx.post(CHAT_URL).mock(return_value=reply("Answer.", thinking="hidden reasoning"))

    async with httpx.AsyncClient() as client:
        assert await make_llm(client).text(tier="opus", system="s", prompt="p") == "Answer."


@respx.mock
async def test_text_passes_max_tokens_as_num_predict() -> None:
    route = respx.post(CHAT_URL).mock(return_value=reply("ok"))

    async with httpx.AsyncClient() as client:
        await make_llm(client).text(tier="opus", system="s", prompt="p", max_tokens=4000)

    assert sent(route)["options"]["num_predict"] == 4000


@respx.mock
async def test_truncated_output_raises() -> None:
    respx.post(CHAT_URL).mock(return_value=reply("cut off mid", done_reason="length"))

    async with httpx.AsyncClient() as client:
        with pytest.raises(LLMError, match="truncated"):
            await make_llm(client).text(tier="opus", system="s", prompt="p")


@respx.mock
async def test_connection_failure_explains_that_ollama_must_be_running() -> None:
    respx.post(CHAT_URL).mock(side_effect=httpx.ConnectError("refused"))

    async with httpx.AsyncClient() as client:
        with pytest.raises(LLMError, match="Ollama"):
            await make_llm(client).text(tier="opus", system="s", prompt="p")


@respx.mock
async def test_http_error_raises_llm_error() -> None:
    respx.post(CHAT_URL).mock(return_value=httpx.Response(404, json={"error": "model not found"}))

    async with httpx.AsyncClient() as client:
        with pytest.raises(LLMError, match="model not found"):
            await make_llm(client).text(tier="opus", system="s", prompt="p")
