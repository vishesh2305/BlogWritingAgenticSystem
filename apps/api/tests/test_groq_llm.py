import json
from typing import Any

import httpx
import pytest
import respx
from pydantic import BaseModel

from cwas.groq_llm import GroqLLM
from cwas.llm import LLMError, LLMRateLimitError

CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
MODELS = {"opus": "openai/gpt-oss-120b", "sonnet": "openai/gpt-oss-120b", "haiku": "llama-small"}
KEY = "gsk-test-key"


class Answer(BaseModel):
    value: str
    score: int


def reply(content: str, finish: str = "stop", **message: Any) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "message": {"role": "assistant", "content": content, **message},
                    "finish_reason": finish,
                }
            ]
        },
    )


def schema_mismatch(generation: str) -> httpx.Response:
    return httpx.Response(
        400,
        json={
            "error": {
                "message": "Generated JSON does not match the expected schema.",
                "code": "json_validate_failed",
                "failed_generation": generation,
            }
        },
    )


def sent(route: respx.Route, call: int = -1) -> dict[str, Any]:
    return json.loads(route.calls[call].request.content)  # type: ignore[no-any-return]


def make_llm(client: httpx.AsyncClient, **kwargs: Any) -> GroqLLM:
    return GroqLLM(client, api_key=KEY, models=MODELS, **kwargs)  # type: ignore[arg-type]


@respx.mock
async def test_structured_sends_json_schema_and_returns_validated_model() -> None:
    route = respx.post(CHAT_URL).mock(return_value=reply('{"value": "Paris", "score": 9}'))

    async with httpx.AsyncClient() as client:
        result = await make_llm(client).structured(
            tier="sonnet", system="Be precise.", prompt="Capital?", schema=Answer
        )

    assert result == Answer(value="Paris", score=9)
    request = route.calls[-1].request
    assert request.headers["authorization"] == f"Bearer {KEY}"
    body = sent(route)
    assert body["model"] == "openai/gpt-oss-120b"
    assert body["messages"] == [
        {"role": "system", "content": "Be precise."},
        {"role": "user", "content": "Capital?"},
    ]
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "Answer", "schema": Answer.model_json_schema()},
    }


@respx.mock
async def test_gpt_oss_models_get_reasoning_effort_and_others_do_not() -> None:
    route = respx.post(CHAT_URL).mock(return_value=reply("ok"))

    async with httpx.AsyncClient() as client:
        llm = make_llm(client)
        await llm.text(tier="opus", system="s", prompt="p", effort="max")
        await llm.text(tier="haiku", system="s", prompt="p", effort="low")

    assert sent(route, 0)["reasoning_effort"] == "high"
    assert sent(route, 1)["model"] == "llama-small"
    assert "reasoning_effort" not in sent(route, 1)


@respx.mock
async def test_text_returns_content_and_ignores_reasoning() -> None:
    route = respx.post(CHAT_URL).mock(return_value=reply("# Title", reasoning="hidden"))

    async with httpx.AsyncClient() as client:
        text = await make_llm(client).text(tier="opus", system="s", prompt="p", max_tokens=4000)

    assert text == "# Title"
    assert sent(route)["max_completion_tokens"] == 4000
    assert "response_format" not in sent(route)


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
    retry = sent(route, 1)["messages"]
    assert retry[2] == {"role": "assistant", "content": '{"value": "Paris"}'}
    assert "score" in retry[3]["content"]


@respx.mock
async def test_server_side_schema_rejection_is_retried_like_a_validation_error() -> None:
    route = respx.post(CHAT_URL).mock(
        side_effect=[schema_mismatch('{"value": 1}'), reply('{"value": "Paris", "score": 9}')]
    )

    async with httpx.AsyncClient() as client:
        result = await make_llm(client).structured(
            tier="sonnet", system="s", prompt="Capital?", schema=Answer
        )

    assert result.value == "Paris"
    retry = sent(route, 1)["messages"]
    assert retry[2] == {"role": "assistant", "content": '{"value": 1}'}


@respx.mock
async def test_structured_gives_up_after_max_attempts() -> None:
    route = respx.post(CHAT_URL).mock(return_value=schema_mismatch("{}"))

    async with httpx.AsyncClient() as client:
        with pytest.raises(LLMError, match="Answer"):
            await make_llm(client, max_attempts=2).structured(
                tier="sonnet", system="s", prompt="p", schema=Answer
            )

    assert route.call_count == 2


@respx.mock
async def test_truncated_output_raises() -> None:
    respx.post(CHAT_URL).mock(return_value=reply("cut", finish="length"))

    async with httpx.AsyncClient() as client:
        with pytest.raises(LLMError, match="truncated"):
            await make_llm(client).text(tier="opus", system="s", prompt="p")


@respx.mock
async def test_rate_limit_raises_with_retry_after() -> None:
    respx.post(CHAT_URL).mock(
        return_value=httpx.Response(429, headers={"retry-after": "2.5"}, json={"error": {}})
    )

    async with httpx.AsyncClient() as client:
        with pytest.raises(LLMRateLimitError) as caught:
            await make_llm(client).text(tier="opus", system="s", prompt="p")

    assert caught.value.retry_after == 2.5


@respx.mock
async def test_request_too_large_raises_llm_error_without_leaking_the_key() -> None:
    respx.post(CHAT_URL).mock(
        return_value=httpx.Response(413, json={"error": {"message": "Request too large"}})
    )

    async with httpx.AsyncClient() as client:
        with pytest.raises(LLMError, match="413") as caught:
            await make_llm(client).text(tier="opus", system="s", prompt="p")

    assert KEY not in str(caught.value)


@respx.mock
async def test_network_failure_raises_llm_error() -> None:
    respx.post(CHAT_URL).mock(side_effect=httpx.ReadTimeout("slow"))

    async with httpx.AsyncClient() as client:
        with pytest.raises(LLMError, match="Groq"):
            await make_llm(client).text(tier="opus", system="s", prompt="p")
