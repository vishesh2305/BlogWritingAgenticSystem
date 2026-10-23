import json
from typing import Any

import httpx
import pytest
import respx
from pydantic import BaseModel

from cwas.gemini_llm import GeminiLLM
from cwas.llm import LLMError, LLMRateLimitError, LLMRefusalError

BASE = "https://generativelanguage.googleapis.com/v1beta/models"
FLASH_URL = f"{BASE}/gemini-flash:generateContent"
LITE_URL = f"{BASE}/gemini-lite:generateContent"
MODELS = {"opus": "gemini-flash", "sonnet": "gemini-flash", "haiku": "gemini-lite"}
KEY = "AIza-test-key"


class Answer(BaseModel):
    value: str
    score: int


def reply(*texts: str, finish: str = "STOP", thought: str | None = None) -> httpx.Response:
    parts: list[dict[str, Any]] = [{"text": thought, "thought": True}] if thought else []
    parts += [{"text": t} for t in texts]
    return httpx.Response(
        200,
        json={
            "candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": finish}]
        },
    )


def sent(route: respx.Route, call: int = -1) -> dict[str, Any]:
    return json.loads(route.calls[call].request.content)  # type: ignore[no-any-return]


def make_llm(client: httpx.AsyncClient, **kwargs: Any) -> GeminiLLM:
    return GeminiLLM(client, api_key=KEY, models=MODELS, **kwargs)  # type: ignore[arg-type]


@respx.mock
async def test_structured_sends_json_schema_and_returns_validated_model() -> None:
    route = respx.post(FLASH_URL).mock(return_value=reply('{"value": "Paris", "score": 9}'))

    async with httpx.AsyncClient() as client:
        result = await make_llm(client).structured(
            tier="sonnet", system="Be precise.", prompt="Capital?", schema=Answer
        )

    assert result == Answer(value="Paris", score=9)
    body = sent(route)
    assert body["systemInstruction"] == {"parts": [{"text": "Be precise."}]}
    assert body["contents"] == [{"role": "user", "parts": [{"text": "Capital?"}]}]
    config = body["generationConfig"]
    assert config["responseMimeType"] == "application/json"
    assert config["responseJsonSchema"] == Answer.model_json_schema()


@respx.mock
async def test_api_key_is_sent_in_a_header_never_in_the_url() -> None:
    route = respx.post(FLASH_URL).mock(return_value=reply("ok"))

    async with httpx.AsyncClient() as client:
        await make_llm(client).text(tier="opus", system="s", prompt="p")

    request = route.calls[-1].request
    assert request.headers["x-goog-api-key"] == KEY
    assert KEY not in str(request.url)


@respx.mock
async def test_each_tier_uses_its_configured_model() -> None:
    flash = respx.post(FLASH_URL).mock(return_value=reply("big"))
    lite = respx.post(LITE_URL).mock(return_value=reply("small"))

    async with httpx.AsyncClient() as client:
        llm = make_llm(client)
        assert await llm.text(tier="opus", system="s", prompt="p") == "big"
        assert await llm.text(tier="haiku", system="s", prompt="p") == "small"

    assert flash.call_count == 1
    assert lite.call_count == 1


@respx.mock
async def test_effort_maps_to_a_speed_biased_thinking_level() -> None:
    route = respx.post(FLASH_URL).mock(return_value=reply("ok"))

    async with httpx.AsyncClient() as client:
        llm = make_llm(client)
        for effort in ("low", "medium", "high", "max"):
            await llm.text(tier="opus", system="s", prompt="p", effort=effort)

    levels = [
        sent(route, i)["generationConfig"]["thinkingConfig"]["thinkingLevel"] for i in range(4)
    ]
    assert levels == ["low", "low", "medium", "high"]


@respx.mock
async def test_text_joins_answer_parts_and_drops_thoughts() -> None:
    respx.post(FLASH_URL).mock(return_value=reply("# Title\n\n", "Body.", thought="planning"))

    async with httpx.AsyncClient() as client:
        text = await make_llm(client).text(tier="opus", system="s", prompt="p")

    assert text == "# Title\n\nBody."


@respx.mock
async def test_text_passes_max_tokens() -> None:
    route = respx.post(FLASH_URL).mock(return_value=reply("ok"))

    async with httpx.AsyncClient() as client:
        await make_llm(client).text(tier="opus", system="s", prompt="p", max_tokens=4000)

    assert sent(route)["generationConfig"]["maxOutputTokens"] == 4000


@respx.mock
async def test_structured_retries_with_the_validation_error_then_succeeds() -> None:
    route = respx.post(FLASH_URL).mock(
        side_effect=[reply('{"value": "Paris"}'), reply('{"value": "Paris", "score": 9}')]
    )

    async with httpx.AsyncClient() as client:
        result = await make_llm(client).structured(
            tier="sonnet", system="s", prompt="Capital?", schema=Answer
        )

    assert result.score == 9
    retry = sent(route, 1)["contents"]
    assert retry[1] == {"role": "model", "parts": [{"text": '{"value": "Paris"}'}]}
    assert retry[2]["role"] == "user"
    assert "score" in retry[2]["parts"][0]["text"]


@respx.mock
async def test_structured_gives_up_after_max_attempts() -> None:
    route = respx.post(FLASH_URL).mock(return_value=reply("not json"))

    async with httpx.AsyncClient() as client:
        with pytest.raises(LLMError, match="Answer"):
            await make_llm(client, max_attempts=3).structured(
                tier="sonnet", system="s", prompt="p", schema=Answer
            )

    assert route.call_count == 3


@respx.mock
async def test_truncated_output_raises() -> None:
    respx.post(FLASH_URL).mock(return_value=reply("cut off", finish="MAX_TOKENS"))

    async with httpx.AsyncClient() as client:
        with pytest.raises(LLMError, match="truncated"):
            await make_llm(client).text(tier="opus", system="s", prompt="p")


@pytest.mark.parametrize("finish", ["SAFETY", "RECITATION", "PROHIBITED_CONTENT"])
@respx.mock
async def test_blocked_candidate_raises_refusal(finish: str) -> None:
    respx.post(FLASH_URL).mock(return_value=reply("", finish=finish))

    async with httpx.AsyncClient() as client:
        with pytest.raises(LLMRefusalError, match=finish):
            await make_llm(client).text(tier="opus", system="s", prompt="p")


@respx.mock
async def test_blocked_prompt_raises_refusal() -> None:
    respx.post(FLASH_URL).mock(
        return_value=httpx.Response(200, json={"promptFeedback": {"blockReason": "SAFETY"}})
    )

    async with httpx.AsyncClient() as client:
        with pytest.raises(LLMRefusalError, match="SAFETY"):
            await make_llm(client).text(tier="opus", system="s", prompt="p")


@respx.mock
async def test_rate_limit_raises_with_retry_after() -> None:
    respx.post(FLASH_URL).mock(
        return_value=httpx.Response(429, headers={"retry-after": "17"}, json={"error": "quota"})
    )

    async with httpx.AsyncClient() as client:
        with pytest.raises(LLMRateLimitError) as caught:
            await make_llm(client).text(tier="opus", system="s", prompt="p")

    assert caught.value.retry_after == 17.0


@respx.mock
async def test_server_error_raises_llm_error_without_leaking_the_key() -> None:
    respx.post(FLASH_URL).mock(return_value=httpx.Response(503, json={"error": "overloaded"}))

    async with httpx.AsyncClient() as client:
        with pytest.raises(LLMError, match="503") as caught:
            await make_llm(client).text(tier="opus", system="s", prompt="p")

    assert KEY not in str(caught.value)


@respx.mock
async def test_network_failure_raises_llm_error() -> None:
    respx.post(FLASH_URL).mock(side_effect=httpx.ConnectTimeout("slow"))

    async with httpx.AsyncClient() as client:
        with pytest.raises(LLMError, match="Gemini"):
            await make_llm(client).text(tier="opus", system="s", prompt="p")


@respx.mock
async def test_malformed_response_raises_llm_error() -> None:
    respx.post(FLASH_URL).mock(return_value=httpx.Response(200, json={"candidates": []}))

    async with httpx.AsyncClient() as client:
        with pytest.raises(LLMError, match="no candidates"):
            await make_llm(client).text(tier="opus", system="s", prompt="p")
