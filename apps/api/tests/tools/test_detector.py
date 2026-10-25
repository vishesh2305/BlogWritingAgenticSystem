import json

import httpx
import pytest
import respx

from cwas.tools.detector import DetectorError, SaplingDetector, SentenceScore

URL = "https://api.sapling.ai/api/v1/aidetect"
KEY = "sapling-test-key"


def reply(score: float = 0.82) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "score": score,
            "ai_fraction": 0.5,
            "verdict": "mixed",
            "sentence_scores": [
                {"score": 0.91, "sentence": "Panels work.", "start": 0, "end": 12},
                {"score": 0.08, "sentence": "Mine did.", "start": 13, "end": 22},
            ],
        },
    )


@respx.mock
async def test_scores_text_as_a_percentage_with_sentence_spans() -> None:
    route = respx.post(URL).mock(return_value=reply())

    async with httpx.AsyncClient() as client:
        result = await SaplingDetector(client, api_key=KEY).detect("Panels work. Mine did.")

    assert result.score == 82.0
    assert result.ai_fraction == 0.5
    assert result.sentences == [
        SentenceScore(start=0, end=12, score=0.91),
        SentenceScore(start=13, end=22, score=0.08),
    ]
    request = route.calls.last.request
    assert json.loads(request.content) == {"text": "Panels work. Mine did.", "sent_scores": True}
    assert request.headers["authorization"] == f"Bearer {KEY}"


def test_has_a_provider_name() -> None:
    assert SaplingDetector(httpx.AsyncClient(), api_key=KEY).name == "sapling"


@respx.mock
async def test_rate_limit_raises_detector_error_without_leaking_the_key() -> None:
    respx.post(URL).mock(return_value=httpx.Response(429, json={"msg": "Too many characters"}))

    async with httpx.AsyncClient() as client:
        with pytest.raises(DetectorError, match="429") as caught:
            await SaplingDetector(client, api_key=KEY).detect("text")

    assert KEY not in str(caught.value)


@respx.mock
async def test_network_failure_raises_detector_error() -> None:
    respx.post(URL).mock(side_effect=httpx.ReadTimeout("slow"))

    async with httpx.AsyncClient() as client:
        with pytest.raises(DetectorError, match="Sapling"):
            await SaplingDetector(client, api_key=KEY).detect("text")


@respx.mock
async def test_malformed_response_raises_detector_error() -> None:
    respx.post(URL).mock(return_value=httpx.Response(200, json={"verdict": "ai"}))

    async with httpx.AsyncClient() as client:
        with pytest.raises(DetectorError, match="score"):
            await SaplingDetector(client, api_key=KEY).detect("text")
