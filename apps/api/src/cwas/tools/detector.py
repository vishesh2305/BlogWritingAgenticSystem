"""External AI-text detectors. They judge the finished prose; the local ai_score stays the cheap,
always-available pre-check."""

from typing import Protocol

import httpx
from pydantic import BaseModel


class DetectorError(Exception):
    pass


class SentenceScore(BaseModel):
    start: int
    end: int
    score: float = 0.0  # 0-1: how machine-written this sentence looks


class DetectorResult(BaseModel):
    score: float  # 0-100: how machine-written the whole text looks
    ai_fraction: float | None = None
    sentences: list[SentenceScore] = []


class AIDetector(Protocol):
    name: str

    async def detect(self, text: str) -> DetectorResult: ...


class SaplingDetector:
    """Sapling's AI detector: one request scores the text and each sentence. Identical text is
    cached by Sapling for 3 days, so rechecking unchanged text costs no quota."""

    name = "sapling"

    def __init__(
        self,
        client: httpx.AsyncClient,
        api_key: str,
        base_url: str = "https://api.sapling.ai/api/v1",
        timeout: float = 20.0,
    ) -> None:
        self._client = client
        self._api_key = api_key
        self._url = f"{base_url.rstrip('/')}/aidetect"
        self._timeout = timeout

    async def detect(self, text: str) -> DetectorResult:
        try:
            response = await self._client.post(
                self._url,
                headers={"authorization": f"Bearer {self._api_key}"},
                json={"text": text, "sent_scores": True},
                timeout=self._timeout,
            )
        except httpx.HTTPError as exc:
            raise DetectorError(f"Sapling request failed: {exc!r}") from exc
        if response.is_error:
            raise DetectorError(
                f"Sapling returned HTTP {response.status_code}: {response.text[:300]}"
            )

        data = response.json()
        if "score" not in data:
            raise DetectorError("Sapling returned no score")
        return DetectorResult(
            score=round(float(data["score"]) * 100, 1),
            ai_fraction=data.get("ai_fraction"),
            sentences=[
                SentenceScore(start=s["start"], end=s["end"], score=s["score"])
                for s in data.get("sentence_scores") or []
            ],
        )
