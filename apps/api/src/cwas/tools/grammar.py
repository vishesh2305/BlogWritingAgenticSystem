"""Deterministic grammar checking (LanguageTool) and readability scoring (textstat)."""

import re
from typing import Protocol

import httpx
import textstat
from pydantic import BaseModel

from cwas.models import Readability

AUTO_FIX_ISSUES = {"grammar", "misspelling", "duplication", "typographical"}

# Markdown spans the checker must never rewrite: citations, links/URLs, inline code.
_PROTECTED = re.compile(r"\[\d+\]|\[[^\]]*\]\([^)]*\)|`[^`]*`|https?://\S+")


class GrammarError(Exception):
    pass


class GrammarMatch(BaseModel):
    offset: int
    length: int
    message: str
    replacements: list[str]
    issue_type: str


class GrammarChecker(Protocol):
    async def check(self, text: str) -> list[GrammarMatch]: ...


class LanguageToolChecker:
    def __init__(self, client: httpx.AsyncClient, base_url: str, language: str = "en-US") -> None:
        self._client = client
        self._base_url = base_url.rstrip("/")
        self._language = language

    async def check(self, text: str) -> list[GrammarMatch]:
        try:
            response = await self._client.post(
                f"{self._base_url}/v2/check",
                data={"text": text, "language": self._language},
                timeout=60.0,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise GrammarError(f"LanguageTool check failed: {exc}") from exc

        return [
            GrammarMatch(
                offset=m["offset"],
                length=m["length"],
                message=m.get("message", ""),
                replacements=[r["value"] for r in m.get("replacements", [])],
                issue_type=m.get("rule", {}).get("issueType", "uncategorized"),
            )
            for m in response.json().get("matches", [])
        ]


def _should_apply(text: str, match: GrammarMatch, protected: list[tuple[int, int]]) -> bool:
    if not match.replacements or match.issue_type not in AUTO_FIX_ISSUES:
        return False
    start, end = match.offset, match.offset + match.length
    if any(start < p_end and p_start < end for p_start, p_end in protected):
        return False
    # Capitalised "misspellings" are usually proper nouns or brands from the sources.
    return not (match.issue_type == "misspelling" and text[start:end][:1].isupper())


def apply_corrections(text: str, matches: list[GrammarMatch]) -> str:
    protected = [m.span() for m in _PROTECTED.finditer(text)]
    accepted: list[GrammarMatch] = []
    last_end = -1
    for match in sorted(matches, key=lambda m: m.offset):
        if match.offset >= last_end and _should_apply(text, match, protected):
            accepted.append(match)
            last_end = match.offset + match.length

    # Apply right-to-left so earlier offsets stay valid.
    for match in reversed(accepted):
        end = match.offset + match.length
        text = text[: match.offset] + match.replacements[0] + text[end:]
    return text


def readability(text: str) -> Readability:
    words = int(textstat.lexicon_count(text))
    if words == 0:
        return Readability(flesch_reading_ease=0.0, grade_level=0.0, word_count=0)
    return Readability(
        flesch_reading_ease=float(textstat.flesch_reading_ease(text)),
        grade_level=float(textstat.flesch_kincaid_grade(text)),
        word_count=words,
    )
