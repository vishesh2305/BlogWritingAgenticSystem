"""Deterministic fact extraction: source sentences that carry a number.

Taking sentences verbatim (no LLM) keeps it free and makes every statistic traceable, so later
agents can prove that a number they add exists in a source.
"""

import re

from cwas.models import Fact, ScrapedPage

_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(])")
_LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_WORD = re.compile(r"\w+")
MIN_WORDS = 6
MAX_WORDS = 60


def numbers_in(text: str) -> list[str]:
    """Numbers in reading order, normalised so that 78,000 and 78000 compare equal."""
    return [n.replace(",", "") for n in _NUMBER.findall(text)]


def extract_facts(pages: list[ScrapedPage], max_per_page: int = 15) -> list[Fact]:
    facts: list[Fact] = []
    seen: set[str] = set()
    for page in pages:
        kept = 0
        for sentence in _sentences(page.text):
            key = " ".join(_WORD.findall(sentence.lower()))
            if kept == max_per_page or key in seen:
                continue
            seen.add(key)
            kept += 1
            facts.append(
                Fact(
                    id=f"f{len(facts) + 1}",
                    text=sentence,
                    numbers=numbers_in(sentence),
                    source_url=page.url,
                    source_title=page.title,
                    source_type=page.source_type,
                )
            )
    return facts


def _sentences(markdown: str) -> list[str]:
    found: list[str] = []
    for line in markdown.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "|", "```")):
            continue
        line = _LIST_ITEM.sub("", line).replace("**", "").replace("__", "")
        for sentence in _SENTENCE_BREAK.split(line):
            sentence = sentence.strip()
            if _NUMBER.search(sentence) and MIN_WORDS <= len(_WORD.findall(sentence)) <= MAX_WORDS:
                found.append(sentence)
    return found
