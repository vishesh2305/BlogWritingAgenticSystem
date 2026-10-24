"""Deterministic AI-likeness score built from the stylistic signals AI-text detectors key on.

It is a proxy, not a detector: it measures what an editor can fix (stock phrases, even
sentence rhythm, formal transition openers, em-dash and bullet-list habits, and missing
contractions). Headings, code, tables, links and citation markers are ignored.
"""

import re
import statistics
from dataclasses import dataclass, field

from cwas.models import AIScore, FlaggedParagraph

_STOCK_PHRASES = [
    r"delv(?:e|es|ing)",
    r"in today's (?:fast-paced|digital|modern|ever-changing) (?:world|age|landscape)",
    r"it(?:'s| is) (?:important|crucial|worth|essential) to (?:note|remember|mention|consider)",
    r"in conclusion",
    r"in summary",
    r"to sum up",
    r"tapestry",
    r"a testament to",
    r"navigat(?:e|ing) the (?:complexities|world|landscape)",
    r"in the realm of",
    r"a myriad of",
    r"plays? a (?:crucial|pivotal|vital|key) role",
    r"game[- ]changer",
    r"unlock(?:s|ing)? the (?:power|potential|secrets)",
    r"embark(?:s|ing)? on",
    r"seamless(?:ly)?",
    r"ever-(?:evolving|changing)",
    r"dive (?:deep(?:er)? )?into",
    r"look no further",
    r"harness(?:es|ing)? the power",
    r"compelling",
    r"pivotal",
    r"whether you're a (?:beginner|seasoned)",
]
_STOCK = re.compile(r"\b(?:" + "|".join(_STOCK_PHRASES) + r")\b", re.IGNORECASE)
_TRANSITION = re.compile(
    r"^(?:furthermore|moreover|additionally|in addition|consequently|ultimately|overall|"
    r"notably|importantly|in conclusion|in summary|to summarize|as a result)\b",
    re.IGNORECASE,
)
_CONTRACTION = re.compile(
    r"\b(?:\w+n't|\w+'(?:re|ve|ll|d|m)|(?:it|that|there|here|what|who|let|he|she)'s)\b",
    re.IGNORECASE,
)
_WORD = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])[\"'”’)\]]*\s+")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_CITATION = re.compile(r"\s?\[\d+\]")
_LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_BLOCK_BREAK = re.compile(r"\n\s*\n")
_FENCE = re.compile(r"^(?:```|~~~)", re.MULTILINE)

SIGNALS = (
    "stock_phrases",
    "uniform_rhythm",
    "transitions",
    "em_dashes",
    "lists",
    "no_contractions",
)


@dataclass
class _Stats:
    sentences: list[str] = field(default_factory=list)
    prose: str = ""
    lines: int = 0
    list_lines: int = 0

    @property
    def lengths(self) -> list[int]:
        return [len(_WORD.findall(s)) for s in self.sentences]

    @property
    def words(self) -> int:
        return sum(self.lengths)


def split_blocks(text: str) -> list[str]:
    """Split Markdown into blank-line separated blocks, keeping fenced code blocks whole."""
    blocks: list[str] = []
    for raw in _BLOCK_BREAK.split(text.strip()):
        block = raw.strip()
        if not block:
            continue
        if blocks and len(_FENCE.findall(blocks[-1])) % 2 == 1:
            blocks[-1] += "\n\n" + block  # still inside an open code fence
        else:
            blocks.append(block)
    return blocks


def is_prose(block: str) -> bool:
    return not block.startswith(("#", "```", "~~~", "|"))


def ai_score(text: str) -> AIScore:
    stats = _analyze([b for b in split_blocks(text) if is_prose(b)])
    signals = _signals(stats)
    return AIScore(score=round(min(100.0, sum(signals.values())), 1), signals=signals)


def flag_paragraphs(
    text: str, rhythm_variation: float = 0.3, rhythm_min_sentences: int = 4
) -> list[FlaggedParagraph]:
    """Prose paragraphs that show machine-typical patterns on their own, with reasons.

    A paragraph's rhythm is too even when it has at least `rhythm_min_sentences` sentences
    whose length variation is below `rhythm_variation`."""
    flagged: list[FlaggedParagraph] = []
    for index, block in enumerate(split_blocks(text)):
        if not is_prose(block):
            continue
        stats = _analyze([block])
        reasons: list[str] = []
        phrases = sorted({m.lower() for m in _STOCK.findall(stats.prose)})
        if phrases:
            reasons.append("stock phrases: " + ", ".join(phrases))
        openers = [m.group(0) for s in stats.sentences if (m := _TRANSITION.match(s))]
        if len(openers) >= 2:
            reasons.append("formal transition openers: " + ", ".join(openers))
        lengths = stats.lengths
        if len(lengths) >= rhythm_min_sentences and _variation(lengths) < rhythm_variation:
            reasons.append(f"sentence lengths too even (variation {_variation(lengths):.2f})")
        dashes = _dash_count(stats.prose)
        if dashes >= 2:
            reasons.append(f"{dashes} em dashes")
        if reasons:
            flagged.append(FlaggedParagraph(index=index, reasons=reasons))
    return flagged


def _analyze(blocks: list[str]) -> _Stats:
    stats = _Stats()
    prose_parts: list[str] = []
    for block in blocks:
        paragraph: list[str] = []
        units: list[str] = []
        for line in block.splitlines():
            if not line.strip():
                continue
            stats.lines += 1
            if _LIST_ITEM.match(line):
                stats.list_lines += 1
                units.append(_clean(_LIST_ITEM.sub("", line)))
            else:
                paragraph.append(_clean(line))
        if paragraph:
            units.append(" ".join(paragraph))
        for unit in units:
            prose_parts.append(unit)
            stats.sentences += [s for s in _SENTENCE_BREAK.split(unit) if _WORD.search(s)]
    stats.prose = "\n".join(prose_parts)
    return stats


def _clean(line: str) -> str:
    line = _LINK.sub(r"\1", line)
    line = _CITATION.sub("", line)
    return re.sub(r"[*`]+", "", line).replace("’", "'").strip()


def _signals(stats: _Stats) -> dict[str, float]:
    words = stats.words
    if words == 0:
        return dict.fromkeys(SIGNALS, 0.0)
    per_thousand = 1000 / max(words, 100)  # short texts would otherwise explode the rates

    lengths = stats.lengths
    uniform = _ramp(0.55 - _variation(lengths), 0.30) if len(lengths) >= 5 else 0.0
    openers = sum(bool(_TRANSITION.match(s)) for s in stats.sentences) / len(stats.sentences)
    dash_rate = _dash_count(stats.prose) * per_thousand
    list_share = stats.list_lines / stats.lines if stats.lines else 0.0
    contraction_rate = len(_CONTRACTION.findall(stats.prose)) * per_thousand

    points = {
        "stock_phrases": min(40.0, len(_STOCK.findall(stats.prose)) * per_thousand * 4),
        "uniform_rhythm": uniform * 30,
        "transitions": _ramp(openers - 0.05, 0.20) * 10,
        "em_dashes": _ramp(dash_rate - 2, 8) * 10,
        "lists": _ramp(list_share - 0.3, 0.4) * 10,
        # Short passages carry too few words to judge the absence of contractions.
        "no_contractions": _ramp(5 - contraction_rate, 5) * 8 if words >= 150 else 0.0,
    }
    return {name: round(value, 1) for name, value in points.items()}


def _variation(lengths: list[int]) -> float:
    mean = statistics.fmean(lengths)
    return statistics.pstdev(lengths) / mean if mean else 0.0


def _dash_count(text: str) -> int:
    return text.count("—") + text.count(" -- ")


def _ramp(value: float, span: float) -> float:
    return min(1.0, max(0.0, value / span))
