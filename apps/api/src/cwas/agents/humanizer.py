"""Agent 7 - Humanizer: rewrites machine-sounding paragraphs until the AI-likeness score is low.

With an external detector configured, its score is the target and its sentence scores pick the
paragraphs to rewrite; it is called once up front and once per changed round, never more.
Only flagged prose paragraphs are rewritten, so headings, code and the rest of the article
stay byte-for-byte intact. A rewrite that changes a paragraph's citations, adds a number or
drops the primary keyword is rejected, and a round that does not lower the score is discarded.
"""

import re
from dataclasses import dataclass
from html import escape
from typing import Any

from cwas.agents import prompts
from cwas.agents.deps import AgentDeps
from cwas.graph.state import RunState
from cwas.llm import LLMError
from cwas.models import AIScore, Brief, DetectorScore, FlaggedParagraph, Rewrite
from cwas.tools.ai_score import (
    ai_score,
    flag_paragraphs,
    is_prose,
    sentence_count,
    split_blocks,
)
from cwas.tools.detector import AIDetector, DetectorError
from cwas.tools.guards import rewrite_violations

_DASH_RANGE = re.compile(r"(\d)\s*—\s*(\d)")
_DASH_PAIR = re.compile(r"\s*—\s*([^—\n.!?]+?)\s*—\s*")
_DASH = re.compile(r"\s*—\s*")
_DOUBLED_PUNCTUATION = re.compile(r",\s*([,.;:!?])")


_RHYTHM_REASON = (
    "the article's sentence rhythm is too even: mix sentences under 6 words with sentences "
    "over 25 words"
)


def replace_em_dashes(text: str) -> str:
    """Number ranges get an en dash; dashes around or before a clause become commas."""
    text = _DASH_RANGE.sub(r"\1–\2", text)
    text = _DASH_PAIR.sub(r", \1, ", text)
    text = _DASH.sub(", ", text)
    return _DOUBLED_PUNCTUATION.sub(r"\1", text)


# Negations contract anywhere; a pronoun + verb only when more words follow, so "as it is." and
# "that is, roughly" keep their meaning.
_NEGATIONS = {
    "do not": "don't",
    "does not": "doesn't",
    "did not": "didn't",
    "is not": "isn't",
    "are not": "aren't",
    "was not": "wasn't",
    "were not": "weren't",
    "has not": "hasn't",
    "have not": "haven't",
    "had not": "hadn't",
    "should not": "shouldn't",
    "would not": "wouldn't",
    "could not": "couldn't",
    "will not": "won't",
    "can not": "can't",
    "cannot": "can't",
}
_PRONOUN_VERBS = {
    "it is": "it's",
    "that is": "that's",
    "there is": "there's",
    "here is": "here's",
    "what is": "what's",
    "you are": "you're",
    "we are": "we're",
    "they are": "they're",
    "you will": "you'll",
    "we will": "we'll",
    "they will": "they'll",
    "i am": "I'm",
    "let us": "let's",
}


def _phrase(words: str) -> str:
    return r"\b" + r"\s+".join(map(re.escape, words.split())) + r"\b"


_CONTRACTIONS = [
    (re.compile(_phrase(full), re.IGNORECASE), short) for full, short in _NEGATIONS.items()
] + [
    (re.compile(_phrase(full) + r"(?=\s+[\w₹$])", re.IGNORECASE), short)
    for full, short in _PRONOUN_VERBS.items()
]


def add_contractions(text: str) -> str:
    """Contract "do not", "it is", "you are" and similar the way people write them."""

    for pattern, short in _CONTRACTIONS:

        def contract(match: re.Match[str], short: str = short) -> str:
            return short[0].upper() + short[1:] if match.group(0)[0].isupper() else short

        text = pattern.sub(contract, text)
    return text


def _polish(text: str, contractions: bool) -> str:
    """Fix style habits that need no model: em dashes and, unless formal, contractions."""
    if "—" not in text and not contractions:
        return text

    def fix(block: str) -> str:
        block = replace_em_dashes(block)
        return add_contractions(block) if contractions else block

    return "\n\n".join(fix(b) if is_prose(b) else b for b in split_blocks(text))


async def humanizer(state: RunState, deps: AgentDeps) -> dict[str, Any]:
    warnings: list[str] = []
    drafts = []
    for draft in state["drafts"]:
        text, score, verdict = await _humanize(draft.markdown, state["brief"], deps, warnings)
        detector_score = (
            DetectorScore(
                provider=verdict.provider, score=verdict.score, ai_fraction=verdict.ai_fraction
            )
            if verdict
            else None
        )
        drafts.append(
            draft.model_copy(
                update={"markdown": text, "ai_score": score, "detector_score": detector_score}
            )
        )
    return {"drafts": drafts, "warnings": warnings}


@dataclass
class _Verdict:
    provider: str
    score: float
    ai_fraction: float | None
    flagged: dict[int, int]  # block index -> number of sentences the detector flags


class _DetectorGate:
    """Runs the external detector at most `max_calls` times and stops using it after the first
    failure, so a slow or exhausted detector can never hold up or break a run."""

    def __init__(self, detector: AIDetector | None, max_calls: int, warnings: list[str]) -> None:
        self._detector = detector
        self._calls_left = max_calls
        self._warnings = warnings

    async def check(self, text: str) -> _Verdict | None:
        if self._detector is None or self._calls_left <= 0:
            return None
        self._calls_left -= 1
        blocks = split_blocks(text)
        prose, spans = _prose_with_spans(blocks)
        try:
            result = await self._detector.detect(prose)
        except DetectorError as exc:
            self._detector = None
            self._warnings.append(
                f"AI detector unavailable; the local AI-likeness score was used instead: {exc}"
            )
            return None
        flagged: dict[int, int] = {}
        for sentence in result.sentences:
            if sentence.score >= 0.5:
                for index, start, end in spans:
                    if start <= sentence.start < end:
                        flagged[index] = flagged.get(index, 0) + 1
        return _Verdict(self._detector.name, result.score, result.ai_fraction, flagged)


def _prose_with_spans(blocks: list[str]) -> tuple[str, list[tuple[int, int, int]]]:
    """Join the prose blocks (no headings, tables or code) and remember where each one sits."""
    parts: list[str] = []
    spans: list[tuple[int, int, int]] = []
    offset = 0
    for index, block in enumerate(blocks):
        if not is_prose(block):
            continue
        spans.append((index, offset, offset + len(block)))
        parts.append(block)
        offset += len(block) + 2
    return "\n\n".join(parts), spans


async def _humanize(
    text: str, brief: Brief, deps: AgentDeps, warnings: list[str]
) -> tuple[str, AIScore, _Verdict | None]:
    threshold = deps.settings.ai_score_threshold
    max_rounds = deps.settings.max_humanize_rounds
    contractions = "formal" not in brief.tone.lower()
    gate = _DetectorGate(deps.detector, max_rounds + 1, warnings)
    score = ai_score(text)
    if score.score >= threshold or deps.detector is not None:
        # Em dashes and missing contractions are habits, not wording; no model needed.
        text = _polish(text, contractions)
        score = ai_score(text)
    # With a detector, its verdict decides; the local score is the free fallback.
    verdict = await gate.check(text)

    def gating(local: AIScore, judged: _Verdict | None) -> float:
        return judged.score if judged else local.score

    settings = deps.settings
    if verdict and verdict.score >= settings.detector_rewrite_ceiling:
        # LLM rewrites don't move a detector this far off (live: 99.7 after two rounds), so
        # report the score instead of spending rounds and detector calls on it.
        warnings.append(
            f"AI detector ({verdict.provider}) score {verdict.score:.1f} is out of reach for "
            f"automatic rewriting (rewrites are tried below {settings.detector_rewrite_ceiling:g})"
        )
        return text, score, verdict

    rounds = 0
    while gating(score, verdict) >= threshold and rounds < max_rounds:
        rounds += 1
        blocks = split_blocks(text)
        targets = _targets(text, blocks, score, verdict.flagged if verdict else {})
        try:
            rewrite = await deps.llm.structured(
                tier="sonnet",
                system=prompts.HUMANIZER,
                prompt=_prompt(brief, text, blocks, targets),
                schema=Rewrite,
                effort="medium",
            )
        except LLMError as exc:
            warnings.append(f"Humanizer skipped; the draft was kept as written: {exc}")
            break

        candidate, rejected = _apply(blocks, targets, rewrite, brief.primary_keyword)
        candidate = _polish(candidate, contractions)
        if rejected:
            warnings.append(
                f"Humanizer rejected {len(rejected)} rewritten paragraph(s): "
                + "; ".join(sorted(set(rejected)))
            )
        if candidate == text:
            continue  # nothing changed: no need to spend a detector call
        candidate_score = ai_score(candidate)
        candidate_verdict = await gate.check(candidate) if verdict else None
        if verdict and candidate_verdict is None:
            verdict = None  # the detector dropped out: compare locally from here on
        gain = gating(score, verdict) - gating(candidate_score, candidate_verdict)
        if gain > 0:
            text, score, verdict = candidate, candidate_score, candidate_verdict
        if candidate_verdict and gain < settings.detector_min_gain:
            break  # judged by the detector, this round barely helped; another won't either

    final = gating(score, verdict)
    if final >= threshold:
        label = f"AI detector ({verdict.provider})" if verdict else "AI-likeness"
        warnings.append(
            f"{label} score {final:.1f} is still above the target of {threshold:g} "
            f"after {rounds} rewrite round(s)"
        )
    return text, score, verdict


def _targets(
    text: str, blocks: list[str], score: AIScore, detector_flags: dict[int, int]
) -> list[FlaggedParagraph]:
    flagged = flag_paragraphs(text)
    if detector_flags:
        # The detector points at the exact paragraphs; add the local findings for detail.
        by_index = {f.index: f for f in flagged}
        for index, count in detector_flags.items():
            target = by_index.setdefault(index, FlaggedParagraph(index=index, reasons=[]))
            target.reasons.append(
                f"an AI detector rates {count} of its sentences as machine-written; change their "
                "wording and rhythm, not just a few words"
            )
        return sorted(by_index.values(), key=lambda f: f.index)
    if score.signals.get("uniform_rhythm", 0) > 0:
        # Even pacing is spread over the whole article, so every paragraph with more than one
        # sentence has to change its rhythm, not just the ones that stand out.
        by_index = {f.index: f for f in flagged}
        for index, block in enumerate(blocks):
            if is_prose(block) and sentence_count(block) >= 2:
                target = by_index.setdefault(index, FlaggedParagraph(index=index, reasons=[]))
                target.reasons.append(_RHYTHM_REASON)
        return sorted(by_index.values(), key=lambda f: f.index)
    if flagged:
        return flagged
    # Nothing stands out on its own, so the article as a whole reads machine-written.
    top = [name for name, points in score.signals.items() if points > 0]
    reason = "the article as a whole reads machine-written (" + ", ".join(top) + ")"
    return [
        FlaggedParagraph(index=i, reasons=[reason]) for i, b in enumerate(blocks) if is_prose(b)
    ]


def _prompt(brief: Brief, text: str, blocks: list[str], targets: list[FlaggedParagraph]) -> str:
    paragraphs = "\n".join(
        f'<paragraph index="{t.index}" reasons="{escape("; ".join(t.reasons))}">\n'
        f"{blocks[t.index]}\n</paragraph>"
        for t in targets
    )
    prose = sum(1 for b in blocks if is_prose(b))
    if len(targets) * 2 <= prose:
        context = f"<untrusted_article>\n{text}\n</untrusted_article>"
    else:
        # Most of the article is being rewritten anyway; sending it twice would overflow small
        # tokens-per-minute quotas (Groq's free tier allows 8,000), so send its outline.
        outline = "\n".join(b for b in blocks if b.startswith("#"))
        context = f"<untrusted_outline>\n{outline}\n</untrusted_outline>"
    return (
        f"<brief>\nAudience: {brief.audience}\nTone: {brief.tone}\n</brief>\n\n"
        f"{context}\n\n"
        f"<paragraphs_to_rewrite>\n{paragraphs}\n</paragraphs_to_rewrite>"
    )


def _apply(
    blocks: list[str], targets: list[FlaggedParagraph], rewrite: Rewrite, keyword: str
) -> tuple[str, list[str]]:
    """Apply the safe rewrites; return the new text and why each unsafe one was rejected."""
    allowed = {t.index for t in targets}
    updated = list(blocks)
    rejected: list[str] = []
    for paragraph in rewrite.paragraphs:
        new_text = paragraph.text.strip()
        if paragraph.index not in allowed or not new_text:
            continue
        violations = rewrite_violations(blocks[paragraph.index], new_text, keyword=keyword)
        if violations:
            rejected.append("; ".join(violations))
            continue
        updated[paragraph.index] = new_text
    return "\n\n".join(updated), rejected
