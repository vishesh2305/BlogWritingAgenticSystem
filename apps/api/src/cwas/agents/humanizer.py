"""Agent 7 - Humanizer: rewrites machine-sounding paragraphs until the AI-likeness score is low.

Only flagged prose paragraphs are rewritten, so headings, code and the rest of the article
stay byte-for-byte intact. A rewrite that changes a paragraph's citations, adds a number or
drops the primary keyword is rejected, and a round that does not lower the score is discarded.
"""

import re
from html import escape
from typing import Any

from cwas.agents import prompts
from cwas.agents.deps import AgentDeps
from cwas.graph.state import RunState
from cwas.llm import LLMError
from cwas.models import AIScore, Brief, FlaggedParagraph, Rewrite
from cwas.tools.ai_score import (
    ai_score,
    flag_paragraphs,
    is_prose,
    sentence_count,
    split_blocks,
)
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


def _without_em_dashes(text: str) -> str:
    if "—" not in text:
        return text
    return "\n\n".join(replace_em_dashes(b) if is_prose(b) else b for b in split_blocks(text))


async def humanizer(state: RunState, deps: AgentDeps) -> dict[str, Any]:
    warnings: list[str] = []
    drafts = []
    for draft in state["drafts"]:
        text, score = await _humanize(draft.markdown, state["brief"], deps, warnings)
        drafts.append(draft.model_copy(update={"markdown": text, "ai_score": score}))
    return {"drafts": drafts, "warnings": warnings}


async def _humanize(
    text: str, brief: Brief, deps: AgentDeps, warnings: list[str]
) -> tuple[str, AIScore]:
    threshold = deps.settings.ai_score_threshold
    score = ai_score(text)
    if score.score >= threshold:
        # Em dashes are a punctuation habit, not wording; fixing them needs no model.
        text = _without_em_dashes(text)
        score = ai_score(text)
    rounds = 0
    while score.score >= threshold and rounds < deps.settings.max_humanize_rounds:
        rounds += 1
        blocks = split_blocks(text)
        targets = _targets(text, blocks, score)
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
        candidate = _without_em_dashes(candidate)
        if rejected:
            warnings.append(
                f"Humanizer rejected {len(rejected)} rewritten paragraph(s): "
                + "; ".join(sorted(set(rejected)))
            )
        candidate_score = ai_score(candidate)
        if candidate_score.score < score.score:
            text, score = candidate, candidate_score

    if score.score >= threshold:
        warnings.append(
            f"AI-likeness score {score.score:.1f} is still above the target of {threshold:g} "
            f"after {rounds} rewrite round(s)"
        )
    return text, score


def _targets(text: str, blocks: list[str], score: AIScore) -> list[FlaggedParagraph]:
    flagged = flag_paragraphs(text)
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
    return (
        f"<brief>\nAudience: {brief.audience}\nTone: {brief.tone}\n</brief>\n\n"
        f"<untrusted_article>\n{text}\n</untrusted_article>\n\n"
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
