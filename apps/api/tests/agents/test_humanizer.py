from typing import Any

import pytest

from cwas.agents import prompts
from cwas.agents.humanizer import humanizer, replace_em_dashes
from cwas.llm import LLMError
from cwas.models import Citation, Draft, Rewrite, RewrittenParagraph
from cwas.tools.ai_score import ai_score
from tests.fakes import FakeLLM, make_brief, make_deps
from tests.samples import AI_PARAGRAPH, HUMAN_PARAGRAPH

HEADING = "# Going solar"
MIXED = f"{HEADING}\n\n{HUMAN_PARAGRAPH}\n\n{AI_PARAGRAPH}"
# Same facts and citation as AI_PARAGRAPH, in a human rhythm.
FIXED = (
    "Solar's no longer a niche thing. Panels turn sunlight into power for the house, and the "
    "monthly bill drops [1]. Subsidies help. Upkeep? Almost none, once they're on the roof."
)


def state(markdown: str) -> dict[str, Any]:
    draft = Draft(
        variant_id="v1", markdown=markdown, citations=[Citation(number=1, url="u", title="t")]
    )
    return {"brief": make_brief(tone="casual"), "drafts": [draft]}


def rewrite(index: int, text: str) -> Rewrite:
    return Rewrite(paragraphs=[RewrittenParagraph(index=index, text=text)])


class FailingLLM(FakeLLM):
    async def structured(self, **kwargs: Any) -> Any:
        raise LLMError("every LLM provider failed")


async def test_human_text_passes_without_any_llm_call() -> None:
    llm = FakeLLM()
    text = f"{HEADING}\n\n{HUMAN_PARAGRAPH}"

    update = await humanizer(state(text), make_deps(llm=llm))

    [draft] = update["drafts"]
    assert draft.markdown == text
    assert draft.ai_score is not None
    assert draft.ai_score.score < 10
    assert llm.calls == []
    assert update["warnings"] == []


async def test_rewrites_only_flagged_paragraphs_and_reaches_the_target() -> None:
    llm = FakeLLM(structured={Rewrite: [rewrite(2, FIXED)]})

    update = await humanizer(state(MIXED), make_deps(llm=llm))

    [draft] = update["drafts"]
    assert draft.markdown == f"{HEADING}\n\n{HUMAN_PARAGRAPH}\n\n{FIXED}"
    assert draft.citations == state(MIXED)["drafts"][0].citations
    assert draft.ai_score is not None
    assert draft.ai_score.score < 10
    assert update["warnings"] == []

    [call] = llm.calls
    assert call["system"] == prompts.HUMANIZER
    assert call["schema"] is Rewrite
    assert '<paragraph index="2"' in call["prompt"]
    assert '<paragraph index="1"' not in call["prompt"]
    assert "delve" in call["prompt"]  # the reasons travel with the paragraph
    assert "casual" in call["prompt"]  # the brief's tone
    assert "<untrusted_article>" in call["prompt"]


async def test_rewrite_that_drops_a_citation_is_rejected() -> None:
    uncited = FIXED.replace(" [1]", "")
    llm = FakeLLM(structured={Rewrite: [rewrite(2, uncited)]})

    update = await humanizer(state(MIXED), make_deps(llm=llm, max_humanize_rounds=1))

    assert update["drafts"][0].markdown == MIXED
    assert any("citation" in w for w in update["warnings"])


async def test_rewrite_that_invents_a_citation_is_rejected() -> None:
    llm = FakeLLM(structured={Rewrite: [rewrite(2, FIXED + " Prices fell 40% [7].")]})

    update = await humanizer(state(MIXED), make_deps(llm=llm, max_humanize_rounds=1))

    assert update["drafts"][0].markdown == MIXED


async def test_rewrites_of_unflagged_or_unknown_paragraphs_are_ignored() -> None:
    llm = FakeLLM(
        structured={
            Rewrite: [
                Rewrite(
                    paragraphs=[
                        RewrittenParagraph(index=0, text="# New title"),
                        RewrittenParagraph(index=1, text="Changed [1]."),
                        RewrittenParagraph(index=99, text="Nowhere [1]."),
                        RewrittenParagraph(index=2, text=FIXED),
                    ]
                )
            ]
        }
    )

    update = await humanizer(state(MIXED), make_deps(llm=llm))

    assert update["drafts"][0].markdown == f"{HEADING}\n\n{HUMAN_PARAGRAPH}\n\n{FIXED}"


async def test_stops_after_max_rounds_and_warns_with_the_score() -> None:
    still_robotic = AI_PARAGRAPH.replace("Let's delve into", "Let us explore")
    llm = FakeLLM(structured={Rewrite: [rewrite(2, still_robotic)]})

    update = await humanizer(state(MIXED), make_deps(llm=llm, max_humanize_rounds=2))

    assert len(llm.calls) == 2
    [warning] = update["warnings"]
    assert "AI-likeness" in warning
    assert "10" in warning
    assert update["drafts"][0].ai_score.score >= 10  # type: ignore[union-attr]


async def test_a_rewrite_that_scores_worse_is_discarded() -> None:
    worse = AI_PARAGRAPH.replace(", ", " — ")
    llm = FakeLLM(structured={Rewrite: [rewrite(2, worse)]})

    update = await humanizer(state(MIXED), make_deps(llm=llm, max_humanize_rounds=1))

    assert update["drafts"][0].markdown == MIXED


async def test_llm_failure_keeps_the_draft_and_records_a_warning() -> None:
    update = await humanizer(state(MIXED), make_deps(llm=FailingLLM()))

    [draft] = update["drafts"]
    assert draft.markdown == MIXED
    assert draft.ai_score == ai_score(MIXED)
    assert any("humanizer" in w.lower() for w in update["warnings"])


async def test_threshold_is_configurable() -> None:
    llm = FakeLLM()

    update = await humanizer(state(MIXED), make_deps(llm=llm, ai_score_threshold=95.0))

    assert llm.calls == []
    assert update["drafts"][0].markdown == MIXED


async def test_writer_and_humanizer_share_the_style_rules() -> None:
    assert prompts.STYLE_RULES in prompts.WRITER
    assert prompts.STYLE_RULES in prompts.HUMANIZER


async def test_when_only_the_whole_article_reads_robotic_every_prose_paragraph_is_sent() -> None:
    sentence = "The installer {} the panels on the roof before noon on day {}."
    verbs = ["checked", "cleaned", "mounted"]
    paragraphs = [
        " ".join(sentence.format(v, p * 3 + i) for i, v in enumerate(verbs)) for p in range(4)
    ]
    text = "# Log\n\n" + "\n\n".join(paragraphs)
    assert ai_score(text).score >= 10
    llm = FakeLLM(structured={Rewrite: [Rewrite(paragraphs=[])]})

    await humanizer(state(text), make_deps(llm=llm, max_humanize_rounds=1))

    prompt = llm.calls[0]["prompt"]
    assert all(f'<paragraph index="{i}"' in prompt for i in range(1, 5))
    assert '<paragraph index="0"' not in prompt


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("a trend—it's smart", "a trend, it's smart"),
        ("panels — especially the best ones — work", "panels, especially the best ones, work"),
        ("savings within 3—5 years", "savings within 3–5 years"),
        ("plain text, no dashes", "plain text, no dashes"),
    ],
)
def test_replace_em_dashes(before: str, after: str) -> None:
    assert replace_em_dashes(before) == after


async def test_em_dashes_in_prose_are_replaced_without_an_llm_call() -> None:
    # Found in a live run: one em dash per paragraph is never flagged on its own, and small
    # models ignore the "use em dashes sparingly" rule, so the score stayed above 10.
    dashed = HUMAN_PARAGRAPH.replace(": one installer", " — one installer").replace(
        ", and that", " — and that"
    )
    extra = "Payback took 3—5 years [1], and the inverter — the part that fails — was replaced."
    text = f"# Solar — worth it?\n\n{dashed}\n\n{extra}"
    assert ai_score(text).score >= 10
    llm = FakeLLM()

    update = await humanizer(state(text), make_deps(llm=llm))

    [draft] = update["drafts"]
    assert llm.calls == []
    assert draft.markdown.startswith("# Solar — worth it?\n\n")  # headings are left alone
    assert "made no sense, one installer" in draft.markdown
    assert "3–5 years [1], and the inverter, the part that fails, was replaced." in draft.markdown
    assert draft.ai_score is not None
    assert draft.ai_score.score < 10
