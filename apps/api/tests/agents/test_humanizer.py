from typing import Any

import pytest

from cwas.agents import prompts
from cwas.agents.humanizer import add_contractions, humanizer, replace_em_dashes
from cwas.llm import LLMError
from cwas.models import Citation, Draft, Rewrite, RewrittenParagraph
from cwas.tools.ai_score import ai_score
from tests.fakes import FakeDetector, FakeLLM, failing_detector, make_brief, make_deps
from tests.samples import AI_PARAGRAPH, HUMAN_PARAGRAPH, UNCONTRACTED

HEADING = "# Going solar"
MIXED = f"{HEADING}\n\n{HUMAN_PARAGRAPH}\n\n{AI_PARAGRAPH}"
# What the no-model polish (contractions, em dashes) makes of MIXED before any rewrite.
POLISHED = MIXED.replace("It is important", "It's important")
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

    assert update["drafts"][0].markdown == POLISHED
    assert any("citation" in w for w in update["warnings"])


async def test_rewrite_that_invents_a_citation_is_rejected() -> None:
    llm = FakeLLM(structured={Rewrite: [rewrite(2, FIXED + " Prices fell 40% [7].")]})

    update = await humanizer(state(MIXED), make_deps(llm=llm, max_humanize_rounds=1))

    assert update["drafts"][0].markdown == POLISHED


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

    assert update["drafts"][0].markdown == POLISHED


async def test_llm_failure_keeps_the_draft_and_records_a_warning() -> None:
    update = await humanizer(state(MIXED), make_deps(llm=FailingLLM()))

    [draft] = update["drafts"]
    assert draft.markdown == POLISHED
    assert draft.ai_score == ai_score(POLISHED)
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


async def test_rewrite_that_adds_a_number_is_rejected() -> None:
    llm = FakeLLM(structured={Rewrite: [rewrite(2, FIXED + " About 40% of homes qualify.")]})

    update = await humanizer(state(MIXED), make_deps(llm=llm, max_humanize_rounds=1))

    assert update["drafts"][0].markdown == POLISHED
    assert any("40" in w for w in update["warnings"])


async def test_rewrite_that_drops_the_primary_keyword_is_rejected() -> None:
    with_keyword = MIXED.replace("solar panels convert", "home solar panels convert")
    llm = FakeLLM(structured={Rewrite: [rewrite(2, FIXED)]})

    update = await humanizer(state(with_keyword), make_deps(llm=llm, max_humanize_rounds=1))

    assert update["drafts"][0].markdown == with_keyword.replace("It is", "It's")
    assert any("keyword" in w for w in update["warnings"])


async def test_rewrite_that_keeps_the_primary_keyword_is_accepted() -> None:
    with_keyword = MIXED.replace("solar panels convert", "home solar panels convert")
    kept = FIXED.replace("Panels turn", "Home solar panels turn")
    llm = FakeLLM(structured={Rewrite: [rewrite(2, kept)]})

    update = await humanizer(state(with_keyword), make_deps(llm=llm))

    assert update["drafts"][0].markdown.endswith(kept)


WORDS = [
    "the",
    "crew",
    "checked",
    "every",
    "panel",
    "on",
    "the",
    "north",
    "roof",
    "and",
    "then",
    "tested",
    "the",
    "new",
    "inverter",
]


def even_paragraph(lengths: tuple[int, ...]) -> str:
    return " ".join(" ".join(WORDS[:n]).capitalize() + "." for n in lengths)


async def test_an_evenly_paced_article_also_gets_its_short_even_paragraphs_rewritten() -> None:
    # Live run: AI-likeness 13.0, almost all of it uniform_rhythm, yet only the two paragraphs
    # with 4+ sentences were flagged, so a rewrite round could not bring the score down.
    even = [even_paragraph((10, 12, 14)), even_paragraph((11, 13, 12))]
    short = even_paragraph((12,))  # a single sentence has no rhythm to fix
    text = "\n\n".join(["# Log", AI_PARAGRAPH, *even, short])
    assert ai_score(text).signals["uniform_rhythm"] > 0
    llm = FakeLLM(structured={Rewrite: [Rewrite(paragraphs=[])]})

    await humanizer(state(text), make_deps(llm=llm, max_humanize_rounds=1))

    prompt = llm.calls[0]["prompt"]
    assert '<paragraph index="1"' in prompt  # the stock-phrase paragraph
    assert '<paragraph index="2"' in prompt  # 3-sentence, evenly paced
    assert '<paragraph index="3"' in prompt
    assert '<paragraph index="4"' not in prompt  # one sentence: left alone


async def test_an_evenly_paced_article_gets_every_multi_sentence_paragraph_rewritten() -> None:
    # Live run: AI-likeness 23.4, uniform_rhythm 20.7 spread over many short paragraphs, many
    # glued under their headings; only 1 of 46 blocks was sent for rewriting.
    text = "\n\n".join(
        [
            "# Guide",
            "## Costs\n" + even_paragraph((12, 13)),
            "## Savings\n" + even_paragraph((13, 12)),
            even_paragraph((12, 14)),
            even_paragraph((13,)),
        ]
    )
    assert ai_score(text).signals["uniform_rhythm"] > 0
    llm = FakeLLM(structured={Rewrite: [Rewrite(paragraphs=[])]})

    await humanizer(state(text), make_deps(llm=llm, max_humanize_rounds=1))

    prompt = llm.calls[0]["prompt"]
    sent = [i for i in range(8) if f'<paragraph index="{i}"' in prompt]
    assert sent == [2, 4, 5]  # the bodies under both headings and the free paragraph
    assert "under 6 words" in prompt  # a concrete instruction, not just "too even"


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("It is cheap and you are covered.", "It's cheap and you're covered."),
        ("Do not wait. It does not matter.", "Don't wait. It doesn't matter."),
        ("Solar is not magic, and it is not cheap.", "Solar isn't magic, and it isn't cheap."),
        ("You cannot and will not regret it.", "You can't and won't regret it."),
        ("There is a catch. That is the part.", "There's a catch. That's the part."),
        ("Leave it as it is. That is, roughly.", "Leave it as it is. That is, roughly."),
        ("Let us start; I am sure.", "Let's start; I'm sure."),
    ],
)
def test_add_contractions(before: str, after: str) -> None:
    assert add_contractions(before) == after


async def test_missing_contractions_are_added_without_an_llm_call() -> None:
    # Live runs ended at 13.0 and 13.1 with no_contractions worth 3.8 to 6.7 points.
    text = f"## What it is like\n\n{UNCONTRACTED}"
    assert ai_score(text).signals["no_contractions"] > 0
    llm = FakeLLM()

    update = await humanizer(state(text), make_deps(llm=llm, ai_score_threshold=5.0))

    [draft] = update["drafts"]
    assert llm.calls == []
    assert draft.markdown.startswith("## What it is like\n\n")  # headings keep their wording
    assert "Solar isn't magic. It's a roof" in draft.markdown
    assert "most families don't notice" in draft.markdown.lower()
    assert "under 900 in the first summer [1]" in draft.markdown
    assert draft.ai_score is not None
    assert draft.ai_score.signals["no_contractions"] == 0


async def test_a_formal_brief_keeps_its_uncontracted_style() -> None:
    llm = FakeLLM(structured={Rewrite: [Rewrite(paragraphs=[])]})
    deps = make_deps(llm=llm, ai_score_threshold=5.0, max_humanize_rounds=1)
    formal = state(UNCONTRACTED)
    formal["brief"] = make_brief(tone="formal and academic")

    update = await humanizer(formal, deps)

    assert "isn't" not in update["drafts"][0].markdown


async def test_rewriting_most_paragraphs_sends_an_outline_not_the_article_twice() -> None:
    # Live run: the whole article plus every paragraph again overflowed Groq's free tier
    # (8,000 tokens/min), so the fallback could not take the humanizer call.
    body = even_paragraph((12, 13))
    text = "\n\n".join(
        ["# Guide", "## Costs\n" + body, "## Savings\n" + even_paragraph((13, 12)), body + " Ok."]
    )
    llm = FakeLLM(structured={Rewrite: [Rewrite(paragraphs=[])]})

    await humanizer(state(text), make_deps(llm=llm, max_humanize_rounds=1))

    prompt = llm.calls[0]["prompt"]
    assert "<untrusted_article>" not in prompt
    assert "<untrusted_outline>\n# Guide\n## Costs\n## Savings\n</untrusted_outline>" in prompt
    assert prompt.count(body) == 2  # once per paragraph that contains it, not again as context


# --- a real AI detector in the loop -------------------------------------------------------


async def test_a_detector_score_under_target_ends_after_one_call_without_an_llm() -> None:
    detector, llm = FakeDetector([4.0]), FakeLLM()

    update = await humanizer(state(MIXED), make_deps(llm=llm, detector=detector))

    [draft] = update["drafts"]
    assert llm.calls == []
    assert len(detector.texts) == 1
    assert "# Going solar" not in detector.texts[0]  # prose only: headings are not sent
    assert draft.detector_score is not None
    assert (draft.detector_score.provider, draft.detector_score.score) == ("fake", 4.0)
    assert update["warnings"] == []


async def test_paragraphs_the_detector_flags_are_rewritten_then_rechecked() -> None:
    detector = FakeDetector([60.0, 5.0], flag="Inverters")  # flags the human paragraph
    rewritten = HUMAN_PARAGRAPH.replace(
        "Inverters, it turned out.", "The inverters, it turned out."
    )
    llm = FakeLLM(structured={Rewrite: [rewrite(1, rewritten)]})

    update = await humanizer(state(MIXED), make_deps(llm=llm, detector=detector))

    [draft] = update["drafts"]
    prompt = llm.calls[0]["prompt"]
    assert '<paragraph index="1"' in prompt  # flagged by the detector, not the local score
    assert "AI detector" in prompt
    assert len(detector.texts) == 2
    assert rewritten in draft.markdown
    assert draft.detector_score.score == 5.0  # type: ignore[union-attr]
    assert update["warnings"] == []


async def test_a_round_the_detector_does_not_score_lower_is_discarded() -> None:
    detector = FakeDetector([60.0, 70.0], flag="Inverters")
    llm = FakeLLM(structured={Rewrite: [rewrite(2, FIXED)]})

    update = await humanizer(
        state(MIXED), make_deps(llm=llm, detector=detector, max_humanize_rounds=1)
    )

    [draft] = update["drafts"]
    assert draft.markdown == POLISHED
    assert draft.detector_score.score == 60.0  # type: ignore[union-attr]
    assert any("AI detector" in w and "60" in w for w in update["warnings"])


async def test_detector_calls_are_capped_at_one_per_round_plus_one() -> None:
    detector = FakeDetector([60.0], flag="Inverters")
    llm = FakeLLM(structured={Rewrite: [Rewrite(paragraphs=[])]})

    await humanizer(state(MIXED), make_deps(llm=llm, detector=detector, max_humanize_rounds=2))

    assert len(detector.texts) == 3


async def test_a_failing_detector_falls_back_to_the_local_score_and_is_not_retried() -> None:
    detector = failing_detector()
    llm = FakeLLM(structured={Rewrite: [rewrite(2, FIXED)]})

    update = await humanizer(state(MIXED), make_deps(llm=llm, detector=detector))

    [draft] = update["drafts"]
    assert draft.markdown == f"{HEADING}\n\n{HUMAN_PARAGRAPH}\n\n{FIXED}"
    assert draft.detector_score is None
    assert len(detector.texts) == 1
    assert any("detector" in w.lower() for w in update["warnings"])
