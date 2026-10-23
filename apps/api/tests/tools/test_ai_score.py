import pytest

from cwas.tools.ai_score import ai_score, flag_paragraphs, split_blocks
from tests.samples import AI_PARAGRAPH, HUMAN_PARAGRAPH


def test_machine_typical_text_scores_high() -> None:
    result = ai_score(AI_PARAGRAPH)

    assert result.score >= 50
    assert result.signals["stock_phrases"] > 0
    assert result.signals["transitions"] > 0
    assert result.signals["uniform_rhythm"] > 0


def test_human_typical_text_scores_below_ten() -> None:
    assert ai_score(HUMAN_PARAGRAPH).score < 10


def test_score_is_deterministic() -> None:
    assert ai_score(AI_PARAGRAPH) == ai_score(AI_PARAGRAPH)


def test_markdown_structure_and_citations_do_not_affect_the_score() -> None:
    decorated = (
        "# Going solar\n\n"
        + HUMAN_PARAGRAPH.replace("in March.", "in March [2][3].").replace(
            "the inverter", "the [inverter](https://example.com/inverters)"
        )
        + "\n\n```\nconfig = 1\n```"
    )

    assert ai_score(decorated).score == ai_score(HUMAN_PARAGRAPH).score


def test_empty_text_scores_zero() -> None:
    assert ai_score("").score == 0
    assert ai_score("# Only a heading").score == 0


def test_heavy_em_dash_use_is_a_signal() -> None:
    dashed = HUMAN_PARAGRAPH.replace(", ", " — ")

    assert ai_score(dashed).signals["em_dashes"] > 0
    assert ai_score(HUMAN_PARAGRAPH).signals["em_dashes"] == 0


def test_list_heavy_text_is_a_signal() -> None:
    bullets = "\n".join(f"- Point number {i} about solar panels." for i in range(10))

    assert ai_score(bullets).signals["lists"] > 0


def test_long_text_without_contractions_is_a_signal() -> None:
    formal = " ".join(
        f"The installer {verb} the panels on the roof before noon on day {i}."
        for i, verb in enumerate(["checked", "cleaned", "mounted", "wired", "tested"] * 6)
    )

    assert ai_score(formal).signals["no_contractions"] > 0
    assert ai_score(HUMAN_PARAGRAPH).signals["no_contractions"] == 0


def test_score_never_exceeds_one_hundred() -> None:
    worst = "\n\n".join([AI_PARAGRAPH.replace(", ", " — ")] * 5)

    assert ai_score(worst).score <= 100


def test_split_blocks_round_trips() -> None:
    text = "# Title\n\nFirst para.\n\n\nSecond para.\n- a\n- b"

    blocks = split_blocks(text)

    assert blocks == ["# Title", "First para.", "Second para.\n- a\n- b"]


def test_flags_machine_typical_paragraphs_with_reasons_and_skips_headings() -> None:
    text = f"# Solar\n\n{HUMAN_PARAGRAPH}\n\n{AI_PARAGRAPH}"

    flagged = flag_paragraphs(text)

    assert [f.index for f in flagged] == [2]
    reasons = " ".join(flagged[0].reasons)
    assert "delve" in reasons
    assert "transition" in reasons


@pytest.mark.parametrize("block", ["# Heading with delve", "```\nlet's delve\n```", "| a | b |"])
def test_never_flags_non_prose_blocks(block: str) -> None:
    assert flag_paragraphs(block) == []
