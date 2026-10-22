import httpx
import pytest
import respx

from cwas.tools.grammar import (
    GrammarError,
    GrammarMatch,
    LanguageToolChecker,
    apply_corrections,
    readability,
)

LT_URL = "http://lt:8010/v2/check"


def match(
    offset: int, length: int, replacement: str | None, issue: str = "grammar"
) -> GrammarMatch:
    return GrammarMatch(
        offset=offset,
        length=length,
        message="msg",
        replacements=[replacement] if replacement else [],
        issue_type=issue,
    )


@respx.mock
async def test_check_parses_languagetool_matches() -> None:
    route = respx.post(LT_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "matches": [
                    {
                        "message": "Possible agreement error",
                        "offset": 5,
                        "length": 3,
                        "replacements": [{"value": "are"}, {"value": "were"}],
                        "rule": {"id": "AGREEMENT", "issueType": "grammar"},
                    }
                ]
            },
        )
    )
    async with httpx.AsyncClient() as client:
        matches = await LanguageToolChecker(client, "http://lt:8010").check("They is here.")

    assert matches == [
        GrammarMatch(
            offset=5,
            length=3,
            message="Possible agreement error",
            replacements=["are", "were"],
            issue_type="grammar",
        )
    ]
    sent = route.calls.last.request.content.decode()
    assert "language=en-US" in sent


@respx.mock
async def test_check_raises_grammar_error_when_service_unavailable() -> None:
    respx.post(LT_URL).mock(side_effect=httpx.ConnectError("down"))
    async with httpx.AsyncClient() as client:
        with pytest.raises(GrammarError):
            await LanguageToolChecker(client, "http://lt:8010").check("text")


def test_apply_corrections_uses_first_replacement_and_keeps_offsets_valid() -> None:
    text = "They is here and it are good."
    result = apply_corrections(text, [match(5, 2, "are"), match(20, 3, "is")])
    assert result == "They are here and it is good."


def test_apply_corrections_skips_matches_without_replacements() -> None:
    assert apply_corrections("Hello world", [match(0, 5, None)]) == "Hello world"


def test_apply_corrections_skips_overlapping_matches() -> None:
    text = "abcdef"
    result = apply_corrections(text, [match(0, 3, "X"), match(2, 2, "Y")])
    assert result == "Xdef"


def test_apply_corrections_ignores_style_and_whitespace_issues() -> None:
    text = "Very very good."
    assert apply_corrections(text, [match(0, 9, "Very", issue="style")]) == text
    assert apply_corrections(text, [match(4, 1, "  ", issue="whitespace")]) == text


def test_apply_corrections_does_not_touch_capitalised_words_flagged_as_misspelled() -> None:
    text = "Use Kubernetes with Anthropic."
    flagged = [match(4, 10, "Kubernetis", issue="misspelling")]
    assert apply_corrections(text, flagged) == text


def test_apply_corrections_fixes_lowercase_misspellings() -> None:
    text = "This is teh best."
    assert apply_corrections(text, [match(8, 3, "the", issue="misspelling")]) == "This is the best."


def test_apply_corrections_never_edits_citations_links_or_code() -> None:
    text = "See [1] and [docs](https://x.io/a_b) and `foo_bar`."
    protected = [
        match(4, 3, "[2]"),
        match(26, 9, "a-b"),
        match(42, 9, "foo bar"),
    ]
    assert apply_corrections(text, protected) == text


def test_readability_reports_scores_and_word_count() -> None:
    score = readability("The cat sat on the mat. It was a sunny day.")
    assert score.word_count == 11
    assert score.flesch_reading_ease > 80
    assert score.grade_level < 6


def test_readability_of_empty_text_is_zero() -> None:
    score = readability("")
    assert score.word_count == 0
    assert score.flesch_reading_ease == 0.0
