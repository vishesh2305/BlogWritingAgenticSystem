from cwas.tools.guards import citations, keyword_count, rewrite_violations

ORIGINAL = "Home solar panels cost about 78,000 rupees after subsidy [1]. Payback takes years [2]."


def test_citations_are_the_bracketed_source_numbers() -> None:
    assert citations("a [1] b [2][3] see [the guide](https://x.com) again [1]") == {1, 2, 3}


def test_keyword_count_is_case_insensitive_and_whole_word() -> None:
    text = "Home solar panels work. HOME SOLAR PANELS pay. home solar panelsets do not count."

    assert keyword_count(text, "home solar panels") == 2
    assert keyword_count(text, "") == 0


def test_a_faithful_rewrite_has_no_violations() -> None:
    rewritten = (
        "After the subsidy, home solar panels run you about 78000 rupees [1]. It pays back [2]."
    )

    assert rewrite_violations(ORIGINAL, rewritten, keyword="home solar panels") == []


def test_changed_citations_are_a_violation() -> None:
    [violation] = rewrite_violations(ORIGINAL, ORIGINAL.replace(" [2]", " [3]"))

    assert "citation" in violation


def test_new_numbers_are_a_violation_unless_allowed() -> None:
    rewritten = ORIGINAL + " About 40% of homes qualify, and 3 kW is typical."

    [violation] = rewrite_violations(ORIGINAL, rewritten)
    assert "40" in violation
    assert "3" in violation
    assert rewrite_violations(ORIGINAL, rewritten, allowed_numbers={"40", "3"}) == []


def test_dropping_the_keyword_is_a_violation() -> None:
    rewritten = ORIGINAL.replace("Home solar panels", "Rooftop systems")

    [violation] = rewrite_violations(ORIGINAL, rewritten, keyword="home solar panels")
    assert "keyword" in violation
    assert rewrite_violations(ORIGINAL, rewritten) == []
