from cwas.agents.grammar import grammar_agent
from cwas.models import Citation, Draft
from cwas.tools.grammar import GrammarError, GrammarMatch
from tests.fakes import FakeGrammar, make_deps


def draft(text: str) -> Draft:
    return Draft(variant_id="v1", markdown=text, citations=[Citation(number=1, url="u", title="t")])


def teh_fixer(text: str) -> list[GrammarMatch]:
    i = text.index("teh")
    return [GrammarMatch(offset=i, length=3, message="typo", replacements=["the"], issue_type="misspelling")]


async def test_applies_corrections_and_scores_readability() -> None:
    deps = make_deps(grammar=FakeGrammar(teh_fixer))

    update = await grammar_agent({"drafts": [draft("This is teh plan [1].")]}, deps)

    [fixed] = update["drafts"]
    assert fixed.markdown == "This is the plan [1]."
    assert fixed.citations == draft("x").citations
    assert fixed.readability is not None
    assert fixed.readability.word_count > 0


async def test_checker_failure_keeps_text_and_records_warning() -> None:
    deps = make_deps(grammar=FakeGrammar(GrammarError("down")))

    update = await grammar_agent({"drafts": [draft("This is teh plan.")]}, deps)

    assert update["drafts"][0].markdown == "This is teh plan."
    assert update["drafts"][0].readability is not None
    assert any("LanguageTool" in w for w in update["warnings"])


async def test_without_checker_only_scores_readability() -> None:
    update = await grammar_agent({"drafts": [draft("Short text here.")]}, make_deps(grammar=None))

    assert update["drafts"][0].markdown == "Short text here."
    assert update["drafts"][0].readability is not None
    assert update["warnings"] == []
