import pytest

from cwas.agents.researcher import ResearchError, researcher
from tests.fakes import FakeScraper, FakeSearch, make_brief, make_deps

TEXT = "Solar panels convert sunlight to electricity."


async def test_tags_ranking_pages_as_competitors_and_dedupes_urls() -> None:
    brief = make_brief(search_queries=["q1", "q2"], primary_keyword="solar")
    search = FakeSearch(
        {
            "q1": ["https://facts.org/a", "https://rival.com/blog"],
            "q2": ["https://facts.org/a", "https://gov.uk/b"],
            "solar": ["https://rival.com/blog"],
        }
    )
    scraper = FakeScraper(
        {u: TEXT for u in ["https://facts.org/a", "https://gov.uk/b", "https://rival.com/blog"]}
    )

    update = await researcher({"brief": brief}, make_deps(search=search, scraper=scraper))

    by_url = {p.url: p.source_type for p in update["sources"]}
    assert by_url == {
        "https://facts.org/a": "source",
        "https://gov.uk/b": "source",
        "https://rival.com/blog": "competitor",
    }
    assert sorted(scraper.fetched) == sorted(by_url)


async def test_pages_that_fail_to_scrape_are_dropped() -> None:
    brief = make_brief(search_queries=["q"], primary_keyword="kw")
    search = FakeSearch({"q": ["https://ok.com", "https://blocked.com"]})
    scraper = FakeScraper({"https://ok.com": TEXT})

    update = await researcher({"brief": brief}, make_deps(search=search, scraper=scraper))

    assert [p.url for p in update["sources"]] == ["https://ok.com"]


async def test_a_failing_query_becomes_a_warning() -> None:
    brief = make_brief(search_queries=["bad", "good"], primary_keyword="kw")
    search = FakeSearch({"good": ["https://ok.com"]}, failing={"bad"})
    scraper = FakeScraper({"https://ok.com": TEXT})

    update = await researcher({"brief": brief}, make_deps(search=search, scraper=scraper))

    assert len(update["sources"]) == 1
    assert any("bad" in w for w in update["warnings"])


async def test_raises_when_no_page_could_be_collected() -> None:
    brief = make_brief(search_queries=["q"], primary_keyword="kw")
    search = FakeSearch({"q": ["https://nothing.com"]})

    with pytest.raises(ResearchError):
        await researcher({"brief": brief}, make_deps(search=search, scraper=FakeScraper({})))


async def test_respects_query_and_source_limits() -> None:
    brief = make_brief(search_queries=[f"q{i}" for i in range(10)], primary_keyword="kw")
    urls = {f"q{i}": [f"https://s{i}-{j}.com" for j in range(5)] for i in range(10)}
    search = FakeSearch(urls)
    scraper = FakeScraper({u: TEXT for group in urls.values() for u in group})

    deps = make_deps(search=search, scraper=scraper, max_search_queries=3, max_sources=4)
    update = await researcher({"brief": brief}, deps)

    assert set(search.queries) == {"q0", "q1", "q2", "kw"}
    assert len([p for p in update["sources"] if p.source_type == "source"]) == 4


async def test_falls_back_to_topic_when_brief_has_no_queries() -> None:
    brief = make_brief(search_queries=[], topic="heat pumps", primary_keyword="kw")
    search = FakeSearch({"heat pumps": ["https://hp.com"]})
    scraper = FakeScraper({"https://hp.com": TEXT})

    await researcher({"brief": brief}, make_deps(search=search, scraper=scraper))

    assert "heat pumps" in search.queries


async def test_warns_when_no_competitor_page_could_be_collected() -> None:
    # Live run: the keyword search found nothing usable, and the run went on silently.
    search = FakeSearch({"how do solar panels work": ["https://facts.org/a"]})
    scraper = FakeScraper({"https://facts.org/a": "Panels convert sunlight."})
    deps = make_deps(search=search, scraper=scraper)

    update = await researcher({"brief": make_brief()}, deps)

    assert [p.source_type for p in update["sources"]] == ["source"]
    assert any("competitor" in w for w in update["warnings"])
