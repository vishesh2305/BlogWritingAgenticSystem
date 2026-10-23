from datetime import UTC, datetime

from cwas.kb.facts import extract_facts
from cwas.models import ScrapedPage, SourceType


def page(url: str, text: str, source_type: SourceType = "source") -> ScrapedPage:
    return ScrapedPage(
        url=url, title=f"T {url}", text=text, source_type=source_type, fetched_at=datetime.now(UTC)
    )


COST_PAGE = """# Solar costs

A 3 kW rooftop system costs about 1.5 lakh rupees before the subsidy is applied. Panels \
are a good idea for most families who own their roof.

## Subsidy

- The PM Surya Ghar scheme pays up to **78,000 rupees** for systems of 3 kW or more.
- Installers are listed on the national portal.

| Size | Cost |
| 3 kW | 1.5 lakh |

Prices fell 40% between 2015 and 2024 according to industry data."""


def test_keeps_only_sentences_that_carry_numbers() -> None:
    facts = extract_facts([page("https://a.com", COST_PAGE)])

    assert [f.text for f in facts] == [
        "A 3 kW rooftop system costs about 1.5 lakh rupees before the subsidy is applied.",
        "The PM Surya Ghar scheme pays up to 78,000 rupees for systems of 3 kW or more.",
        "Prices fell 40% between 2015 and 2024 according to industry data.",
    ]


def test_facts_carry_ids_sources_and_normalised_numbers() -> None:
    [first, second, _] = extract_facts([page("https://a.com", COST_PAGE, "competitor")])

    assert first.id == "f1"
    assert second.id == "f2"
    assert first.source_url == "https://a.com"
    assert first.source_title == "T https://a.com"
    assert first.source_type == "competitor"
    assert first.numbers == ["3", "1.5"]
    assert second.numbers == ["78000", "3"]


def test_skips_fragments_and_run_on_text() -> None:
    long = "In 2024 " + "the market grew and grew " * 15 + "fast."
    text = f"Up 5% today.\n\n{long}"

    assert extract_facts([page("https://a.com", text)]) == []


def test_duplicates_across_pages_are_kept_once() -> None:
    same = "Panels last about 25 years with very little maintenance needed."

    facts = extract_facts([page("https://a.com", same), page("https://b.com", same)])

    assert [f.source_url for f in facts] == ["https://a.com"]


def test_caps_facts_per_page() -> None:
    text = " ".join(f"Panel model {i} produces {300 + i} watts in full sun." for i in range(10))

    facts = extract_facts([page("https://a.com", text), page("https://b.com", COST_PAGE)], 4)

    assert [f.source_url for f in facts].count("https://a.com") == 4
    assert [f.id for f in facts] == [f"f{i}" for i in range(1, 8)]


def test_skips_list_introductions_and_year_only_sentences() -> None:
    # Both came from a live run: neither is a statistic anyone could cite.
    text = (
        "This guide breaks down everything you need to know, with realistic numbers for 2026. "
        "In 2026, rooftop solar costs approximately:\n"
        "- 45,000 to 60,000 rupees per kW before the subsidy is applied.\n"
        "In 2024 India added 15 GW of rooftop and utility solar capacity."
    )

    facts = extract_facts([page("https://a.com", text)])

    assert [f.text for f in facts] == [
        "45,000 to 60,000 rupees per kW before the subsidy is applied.",
        "In 2024 India added 15 GW of rooftop and utility solar capacity.",
    ]
