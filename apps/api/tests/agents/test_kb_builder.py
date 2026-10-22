from datetime import UTC, datetime

from cwas.agents.kb_builder import kb_builder
from cwas.models import ScrapedPage
from tests.fakes import make_deps


def page(url: str, text: str) -> ScrapedPage:
    return ScrapedPage(url=url, title=url, text=text, fetched_at=datetime.now(UTC))


async def test_ingests_sources_into_the_run_collection() -> None:
    deps = make_deps()
    sources = [
        page("https://a.com", "Inverters convert DC to AC power for the home."),
        page("https://b.com", "Net metering credits exported solar electricity."),
    ]

    update = await kb_builder({"run_id": "run-7", "sources": sources}, deps)

    assert update == {"kb_chunks": 2}
    hits = await deps.kb.retrieve("run-7", "net metering credits", k=1)
    assert hits[0].chunk.source_url == "https://b.com"
