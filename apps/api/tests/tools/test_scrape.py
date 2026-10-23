import httpx
import respx

from cwas.tools.scrape import InMemoryPageCache, Scraper

ARTICLE_HTML = """
<html><head><title>How Solar Panels Work</title></head>
<body>
<nav>Home | About | Contact</nav>
<article>
<h1>How Solar Panels Work</h1>
<p>Solar panels convert sunlight into electricity using photovoltaic cells made of silicon.
When photons hit the cell, they knock electrons loose and create a flow of current.</p>
<p>A typical residential system produces between 250 and 400 watts per panel, depending on
the efficiency of the cells and the amount of direct sunlight the roof receives each day.</p>
<p>Inverters then convert the direct current produced by the panels into alternating current
that household appliances can use, and any surplus can be exported back to the grid.</p>
<p>Maintenance is minimal: occasional cleaning and an annual inspection keep output high for
twenty five years or more, which is why panels are a popular long term investment.</p>
</article>
<footer>Copyright 2026</footer>
</body></html>
"""

ROBOTS_ALLOW = "User-agent: *\nAllow: /\n"
ROBOTS_DENY = "User-agent: *\nDisallow: /\n"


async def allow_all(url: str) -> bool:
    return True


def html(body: str = ARTICLE_HTML) -> httpx.Response:
    return httpx.Response(200, text=body, headers={"content-type": "text/html; charset=utf-8"})


@respx.mock
async def test_fetch_extracts_main_article_text_and_title() -> None:
    respx.get("https://solar.com/robots.txt").mock(
        return_value=httpx.Response(200, text=ROBOTS_ALLOW)
    )
    respx.get("https://solar.com/guide").mock(return_value=html())

    async with httpx.AsyncClient() as client:
        page = await Scraper(client, url_guard=allow_all).fetch(
            "https://solar.com/guide", "competitor"
        )

    assert page is not None
    assert page.url == "https://solar.com/guide"
    assert page.title == "How Solar Panels Work"
    assert "photovoltaic cells" in page.text
    assert "Home | About" not in page.text
    assert page.source_type == "competitor"


@respx.mock
async def test_fetch_respects_robots_disallow() -> None:
    respx.get("https://private.com/robots.txt").mock(
        return_value=httpx.Response(200, text=ROBOTS_DENY)
    )
    page_route = respx.get("https://private.com/post").mock(return_value=html())

    async with httpx.AsyncClient() as client:
        page = await Scraper(client, url_guard=allow_all).fetch("https://private.com/post")

    assert page is None
    assert not page_route.called


@respx.mock
async def test_missing_robots_txt_means_allowed() -> None:
    respx.get("https://norobots.com/robots.txt").mock(return_value=httpx.Response(404))
    respx.get("https://norobots.com/a").mock(return_value=html())

    async with httpx.AsyncClient() as client:
        page = await Scraper(client, url_guard=allow_all).fetch("https://norobots.com/a")

    assert page is not None


@respx.mock
async def test_robots_server_error_means_disallowed() -> None:
    respx.get("https://flaky.com/robots.txt").mock(return_value=httpx.Response(503))
    respx.get("https://flaky.com/a").mock(return_value=html())

    async with httpx.AsyncClient() as client:
        assert await Scraper(client, url_guard=allow_all).fetch("https://flaky.com/a") is None


@respx.mock
async def test_fetch_returns_none_on_http_error() -> None:
    respx.get("https://gone.com/robots.txt").mock(return_value=httpx.Response(404))
    respx.get("https://gone.com/a").mock(return_value=httpx.Response(404))

    async with httpx.AsyncClient() as client:
        assert await Scraper(client, url_guard=allow_all).fetch("https://gone.com/a") is None


@respx.mock
async def test_fetch_returns_none_for_non_html_content() -> None:
    respx.get("https://files.com/robots.txt").mock(return_value=httpx.Response(404))
    respx.get("https://files.com/a.pdf").mock(
        return_value=httpx.Response(
            200, content=b"%PDF", headers={"content-type": "application/pdf"}
        )
    )

    async with httpx.AsyncClient() as client:
        assert await Scraper(client, url_guard=allow_all).fetch("https://files.com/a.pdf") is None


@respx.mock
async def test_fetch_returns_none_on_network_error() -> None:
    respx.get("https://down.com/robots.txt").mock(return_value=httpx.Response(404))
    respx.get("https://down.com/a").mock(side_effect=httpx.ConnectError("down"))

    async with httpx.AsyncClient() as client:
        assert await Scraper(client, url_guard=allow_all).fetch("https://down.com/a") is None


@respx.mock
async def test_fetch_refuses_unsafe_url_without_any_request() -> None:
    async def deny(url: str) -> bool:
        return False

    route = respx.get("http://10.0.0.1/admin").mock(return_value=html())
    async with httpx.AsyncClient() as client:
        assert await Scraper(client, url_guard=deny).fetch("http://10.0.0.1/admin") is None
    assert not route.called


@respx.mock
async def test_fetch_does_not_follow_redirect_to_unsafe_url() -> None:
    async def guard(url: str) -> bool:
        return "169.254" not in url

    respx.get("https://evil.com/robots.txt").mock(return_value=httpx.Response(404))
    respx.get("https://evil.com/a").mock(
        return_value=httpx.Response(302, headers={"location": "http://169.254.169.254/meta"})
    )
    metadata = respx.get("http://169.254.169.254/meta").mock(return_value=html())

    async with httpx.AsyncClient() as client:
        assert await Scraper(client, url_guard=guard).fetch("https://evil.com/a") is None
    assert not metadata.called


@respx.mock
async def test_fetch_follows_safe_redirect() -> None:
    respx.get("https://old.com/robots.txt").mock(return_value=httpx.Response(404))
    respx.get("https://new.com/robots.txt").mock(return_value=httpx.Response(404))
    respx.get("https://old.com/a").mock(
        return_value=httpx.Response(301, headers={"location": "https://new.com/a"})
    )
    respx.get("https://new.com/a").mock(return_value=html())

    async with httpx.AsyncClient() as client:
        page = await Scraper(client, url_guard=allow_all).fetch("https://old.com/a")

    assert page is not None
    assert page.url == "https://new.com/a"


@respx.mock
async def test_fetch_uses_cache_on_second_call() -> None:
    respx.get("https://cached.com/robots.txt").mock(return_value=httpx.Response(404))
    route = respx.get("https://cached.com/a").mock(return_value=html())

    async with httpx.AsyncClient() as client:
        scraper = Scraper(client, cache=InMemoryPageCache(), url_guard=allow_all)
        first = await scraper.fetch("https://cached.com/a")
        second = await scraper.fetch("https://cached.com/a", "competitor")

    assert route.call_count == 1
    assert first is not None and second is not None
    assert second.text == first.text
    assert second.source_type == "competitor"


@respx.mock
async def test_fetch_returns_none_when_no_article_text_extracted() -> None:
    respx.get("https://empty.com/robots.txt").mock(return_value=httpx.Response(404))
    respx.get("https://empty.com/a").mock(return_value=html("<html><body></body></html>"))

    async with httpx.AsyncClient() as client:
        assert await Scraper(client, url_guard=allow_all).fetch("https://empty.com/a") is None


SEO_HTML = """
<html><head>
<title>Solar Cost Guide</title>
<meta name="description" content="What a home solar system costs in India, with subsidies.">
<script type="application/ld+json">{"@context": "https://schema.org", "@type": "Article"}</script>
<script type="application/ld+json">
{"@graph": [{"@type": ["FAQPage"]}, {"@type": "BreadcrumbList"}, {"@type": "Article"}]}
</script>
<script type="application/ld+json">{not valid json</script>
</head>
<body><article>
<h1>Solar Cost Guide</h1>
<p>Home solar has become far cheaper over the last decade, and the government now pays part
of the bill for most households that install a rooftop system with net metering.</p>
<h2>How much does a 3 kW system cost?</h2>
<p>A 3 kW rooftop system costs about 1.5 lakh rupees before the subsidy is applied, and most
installers quote a payback period of four to six years for an average family home.</p>
<h3>Subsidy</h3>
<p>The PM Surya Ghar scheme pays up to 78,000 rupees for systems of 3 kW or more, credited to
the bank account of the homeowner after the installation passes inspection.</p>
</article></body></html>
"""


@respx.mock
async def test_fetch_keeps_headings_and_captures_seo_metadata() -> None:
    respx.get("https://rival.com/robots.txt").mock(return_value=httpx.Response(404))
    respx.get("https://rival.com/cost").mock(return_value=html(SEO_HTML))

    async with httpx.AsyncClient() as client:
        page = await Scraper(client, url_guard=allow_all).fetch("https://rival.com/cost")

    assert page is not None
    assert "## How much does a 3 kW system cost?" in page.text
    assert "### Subsidy" in page.text
    assert page.description == "What a home solar system costs in India, with subsidies."
    assert page.schema_types == ["Article", "BreadcrumbList", "FAQPage"]


@respx.mock
async def test_pages_without_metadata_get_empty_defaults() -> None:
    respx.get("https://solar.com/robots.txt").mock(return_value=httpx.Response(404))
    respx.get("https://solar.com/guide").mock(return_value=html())

    async with httpx.AsyncClient() as client:
        page = await Scraper(client, url_guard=allow_all).fetch("https://solar.com/guide")

    assert page is not None
    assert page.description == ""
    assert page.schema_types == []
