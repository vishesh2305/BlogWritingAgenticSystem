"""Polite, SSRF-safe page fetching with main-content extraction."""

import logging
from datetime import UTC, datetime
from typing import Protocol
from urllib.parse import urljoin, urlsplit
from urllib.robotparser import RobotFileParser

import httpx
import trafilatura

from cwas.models import ScrapedPage, SourceType
from cwas.tools.url_safety import UrlGuard, is_public_url

logger = logging.getLogger(__name__)

MAX_REDIRECTS = 5
REDIRECT_CODES = {301, 302, 303, 307, 308}


class PageCache(Protocol):
    async def get(self, url: str) -> ScrapedPage | None: ...
    async def set(self, url: str, page: ScrapedPage) -> None: ...


class InMemoryPageCache:
    def __init__(self) -> None:
        self._pages: dict[str, ScrapedPage] = {}

    async def get(self, url: str) -> ScrapedPage | None:
        return self._pages.get(url)

    async def set(self, url: str, page: ScrapedPage) -> None:
        self._pages[url] = page


class Scraper:
    def __init__(
        self,
        client: httpx.AsyncClient,
        cache: PageCache | None = None,
        url_guard: UrlGuard = is_public_url,
        user_agent: str = "CWASResearchBot/0.1",
        timeout: float = 20.0,
    ) -> None:
        self._client = client
        self._cache = cache
        self._url_guard = url_guard
        self._user_agent = user_agent
        self._timeout = timeout
        self._robots: dict[str, RobotFileParser | None] = {}

    async def fetch(self, url: str, source_type: SourceType = "source") -> ScrapedPage | None:
        if self._cache and (cached := await self._cache.get(url)):
            return cached.model_copy(update={"source_type": source_type})

        response = await self._get_following_safe_redirects(url)
        if response is None:
            return None

        content_type = response.headers.get("content-type", "")
        if "html" not in content_type:
            logger.info("skip %s: content-type %s", url, content_type)
            return None

        text = trafilatura.extract(response.text, include_comments=False, include_tables=True)
        if not text:
            return None
        metadata = trafilatura.extract_metadata(response.text)
        title = (metadata.title if metadata and metadata.title else "") or str(response.url)

        page = ScrapedPage(
            url=str(response.url),
            title=title,
            text=text,
            source_type=source_type,
            fetched_at=datetime.now(UTC),
        )
        if self._cache:
            await self._cache.set(url, page)
        return page

    async def _get_following_safe_redirects(self, url: str) -> httpx.Response | None:
        current = url
        for _ in range(MAX_REDIRECTS + 1):
            if not await self._url_guard(current):
                logger.warning("blocked unsafe url %s", current)
                return None
            if not await self._robots_allows(current):
                logger.info("robots.txt disallows %s", current)
                return None
            try:
                response = await self._client.get(
                    current,
                    headers={"User-Agent": self._user_agent},
                    follow_redirects=False,
                    timeout=self._timeout,
                )
            except httpx.HTTPError as exc:
                logger.info("fetch failed %s: %s", current, exc)
                return None

            if response.status_code in REDIRECT_CODES and "location" in response.headers:
                current = urljoin(current, response.headers["location"])
                continue
            if response.status_code != 200:
                return None
            return response
        return None

    async def _robots_allows(self, url: str) -> bool:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin not in self._robots:
            self._robots[origin] = await self._load_robots(origin)
        parser = self._robots[origin]
        return parser is not None and parser.can_fetch(self._user_agent, url)

    async def _load_robots(self, origin: str) -> RobotFileParser | None:
        """Per RFC 9309: 4xx means allow all, 5xx or unreachable means disallow all."""
        parser = RobotFileParser()
        try:
            response = await self._client.get(
                f"{origin}/robots.txt",
                headers={"User-Agent": self._user_agent},
                timeout=self._timeout,
            )
        except httpx.HTTPError:
            return None
        if response.status_code >= 500:
            return None
        parser.parse(response.text.splitlines() if response.status_code == 200 else [])
        return parser
