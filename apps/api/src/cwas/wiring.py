"""Builds the production dependency graph from Settings."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from anthropic import AsyncAnthropic
from langgraph.checkpoint.memory import InMemorySaver

from cwas.agents.deps import AgentDeps
from cwas.config import Settings
from cwas.graph.builder import build_graph
from cwas.kb.knowledge_base import KnowledgeBase
from cwas.kb.store import InMemoryVectorStore
from cwas.llm import AnthropicLLM
from cwas.runs import RunManager
from cwas.tools.embed import Embedder, HashingEmbedder, VoyageEmbedder
from cwas.tools.grammar import LanguageToolChecker
from cwas.tools.scrape import InMemoryPageCache, Scraper
from cwas.tools.search import TavilySearch

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    pass


@asynccontextmanager
async def runtime(settings: Settings) -> AsyncIterator[RunManager]:
    if settings.tavily_api_key is None:
        raise ConfigError("TAVILY_API_KEY is required for web research")

    async with httpx.AsyncClient() as http, AsyncAnthropic() as anthropic_client:
        embedder: Embedder
        if settings.voyage_api_key is not None:
            embedder = VoyageEmbedder(http, settings.voyage_api_key.get_secret_value())
        else:
            logger.warning("VOYAGE_API_KEY not set; using the offline hashing embedder")
            embedder = HashingEmbedder()

        if settings.languagetool_url is None:
            logger.warning("LANGUAGETOOL_URL not set; grammar auto-correction disabled")

        kb = KnowledgeBase(embedder, InMemoryVectorStore())
        deps = AgentDeps(
            llm=AnthropicLLM(anthropic_client, settings.models()),
            search=TavilySearch(http, settings.tavily_api_key.get_secret_value()),
            scraper=Scraper(http, cache=InMemoryPageCache()),
            kb=kb,
            grammar=(
                LanguageToolChecker(http, settings.languagetool_url)
                if settings.languagetool_url
                else None
            ),
            settings=settings,
        )
        yield RunManager(build_graph(deps, checkpointer=InMemorySaver()), cleanup=kb.drop)
