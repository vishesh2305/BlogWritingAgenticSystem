"""Builds the production dependency graph from Settings."""

import logging
import os
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager

import httpx
from anthropic import AsyncAnthropic
from langgraph.checkpoint.memory import InMemorySaver

from cwas.agents.deps import AgentDeps
from cwas.config import Settings
from cwas.graph.builder import build_graph
from cwas.kb.knowledge_base import KnowledgeBase
from cwas.kb.store import InMemoryVectorStore
from cwas.llm import LLM, AnthropicLLM
from cwas.ollama_llm import OllamaLLM
from cwas.runs import RunManager
from cwas.tools.embed import Embedder, HashingEmbedder, VoyageEmbedder
from cwas.tools.grammar import LanguageToolChecker
from cwas.tools.scrape import InMemoryPageCache, Scraper
from cwas.tools.search import DuckDuckGoSearch, SearchClient, TavilySearch

logger = logging.getLogger(__name__)

_ANTHROPIC_CREDENTIAL_VARS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE")


class ConfigError(Exception):
    pass


async def _check_ollama_model(http: httpx.AsyncClient, settings: Settings) -> None:
    base = settings.ollama_base_url.rstrip("/")
    try:
        response = await http.get(f"{base}/api/tags", timeout=10.0)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise ConfigError(
            f"Ollama is not reachable at {base}; start it with `ollama serve` ({exc!r})"
        ) from exc

    installed = {model["name"] for model in response.json().get("models", [])}
    wanted = settings.ollama_model
    if wanted not in installed and f"{wanted}:latest" not in installed:
        raise ConfigError(
            f"Ollama model {wanted!r} is not installed; run `ollama pull {wanted}` "
            "or set OLLAMA_MODEL to one of: " + ", ".join(sorted(installed))
        )


async def _build_llm(settings: Settings, http: httpx.AsyncClient, stack: AsyncExitStack) -> LLM:
    if settings.llm_provider == "ollama":
        await _check_ollama_model(http, settings)
        logger.info("using local Ollama model %s", settings.ollama_model)
        return OllamaLLM(
            http,
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            num_ctx=settings.ollama_num_ctx,
            timeout=settings.ollama_timeout,
        )

    if not any(os.environ.get(name) for name in _ANTHROPIC_CREDENTIAL_VARS):
        raise ConfigError(
            "ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic "
            "(or set LLM_PROVIDER=ollama to use a free local model)"
        )
    client = await stack.enter_async_context(AsyncAnthropic())
    return AnthropicLLM(client, settings.models())


@asynccontextmanager
async def runtime(settings: Settings) -> AsyncIterator[RunManager]:
    async with httpx.AsyncClient() as http, AsyncExitStack() as stack:
        llm = await _build_llm(settings, http, stack)

        search: SearchClient
        if settings.tavily_api_key is not None:
            search = TavilySearch(http, settings.tavily_api_key.get_secret_value())
        else:
            logger.warning("TAVILY_API_KEY not set; using free DuckDuckGo search")
            search = DuckDuckGoSearch()

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
            llm=llm,
            search=search,
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
