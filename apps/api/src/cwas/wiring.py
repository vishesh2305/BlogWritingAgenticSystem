"""Builds the production dependency graph from Settings."""

import logging
import os
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager

import httpx
from anthropic import AsyncAnthropic
from langgraph.checkpoint.memory import InMemorySaver

from cwas.agents.deps import AgentDeps
from cwas.config import Settings, Tier
from cwas.fallback_llm import FallbackLLM
from cwas.gemini_llm import GeminiLLM
from cwas.graph.builder import build_graph
from cwas.groq_llm import GroqLLM
from cwas.kb.knowledge_base import KnowledgeBase
from cwas.kb.store import InMemoryVectorStore
from cwas.llm import LLM, AnthropicLLM
from cwas.ollama_llm import OllamaLLM
from cwas.runs import RunManager
from cwas.throttled_llm import ThrottledLLM
from cwas.tools.detector import AIDetector, SaplingDetector
from cwas.tools.embed import Embedder, GeminiEmbedder, HashingEmbedder, VoyageEmbedder
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


def _throttled(llm: LLM, requests_per_minute: int, settings: Settings) -> ThrottledLLM:
    return ThrottledLLM(
        llm,
        requests_per_minute=requests_per_minute,
        max_concurrent=settings.llm_max_concurrency,
        max_wait=settings.llm_max_wait,
    )


def _build_gemini(settings: Settings, http: httpx.AsyncClient) -> LLM:
    if settings.gemini_api_key is None:
        raise ConfigError(
            "GEMINI_API_KEY is required when LLM_PROVIDER=gemini "
            "(or set LLM_PROVIDER=ollama to use a free local model)"
        )
    key = settings.gemini_api_key.get_secret_value()

    def gemini(models: dict[Tier, str]) -> ThrottledLLM:
        llm = GeminiLLM(http, api_key=key, models=models, timeout=settings.hosted_llm_timeout)
        return _throttled(llm, settings.gemini_requests_per_minute, settings)

    # Gemini quotas are per model, so a second Gemini model is the cheapest first fallback.
    providers: list[tuple[str, LLM]] = [("gemini", gemini(settings.gemini_models()))]
    if lite := settings.gemini_fallback_model:
        providers.append(("gemini-lite", gemini({"opus": lite, "sonnet": lite, "haiku": lite})))
    if settings.groq_api_key is None:
        logger.warning("GROQ_API_KEY not set; no fallback outside Gemini")
    else:
        groq = GroqLLM(
            http,
            api_key=settings.groq_api_key.get_secret_value(),
            models=settings.groq_models(),
            timeout=settings.hosted_llm_timeout,
        )
        providers.append(("groq", _throttled(groq, settings.groq_requests_per_minute, settings)))

    if len(providers) == 1:
        return providers[0][1]
    logger.info("LLM chain: %s", " -> ".join(name for name, _ in providers))
    return FallbackLLM(providers)


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

    if settings.llm_provider == "gemini":
        return _build_gemini(settings, http)

    if not any(os.environ.get(name) for name in _ANTHROPIC_CREDENTIAL_VARS):
        raise ConfigError(
            "ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic "
            "(or set LLM_PROVIDER=ollama to use a free local model)"
        )
    client = await stack.enter_async_context(AsyncAnthropic())
    return AnthropicLLM(client, settings.models())


def _build_embedder(settings: Settings, http: httpx.AsyncClient) -> Embedder:
    if settings.voyage_api_key is not None:
        return VoyageEmbedder(http, settings.voyage_api_key.get_secret_value())
    if settings.gemini_api_key is not None:
        logger.info("using Gemini embeddings (%s)", settings.gemini_embedding_model)
        return GeminiEmbedder(
            http,
            settings.gemini_api_key.get_secret_value(),
            model=settings.gemini_embedding_model,
            dimensions=settings.embedding_dimensions,
        )
    logger.warning("VOYAGE_API_KEY and GEMINI_API_KEY not set; using the offline hashing embedder")
    return HashingEmbedder()


def _build_detector(settings: Settings, http: httpx.AsyncClient) -> AIDetector | None:
    if settings.sapling_api_key is None:
        logger.info("SAPLING_API_KEY not set; the humanizer uses the local AI-likeness score")
        return None
    return SaplingDetector(
        http, settings.sapling_api_key.get_secret_value(), timeout=settings.detector_timeout
    )


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

        if settings.languagetool_url is None:
            logger.warning("LANGUAGETOOL_URL not set; grammar auto-correction disabled")

        kb = KnowledgeBase(
            _build_embedder(settings, http), InMemoryVectorStore(), fallback=HashingEmbedder()
        )
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
            detector=_build_detector(settings, http),
        )
        yield RunManager(build_graph(deps, checkpointer=InMemorySaver()), cleanup=kb.drop)
