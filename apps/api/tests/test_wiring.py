import logging
from contextlib import AsyncExitStack

import httpx
import pytest
import respx

from cwas.api.app import create_app
from cwas.config import Settings
from cwas.fallback_llm import FallbackLLM
from cwas.gemini_llm import GeminiLLM
from cwas.llm import LLM
from cwas.runs import RunManager
from cwas.wiring import ConfigError, _build_llm, runtime
from tests.fakes import make_settings

TAGS_URL = "http://localhost:11434/api/tags"
LOCAL_MODEL = {"models": [{"name": "qwen3-vl:8b-instruct"}, {"name": "llama3.1:latest"}]}


def test_ollama_is_the_default_provider() -> None:
    settings = make_settings()
    assert settings.llm_provider == "ollama"
    assert settings.ollama_model == "qwen3-vl:8b-instruct"


@respx.mock
async def test_ollama_runtime_needs_no_api_keys(caplog: pytest.LogCaptureFixture) -> None:
    respx.get(TAGS_URL).mock(return_value=httpx.Response(200, json=LOCAL_MODEL))
    caplog.set_level(logging.WARNING)

    async with runtime(make_settings(tavily_api_key=None, voyage_api_key=None)) as manager:
        assert isinstance(manager, RunManager)

    assert "DuckDuckGo" in caplog.text


@respx.mock
async def test_ollama_runtime_fails_fast_when_model_is_not_pulled() -> None:
    respx.get(TAGS_URL).mock(return_value=httpx.Response(200, json={"models": []}))

    with pytest.raises(ConfigError, match="ollama pull qwen3-vl:8b-instruct"):
        async with runtime(make_settings()):
            pass


@respx.mock
async def test_ollama_runtime_fails_fast_when_ollama_is_not_running() -> None:
    respx.get(TAGS_URL).mock(side_effect=httpx.ConnectError("refused"))

    with pytest.raises(ConfigError, match="Ollama is not reachable"):
        async with runtime(make_settings()):
            pass


async def test_anthropic_runtime_requires_an_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        async with runtime(make_settings(llm_provider="anthropic")):
            pass


async def test_anthropic_runtime_builds_without_network_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    settings = make_settings(
        llm_provider="anthropic", tavily_api_key="tvly-test", languagetool_url="http://lt:8010"
    )

    async with runtime(settings) as manager:
        assert isinstance(manager, RunManager)


@respx.mock
async def test_app_lifespan_wires_a_real_run_manager() -> None:
    respx.get(TAGS_URL).mock(return_value=httpx.Response(200, json=LOCAL_MODEL))
    app = create_app(settings=make_settings())

    async with app.router.lifespan_context(app):
        assert isinstance(app.state.runs, RunManager)


async def build_llm(settings: Settings) -> LLM:
    async with httpx.AsyncClient() as http, AsyncExitStack() as stack:
        return await _build_llm(settings, http, stack)


def test_gemini_and_groq_defaults_favour_fast_models() -> None:
    settings = make_settings()
    assert settings.gemini_models() == {
        "opus": "gemini-3.8-flash",
        "sonnet": "gemini-3.8-flash",
        "haiku": "gemini-3.5-flash-lite",
    }
    assert settings.groq_models() == {
        "opus": "openai/gpt-oss-120b",
        "sonnet": "openai/gpt-oss-120b",
        "haiku": "openai/gpt-oss-20b",
    }


async def test_gemini_runtime_requires_a_gemini_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="GEMINI_API_KEY"):
        await build_llm(make_settings(llm_provider="gemini", groq_api_key="gsk-test"))


async def test_gemini_with_groq_key_builds_a_fallback_chain() -> None:
    llm = await build_llm(
        make_settings(llm_provider="gemini", gemini_api_key="AIza-test", groq_api_key="gsk-test")
    )

    assert isinstance(llm, FallbackLLM)
    assert llm.provider_names == ["gemini", "groq"]


async def test_gemini_without_groq_key_runs_alone_and_warns(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    caplog.set_level(logging.WARNING)

    llm = await build_llm(make_settings(llm_provider="gemini", gemini_api_key="AIza-test"))

    assert isinstance(llm, GeminiLLM)
    assert "GROQ_API_KEY" in caplog.text


async def test_api_keys_never_reach_the_logs(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)

    await build_llm(
        make_settings(llm_provider="gemini", gemini_api_key="AIza-secret", groq_api_key="gsk-secret")
    )

    assert "AIza-secret" not in caplog.text
    assert "gsk-secret" not in caplog.text


async def test_gemini_runtime_builds_without_network_calls() -> None:
    settings = make_settings(
        llm_provider="gemini", gemini_api_key="AIza-test", groq_api_key="gsk-test"
    )

    async with runtime(settings) as manager:
        assert isinstance(manager, RunManager)
