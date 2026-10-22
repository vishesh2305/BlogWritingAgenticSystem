import pytest

from cwas.api.app import create_app
from cwas.runs import RunManager
from cwas.wiring import ConfigError, runtime
from tests.fakes import make_settings


async def test_runtime_requires_a_search_api_key() -> None:
    with pytest.raises(ConfigError, match="TAVILY_API_KEY"):
        async with runtime(make_settings(tavily_api_key=None)):
            pass


async def test_runtime_builds_a_run_manager_without_network_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    settings = make_settings(tavily_api_key="tvly-test", languagetool_url="http://lt:8010")

    async with runtime(settings) as manager:
        assert isinstance(manager, RunManager)


async def test_app_lifespan_wires_a_real_run_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    app = create_app(settings=make_settings(tavily_api_key="tvly-test"))

    async with app.router.lifespan_context(app):
        assert isinstance(app.state.runs, RunManager)
