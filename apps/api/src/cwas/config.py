"""Typed runtime settings, loaded from the environment / .env."""

from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

Tier = Literal["opus", "sonnet", "haiku"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Third-party services. ANTHROPIC_API_KEY is read by the Anthropic SDK itself.
    tavily_api_key: SecretStr | None = None
    voyage_api_key: SecretStr | None = None
    languagetool_url: str | None = None

    model_opus: str = "claude-opus-5-5"
    model_sonnet: str = "claude-sonnet-5-5"
    model_haiku: str = "claude-haiku-5-5"

    # Research
    max_search_queries: int = 4
    results_per_query: int = 5
    competitor_results: int = 5
    max_sources: int = 12
    scrape_concurrency: int = 6

    # Prompt enhancer
    clarification_threshold: float = 0.6
    max_clarification_rounds: int = 1

    # Writer
    retrieval_k: int = 12

    def models(self) -> dict[Tier, str]:
        return {"opus": self.model_opus, "sonnet": self.model_sonnet, "haiku": self.model_haiku}
