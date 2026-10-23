"""Typed runtime settings, loaded from the environment / .env."""

from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

Tier = Literal["opus", "sonnet", "haiku"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM backend: a local Ollama model (free), Gemini with Groq as fallback (fast, hosted),
    # or the Anthropic API (ANTHROPIC_API_KEY, read by the SDK itself).
    llm_provider: Literal["ollama", "gemini", "anthropic"] = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3-vl:8b-instruct"
    ollama_num_ctx: int = 16384
    ollama_timeout: float = 900.0

    gemini_api_key: SecretStr | None = None
    groq_api_key: SecretStr | None = None
    gemini_model_opus: str = "gemini-3.8-flash"
    gemini_model_sonnet: str = "gemini-3.8-flash"
    gemini_model_haiku: str = "gemini-3.5-flash-lite"
    groq_model_opus: str = "openai/gpt-oss-120b"
    groq_model_sonnet: str = "openai/gpt-oss-120b"
    groq_model_haiku: str = "openai/gpt-oss-20b"
    hosted_llm_timeout: float = 300.0

    # Optional paid services; free fallbacks are used when a key is missing.
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

    # Knowledge base
    max_facts_per_page: int = 15

    # Prompt enhancer
    clarification_threshold: float = 0.6
    max_clarification_rounds: int = 1

    # Writer
    retrieval_k: int = 12

    # Humanizer: rewrite until the AI-likeness score (0-100) is below the threshold
    ai_score_threshold: float = 10.0
    max_humanize_rounds: int = 2

    def models(self) -> dict[Tier, str]:
        return {"opus": self.model_opus, "sonnet": self.model_sonnet, "haiku": self.model_haiku}

    def gemini_models(self) -> dict[Tier, str]:
        return {
            "opus": self.gemini_model_opus,
            "sonnet": self.gemini_model_sonnet,
            "haiku": self.gemini_model_haiku,
        }

    def groq_models(self) -> dict[Tier, str]:
        return {
            "opus": self.groq_model_opus,
            "sonnet": self.groq_model_sonnet,
            "haiku": self.groq_model_haiku,
        }
