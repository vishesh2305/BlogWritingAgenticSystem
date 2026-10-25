"""Dependencies injected into every agent node."""

from dataclasses import dataclass

from cwas.config import Settings
from cwas.kb.knowledge_base import KnowledgeBase
from cwas.llm import LLM
from cwas.tools.detector import AIDetector
from cwas.tools.grammar import GrammarChecker
from cwas.tools.scrape import PageFetcher
from cwas.tools.search import SearchClient


@dataclass(frozen=True)
class AgentDeps:
    llm: LLM
    search: SearchClient
    scraper: PageFetcher
    kb: KnowledgeBase
    grammar: GrammarChecker | None
    settings: Settings
    detector: AIDetector | None = None
