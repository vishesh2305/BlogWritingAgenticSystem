"""Shared state passed between agents in the LangGraph run."""

import operator
from typing import Annotated, TypedDict

from cwas.models import Brief, Draft, ScrapedPage


class RunState(TypedDict, total=False):
    run_id: str
    user_prompt: str
    clarifications: list[str]
    clarification_rounds: int
    brief: Brief
    sources: list[ScrapedPage]
    kb_chunks: int
    drafts: list[Draft]
    final_article: str
    warnings: Annotated[list[str], operator.add]
