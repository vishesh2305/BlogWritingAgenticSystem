"""Domain models shared by tools, knowledge base and agents."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

SourceType = Literal["source", "competitor"]
"""`source` pages supply facts; `competitor` pages are blogs already ranking for the topic."""


class SearchResult(BaseModel):
    url: str
    title: str
    snippet: str = ""
    score: float = 0.0


class ScrapedPage(BaseModel):
    url: str
    title: str
    text: str = Field(description="Main content as Markdown, headings kept")
    source_type: SourceType = "source"
    fetched_at: datetime
    # SEO baseline of the page, compared against when optimizing our own article.
    description: str = ""
    schema_types: list[str] = Field(default_factory=list)


class Fact(BaseModel):
    """A source sentence that carries a number: the only place statistics may come from."""

    id: str
    text: str
    numbers: list[str]
    source_url: str
    source_title: str
    source_type: SourceType = "source"


class Chunk(BaseModel):
    id: str
    text: str
    source_url: str
    source_title: str
    source_type: SourceType
    index: int


class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float


class Readability(BaseModel):
    flesch_reading_ease: float
    grade_level: float
    word_count: int


class Brief(BaseModel):
    """Output of the Prompt Enhancer: the refined, structured version of the user's request."""

    refined_prompt: str = Field(
        description="The user's request rewritten as a precise writing brief"
    )
    topic: str
    audience: str
    tone: str
    content_format: str = Field(description="e.g. how-to guide, listicle, comparison, explainer")
    primary_keyword: str
    secondary_keywords: list[str] = Field(default_factory=list)
    target_word_count: int = 1500
    # Required (no default) so schema-constrained decoding cannot skip it.
    search_queries: list[str] = Field(description="3-6 web search queries that cover the topic")
    confidence: float = Field(ge=0.0, le=1.0, description="How clear the user's intent is")
    clarifying_question: str | None = Field(
        default=None, description="One question to ask when intent is ambiguous"
    )


class Citation(BaseModel):
    number: int
    url: str
    title: str


class AIScore(BaseModel):
    """0-100 estimate of how machine-written the text reads; lower is more human."""

    score: float
    signals: dict[str, float] = Field(description="Points contributed by each signal")


class FlaggedParagraph(BaseModel):
    index: int = Field(description="Position of the block in the article")
    reasons: list[str]


class RewrittenParagraph(BaseModel):
    index: int
    text: str


class Rewrite(BaseModel):
    """Output of the Humanizer: new text for some of the flagged paragraphs."""

    paragraphs: list[RewrittenParagraph]


class Draft(BaseModel):
    variant_id: str
    markdown: str
    citations: list[Citation] = Field(default_factory=list)
    readability: Readability | None = None
    ai_score: AIScore | None = None
