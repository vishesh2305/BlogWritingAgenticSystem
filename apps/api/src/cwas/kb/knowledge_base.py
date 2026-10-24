"""Knowledge base facade: chunk -> embed -> store, and query-time retrieval."""

import hashlib
import logging

from cwas.kb.chunker import chunk_text
from cwas.kb.store import VectorStore
from cwas.models import Chunk, ScoredChunk, ScrapedPage, SourceType
from cwas.tools.embed import Embedder, EmbeddingError

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """When the embedder fails during ingest (e.g. a hosted API's rate limit), the collection
    is embedded with `fallback` instead, and its queries keep using that same embedder."""

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        max_chars: int = 1200,
        overlap: int = 150,
        fallback: Embedder | None = None,
    ) -> None:
        self._embedder = embedder
        self._fallback = fallback
        self._on_fallback: set[str] = set()
        self._store = store
        self._max_chars = max_chars
        self._overlap = overlap

    async def ingest(self, collection: str, pages: list[ScrapedPage]) -> int:
        chunks = [
            Chunk(
                id=hashlib.sha1(f"{page.url}#{i}".encode()).hexdigest(),
                text=text,
                source_url=page.url,
                source_title=page.title,
                source_type=page.source_type,
                index=i,
            )
            for page in pages
            for i, text in enumerate(chunk_text(page.text, self._max_chars, self._overlap))
        ]
        if not chunks:
            return 0
        texts = [c.text for c in chunks]
        try:
            vectors = await self._embedder.embed(texts, "document")
        except EmbeddingError as exc:
            if self._fallback is None:
                raise
            logger.warning(
                "embedding failed for %s, using the fallback embedder: %s", collection, exc
            )
            vectors = await self._fallback.embed(texts, "document")
            self._on_fallback.add(collection)
        return await self._store.add(collection, chunks, vectors)

    def uses_fallback(self, collection: str) -> bool:
        return collection in self._on_fallback

    async def retrieve(
        self, collection: str, query: str, k: int = 8, source_type: SourceType | None = None
    ) -> list[ScoredChunk]:
        embedder = self._fallback if self.uses_fallback(collection) else self._embedder
        [vector] = await embedder.embed([query], "query")  # type: ignore[union-attr]
        return await self._store.search(collection, vector, k, source_type)

    async def drop(self, collection: str) -> None:
        self._on_fallback.discard(collection)
        await self._store.drop(collection)
