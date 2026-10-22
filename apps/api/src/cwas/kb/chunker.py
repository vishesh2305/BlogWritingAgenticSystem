"""Paragraph- and sentence-aware chunking with overlap."""

import re

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")


def _pieces(text: str, max_chars: int) -> list[tuple[str, str]]:
    """Split text into (piece, separator-before-piece) units no longer than max_chars."""
    units: list[tuple[str, str]] = []
    for paragraph in _PARAGRAPH_BREAK.split(text.strip()):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        first = True
        for sentence in _SENTENCE_END.split(paragraph):
            for part, start in enumerate(range(0, len(sentence), max_chars)):
                # Paragraphs join with a blank line, sentences with a space, hard splits directly.
                separator = "\n\n" if first else ("" if part else " ")
                units.append((sentence[start : start + max_chars], separator))
                first = False
    return units


def _tail(chunk: str, overlap: int) -> str:
    if overlap <= 0 or len(chunk) <= overlap:
        return ""
    tail = chunk[-overlap:]
    space = tail.find(" ")
    return tail[space + 1 :] if space != -1 else ""


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 150) -> list[str]:
    chunks: list[str] = []
    current = ""
    for piece, separator in _pieces(text, max_chars):
        if not current:
            current = piece
        elif len(current) + len(separator) + len(piece) <= max_chars:
            current += separator + piece
        else:
            chunks.append(current)
            tail = _tail(current, overlap)
            current = (
                f"{tail} {piece}" if tail and len(tail) + 1 + len(piece) <= max_chars else piece
            )
    if current:
        chunks.append(current)
    return chunks
