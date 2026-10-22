from cwas.kb.chunker import chunk_text


def para(n: int, words: int = 40) -> str:
    return " ".join(f"p{n}w{i}" for i in range(words)) + "."


def test_empty_or_blank_text_yields_no_chunks() -> None:
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []


def test_short_text_is_a_single_chunk() -> None:
    assert chunk_text("Hello world.", max_chars=100) == ["Hello world."]


def test_chunks_never_exceed_max_chars() -> None:
    text = "\n\n".join(para(i) for i in range(30))
    chunks = chunk_text(text, max_chars=500, overlap=80)
    assert len(chunks) > 1
    assert all(len(c) <= 500 for c in chunks)


def test_every_paragraph_appears_in_some_chunk() -> None:
    paragraphs = [para(i, 20) for i in range(15)]
    chunks = chunk_text("\n\n".join(paragraphs), max_chars=400, overlap=0)
    joined = "\n\n".join(chunks)
    for p in paragraphs:
        assert p in joined


def test_consecutive_chunks_overlap() -> None:
    text = "\n\n".join(para(i, 20) for i in range(15))
    chunks = chunk_text(text, max_chars=400, overlap=60)
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        head = nxt.split()[0]
        assert head in prev.split()[-15:]


def test_long_paragraph_is_split_on_sentences() -> None:
    sentences = [f"Sentence number {i} is about solar energy." for i in range(40)]
    chunks = chunk_text(" ".join(sentences), max_chars=300, overlap=0)
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)
    assert all(c.rstrip().endswith(".") for c in chunks)


def test_unbreakable_text_is_hard_split() -> None:
    blob = "x" * 1050
    chunks = chunk_text(blob, max_chars=500, overlap=0)
    assert [len(c) for c in chunks] == [500, 500, 50]
