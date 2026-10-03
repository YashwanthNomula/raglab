import math

from raglab.text import tokenize, bigrams
from raglab.chunking import chunk_text, chunk_documents


def test_tokenize_lowercases_and_strips_stopwords():
    tokens = tokenize("The Quick Brown Fox jumps!")
    assert "the" not in tokens
    assert "quick" in tokens and "fox" in tokens


def test_tokenize_keeps_numbers():
    assert "42" in tokenize("answer 42")


def test_bigrams():
    assert bigrams(["a", "b", "c"]) == [("a", "b"), ("b", "c")]


def test_chunk_text_respects_size():
    text = " ".join(f"word{i}" for i in range(100))
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert all(len(c) <= 130 for c in chunks)


def test_chunk_text_overlap():
    text = " ".join(f"word{i}" for i in range(50))
    chunks = chunk_text(text, chunk_size=120, overlap=60)
    # overlap means some words repeat across chunk boundaries
    words0 = set(chunks[0].split())
    words1 = set(chunks[1].split())
    assert words0 & words1


def test_chunk_documents_ids():
    docs = [{"id": "d1", "title": "T", "text": "hello world " * 100}]
    chunks = chunk_documents(docs, chunk_size=100, overlap=20)
    assert all(c.doc_id == "d1" for c in chunks)
    assert chunks[0].chunk_id == "d1#c0"
    assert chunks[1].chunk_id == "d1#c1"
