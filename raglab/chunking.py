"""Document chunking for RAG ingestion."""

from dataclasses import dataclass


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    text: str
    title: str = ""


def chunk_text(text, chunk_size=400, overlap=80):
    """Split text into word-boundary chunks of ~chunk_size chars with overlap."""
    words = text.split()
    chunks, current, current_len = [], [], 0
    for word in words:
        if current_len + len(word) + 1 > chunk_size and current:
            chunks.append(" ".join(current))
            # overlap: keep the last ~overlap chars worth of words
            keep, keep_len = [], 0
            for w in reversed(current):
                if keep_len + len(w) + 1 > overlap:
                    break
                keep.append(w)
                keep_len += len(w) + 1
            current, current_len = list(reversed(keep)), keep_len
        current.append(word)
        current_len += len(word) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_documents(documents, chunk_size=400, overlap=80):
    """documents: iterable of dicts with id/title/text -> list[Chunk]."""
    chunks = []
    for doc in documents:
        for i, text in enumerate(chunk_text(doc["text"], chunk_size, overlap)):
            chunks.append(
                Chunk(
                    doc_id=doc["id"],
                    chunk_id=f"{doc['id']}#c{i}",
                    text=text,
                    title=doc.get("title", ""),
                )
            )
    return chunks
