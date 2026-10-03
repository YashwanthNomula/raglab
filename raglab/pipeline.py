"""RagPipeline: chunk -> index -> retrieve -> answer (extractive)."""

from dataclasses import dataclass

from .chunking import chunk_documents
from .retrievers import BM25Retriever, TfidfRetriever, HybridRetriever
from .rerank import FeatureReranker


@dataclass
class Document:
    id: str
    title: str
    text: str


@dataclass
class Answer:
    question: str
    contexts: list  # [(chunk_id, title, text, score)]
    config: str


class RagPipeline:
    """Zero-dependency RAG pipeline with swappable retrieval strategies.

    config options: "tfidf", "bm25", "bm25+phrases", "bm25+phrases+rerank"
    """

    def __init__(self, documents, config="hybrid+rerank", chunk_size=400, overlap=80):
        dicts = [
            {"id": d.id, "title": d.title, "text": d.text}
            if isinstance(d, Document)
            else d
            for d in documents
        ]
        self.chunks = chunk_documents(dicts, chunk_size, overlap)
        self.chunk_map = {c.chunk_id: c for c in self.chunks}
        self.config = config
        if config == "tfidf":
            self.retriever = TfidfRetriever(self.chunks)
        elif config == "bm25":
            self.retriever = BM25Retriever(self.chunks)
        elif config == "bm25+phrases":
            self.retriever = BM25Retriever(self.chunks, use_bigrams=True)
        elif config == "bm25+phrases+rerank":
            self.retriever = BM25Retriever(self.chunks, use_bigrams=True)
            self.reranker = FeatureReranker(self.chunks)
        elif config in ("hybrid", "hybrid+rerank"):
            # RRF fusion of BM25 + TF-IDF; kept for dense+lexical fusion via
            # HybridRetriever([BM25Retriever(...), DenseRetriever(...)])
            self.retriever = HybridRetriever([BM25Retriever(self.chunks),
                                              TfidfRetriever(self.chunks)])
            if config == "hybrid+rerank":
                self.reranker = FeatureReranker(self.chunks)
        else:
            raise ValueError(f"unknown config: {config}")
        self.reranker = getattr(self, "reranker", None)

    def ask(self, question, k=3):
        cands = self.retriever.search(question, k=max(k, 20) if self.reranker else k)
        if self.reranker:
            cands = self.reranker.rerank(question, cands, k=k)
        contexts = [
            (cid, self.chunk_map[cid].title, self.chunk_map[cid].text, score)
            for cid, score in cands[:k]
        ]
        return Answer(question=question, contexts=contexts, config=self.config)

    def save_index(self, path):
        import json

        with open(path, "w") as f:
            json.dump(
                [
                    {"id": c.doc_id, "title": c.title, "text": c.text}
                    for c in self.chunks
                ],
                f,
            )

    @staticmethod
    def load_documents(path):
        import json

        with open(path) as f:
            data = json.load(f)
        return [Document(id=d["id"], title=d.get("title", ""), text=d["text"]) for d in data]
