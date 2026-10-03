"""raglab: a RAG evaluation lab.

Build retrieval pipelines (BM25, TF-IDF, dense, hybrid + rerank) and measure
them honestly: recall@k, MRR, NDCG, and head-to-head ablations on real Q&A data.
"""

__version__ = "1.0.0"

from .retrievers import BM25Retriever, TfidfRetriever, DenseRetriever, HybridRetriever
from .rerank import FeatureReranker
from .pipeline import RagPipeline, Document

__all__ = [
    "BM25Retriever",
    "TfidfRetriever",
    "DenseRetriever",
    "HybridRetriever",
    "FeatureReranker",
    "RagPipeline",
    "Document",
]
