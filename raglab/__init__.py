"""raglab: a RAG evaluation lab.

Build retrieval pipelines (BM25, TF-IDF, dense, hybrid + rerank) and measure
them honestly: recall@k, MRR, NDCG, and head-to-head ablations on real Q&A data.
"""

__version__ = "1.2.0"

from .retrievers import BM25Retriever, TfidfRetriever, DenseRetriever, HybridRetriever
from .expansion import RM3Expander, ExpandedRetriever
from .rerank import FeatureReranker
from .pipeline import RagPipeline, Document
from .judge import (
    HeuristicJudge,
    faithfulness,
    answer_relevancy,
    split_claims,
    evaluate_judge_metrics,
)

__all__ = [
    "BM25Retriever",
    "TfidfRetriever",
    "DenseRetriever",
    "HybridRetriever",
    "RM3Expander",
    "ExpandedRetriever",
    "FeatureReranker",
    "RagPipeline",
    "Document",
    "HeuristicJudge",
    "faithfulness",
    "answer_relevancy",
    "split_claims",
    "evaluate_judge_metrics",
]
