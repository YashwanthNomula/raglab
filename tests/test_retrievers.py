import pytest

from raglab.chunking import Chunk
from raglab.retrievers import BM25Retriever, TfidfRetriever, HybridRetriever


def make_chunks():
    return [
        Chunk("d1", "d1#c0", "the cat sat on the mat", "cats"),
        Chunk("d2", "d2#c0", "dogs bark loudly at night dogs dogs", "dogs"),
        Chunk("d3", "d3#c0", "the cat and the dog played together", "pets"),
    ]


def test_bm25_term_frequency_matters():
    chunks = [
        Chunk("d1", "d1#c0", "cat sat", ""),
        Chunk("d2", "d2#c0", "cat cat sat", ""),
    ]
    r = BM25Retriever(chunks)
    res = r.search("cat", k=2)
    # same idf, same query: higher term frequency must win
    assert res[0][0] == "d2#c0"
    assert res[0][1] > res[1][1]


def test_bm25_unknown_term_returns_nothing():
    r = BM25Retriever(make_chunks())
    assert r.search("xylophone", k=3) == []


def test_bm25_idf_downweights_common_terms():
    # 'the' was stopworded out; use a term in every doc vs one in a single doc
    chunks = [
        Chunk("d1", "d1#c0", "alpha beta gamma", ""),
        Chunk("d2", "d2#c0", "alpha delta epsilon", ""),
        Chunk("d3", "d3#c0", "alpha zeta eta", ""),
    ]
    r = BM25Retriever(chunks)
    # 'alpha' is in every doc -> low idf; 'gamma' unique -> high idf
    assert r.idf["gamma"] > r.idf["alpha"]


def test_tfidf_identical_docs_cosine_one():
    chunks = [
        Chunk("d1", "d1#c0", "machine learning models", ""),
        Chunk("d2", "d2#c0", "machine learning models", ""),
        Chunk("d3", "d3#c0", "cooking pasta recipes", ""),
    ]
    r = TfidfRetriever(chunks)
    res = r.search("machine learning models", k=3)
    scores = dict(res)
    # query identical to doc text -> cosine similarity exactly 1.0
    assert abs(scores["d1#c0"] - 1.0) < 1e-9
    assert abs(scores["d1#c0"] - scores["d2#c0"]) < 1e-9
    assert scores["d1#c0"] > scores["d3#c0"]


def test_tfidf_partial_match_scores_lower():
    chunks = [
        Chunk("d1", "d1#c0", "machine learning models", ""),
        Chunk("d2", "d2#c0", "cooking recipes pasta", ""),
    ]
    r = TfidfRetriever(chunks)
    res = r.search("machine learning", k=2)
    assert res[0][0] == "d1#c0"


def test_hybrid_fuses_ranks():
    # retriever A ranks d1 first, retriever B ranks d2 first -> RRF should surface both
    class FakeA:
        def search(self, q, k=5):
            return [("d1#c0", 9.0), ("d2#c0", 1.0)]

    class FakeB:
        def search(self, q, k=5):
            return [("d2#c0", 0.9), ("d1#c0", 0.1)]

    h = HybridRetriever([FakeA(), FakeB()])
    res = dict(h.search("q", k=2))
    # both rank-1 once and rank-2 once -> tied
    assert abs(res["d1#c0"] - res["d2#c0"]) < 1e-9


def test_hybrid_consensus_wins():
    class FakeA:
        def search(self, q, k=5):
            return [("d1#c0", 9.0), ("d3#c0", 1.0)]

    class FakeB:
        def search(self, q, k=5):
            return [("d2#c0", 0.9), ("d1#c0", 0.8)]

    h = HybridRetriever([FakeA(), FakeB()])
    res = h.search("q", k=3)
    # d1 appears in both lists -> highest fused score
    assert res[0][0] == "d1#c0"


def test_retriever_interface_k_respected():
    r = BM25Retriever(make_chunks())
    assert len(r.search("cat dog", k=1)) == 1
    assert len(r.search("cat dog", k=2)) == 2
