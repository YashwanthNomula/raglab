from raglab.chunking import Chunk
from raglab.retrievers import BM25Retriever
from raglab.rerank import FeatureReranker


def make_chunks():
    return [
        # lexically similar to the query but off-topic (the trap)
        Chunk("d1", "d1#c0",
              "Mars chocolate bars are delicious candy with caramel and nougat",
              "Mars candy"),
        # the true match: fewer shared words, but title + phrase match
        Chunk("d2", "d2#c0",
              "The fourth planet from the sun is called the Red Planet",
              "Mars planet"),
    ]


def test_reranker_prefers_title_and_phrase_match():
    chunks = make_chunks()
    bm25 = BM25Retriever(chunks)
    cands = bm25.search("which planet is the Red Planet", k=2)
    # BM25 alone may prefer the candy doc (more 'mars' mentions)
    reranked = FeatureReranker(chunks).rerank("which planet is the Red Planet", cands, k=2)
    assert reranked[0][0] == "d2#c0"


def test_reranker_respects_k():
    chunks = make_chunks()
    r = FeatureReranker(chunks)
    out = r.rerank("mars", [("d1#c0", 1.0), ("d2#c0", 0.5)], k=1)
    assert len(out) == 1


def test_reranker_scores_bounded():
    chunks = make_chunks()
    r = FeatureReranker(chunks)
    for _, score in r.rerank("red planet mars", [("d1#c0", 1.0), ("d2#c0", 0.5)], k=2):
        assert 0.0 <= score <= 1.0
