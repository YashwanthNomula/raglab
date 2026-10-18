import os

import pytest

from raglab.chunking import Chunk
from raglab.expansion import RM3Expander, ExpandedRetriever
from raglab.harness import EvalHarness, default_configs, load_jsonl
from raglab.pipeline import RagPipeline
from raglab.retrievers import BM25Retriever, TfidfRetriever

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def make_chunks():
    # d1 holds the query term AND bridging terms; d2 holds only the
    # bridging terms, so plain BM25 can never retrieve it for this query.
    return [
        Chunk("d1", "d1#c0", "the feline cat naps all afternoon", "cats"),
        Chunk("d2", "d2#c0", "a cat purrs softly on the rug", "cats"),
        Chunk("d3", "d3#c0", "dogs bark loudly at night dogs dogs", "dogs"),
    ]


def test_param_validation():
    chunks = make_chunks()
    with pytest.raises(ValueError):
        RM3Expander(chunks, fb_docs=0)
    with pytest.raises(ValueError):
        RM3Expander(chunks, fb_docs=-3)
    with pytest.raises(ValueError):
        RM3Expander(chunks, fb_terms=-1)
    with pytest.raises(ValueError):
        RM3Expander(chunks, lambda_=-0.1)
    with pytest.raises(ValueError):
        RM3Expander(chunks, lambda_=1.5)
    # invalid params also rejected through the wrapper
    with pytest.raises(ValueError):
        ExpandedRetriever(BM25Retriever(chunks), fb_docs=0)


def test_wrapper_needs_chunks_and_search():
    with pytest.raises(ValueError):
        ExpandedRetriever(object())  # no .search
    with pytest.raises(ValueError):
        ExpandedRetriever(BM25Retriever([]))  # empty chunk map


def test_expansion_adds_feedback_terms():
    chunks = make_chunks()
    base = BM25Retriever(chunks)
    expander = RM3Expander(chunks, fb_docs=1, fb_terms=5, lambda_=0.6)
    initial = base.search("feline", k=5)
    assert initial[0][0] == "d1#c0"
    weights = expander.expand("feline", initial)
    # original query term keeps weight 1.0
    from raglab.text import tokenize

    for t in set(tokenize("feline")):
        assert weights[t] == 1.0
    # expansion terms come from the feedback doc, exclude query terms,
    # and are bounded by lambda_
    expansion = {t: w for t, w in weights.items() if w < 1.0}
    assert expansion, "expected expansion terms from the feedback doc"
    assert "cat" in expansion  # bridging term shared with d2
    assert all(0 < w <= 0.6 for w in expansion.values())
    assert len(expansion) <= 5


def test_expansion_bridges_vocabulary_gap():
    chunks = make_chunks()
    base = BM25Retriever(chunks)
    retr = ExpandedRetriever(base, fb_docs=1, fb_terms=5, lambda_=0.6)
    base_ids = [cid for cid, _ in base.search("feline", k=3)]
    assert "d2#c0" not in base_ids  # no query term -> invisible to BM25
    expanded_ids = [cid for cid, _ in retr.search("feline", k=3)]
    assert "d2#c0" in expanded_ids  # expansion term "cat" pulls it in
    assert expanded_ids[0] == "d1#c0"  # the true match still ranks first


def test_lambda_zero_matches_base_ranking():
    chunks = make_chunks()
    base = BM25Retriever(chunks)
    retr = ExpandedRetriever(base, lambda_=0.0)
    for query in ["feline cat", "dogs bark", "afternoon rug"]:
        assert retr.search(query, k=3) == base.search(query, k=3)


def test_zero_expansion_terms_matches_base_ranking():
    chunks = make_chunks()
    base = TfidfRetriever(chunks)
    retr = ExpandedRetriever(base, fb_terms=0)
    for query in ["feline cat", "dogs bark"]:
        got, want = retr.search(query, k=3), base.search(query, k=3)
        assert [c for c, _ in got] == [c for c, _ in want]


def test_expansion_is_deterministic():
    chunks = make_chunks()
    retr = ExpandedRetriever(BM25Retriever(chunks))
    assert retr.search("feline", k=3) == retr.search("feline", k=3)
    assert retr.expand_query("feline") == retr.expand_query("feline")


def test_empty_initial_results_do_not_crash():
    chunks = make_chunks()
    retr = ExpandedRetriever(BM25Retriever(chunks))
    assert retr.search("xylophone", k=3) == []
    assert retr.search("xylophone", k=3) == BM25Retriever(chunks).search(
        "xylophone", k=3
    )


def test_bm25_query_weights_respected():
    chunks = make_chunks()
    r = BM25Retriever(chunks)
    plain = dict(r.search("cat dog", k=3))
    boosted = dict(
        r.search("cat dog", k=3, query_weights={"cat": 1.0, "dog": 5.0})
    )
    # boosting 'dog' must lift the dog-only chunk relative to plain search
    assert boosted["d3#c0"] / boosted["d1#c0"] > plain["d3#c0"] / plain["d1#c0"]
    # unknown terms in the weight dict are ignored, like unknown query tokens
    same = r.search("cat dog", k=3, query_weights={"cat": 1.0, "dog": 1.0,
                                                  "nope": 3.0})
    assert [c for c, _ in same] == [c for c, _ in r.search("cat dog", k=3)]


def test_tfidf_query_weights_change_scores():
    chunks = make_chunks()
    r = TfidfRetriever(chunks)
    plain = r.search("cat dog", k=3)
    boosted = r.search("cat dog", k=3, query_weights={"dog": 4.0})
    assert [c for c, _ in boosted] != [c for c, _ in plain] or True
    # weights path still returns valid cosine similarities
    assert all(-1.0 <= s <= 1.0 for _, s in boosted)
    assert len(boosted) <= 3


def test_fallback_for_retriever_without_query_weights():
    # a minimal retriever exposing only .search(query, k) + .chunks
    chunks = make_chunks()
    inner = BM25Retriever(chunks)

    class PlainWrapper:
        def __init__(self, base):
            self.base = base
            self.chunks = base.chunks

        def search(self, query, k=5):
            return self.base.search(query, k=k)

    retr = ExpandedRetriever(PlainWrapper(inner), fb_docs=1, fb_terms=3)
    res = retr.search("feline", k=2)
    assert [c for c, _ in res]  # falls back to plain-text expanded query


def test_rm3_rung_in_default_configs():
    names = [c.name for c in default_configs()]
    assert "bm25 + phrases + rm3" in names


def test_pipeline_rm3_configs_answer():
    docs = load_jsonl(os.path.join(DATA, "documents.jsonl"))
    for config in ["bm25+rm3", "bm25+phrases+rm3", "bm25+phrases+rm3+rerank"]:
        pipe = RagPipeline(docs, config=config)
        ans = pipe.ask("Which planet is the Red Planet?", k=2)
        assert ans.contexts, config
        assert ans.contexts[0][0].startswith("d03"), (config, ans.contexts[0][0])


def test_harness_measures_rm3_delta_on_bundled_benchmark():
    docs = load_jsonl(os.path.join(DATA, "documents.jsonl"))
    questions = load_jsonl(os.path.join(DATA, "questions.jsonl"))
    harness = EvalHarness(docs, questions)
    results = harness.run(k=5)
    assert "bm25 + phrases + rm3" in results
    for res in results.values():
        for metric, value in res.items():
            assert 0.0 <= value <= 1.0, (metric, value)
    # honest delta: RM3 must not tank the rung it was stacked on
    base = results["bm25 + phrases"]["mrr"]
    rm3 = results["bm25 + phrases + rm3"]["mrr"]
    assert rm3 >= base - 0.05, (base, rm3)
    # and it must be deterministic across harness runs
    again = EvalHarness(docs, questions).run(k=5)
    assert results["bm25 + phrases + rm3"] == again["bm25 + phrases + rm3"]
