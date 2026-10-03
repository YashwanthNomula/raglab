import math

from raglab import metrics


def test_mrr_first_hit():
    assert metrics.mrr(["a", "b", "c"], {"b"}) == 0.5


def test_mrr_no_hit():
    assert metrics.mrr(["a", "b"], {"z"}) == 0.0


def test_recall_at_k():
    assert metrics.recall_at_k(["a", "b", "c"], {"a", "c", "z"}, 2) == 1 / 3


def test_precision_at_k():
    assert metrics.precision_at_k(["a", "b", "c"], {"a", "z"}, 2) == 0.5


def test_ndcg_perfect():
    assert abs(metrics.ndcg_at_k(["a", "b"], {"a", "b"}, 2) - 1.0) < 1e-9


def test_ndcg_partial():
    # relevant at rank 2 only: DCG = 1/log2(3); IDCG = 1/log2(2) + 1/log2(3)
    got = metrics.ndcg_at_k(["x", "a"], {"a", "b"}, 2)
    expected = (1 / math.log2(3)) / (1 / math.log2(2) + 1 / math.log2(3))
    assert abs(got - expected) < 1e-9


def test_average_precision():
    # hits at ranks 1 and 3: AP = (1/1 + 2/3) / 2
    got = metrics.average_precision(["a", "x", "b"], {"a", "b"})
    assert abs(got - (1 + 2 / 3) / 2) < 1e-9


def test_to_doc_ids_dedupes_chunks():
    ranked = ["d2#c1", "d1#c0", "d2#c0", "d3#c0"]
    assert metrics.to_doc_ids(ranked) == ["d2", "d1", "d3"]


def test_evaluate_run_averages():
    ranked = {"q1": ["a", "b"], "q2": ["x", "a"]}
    rel = {"q1": {"a"}, "q2": {"a"}}
    res = metrics.evaluate_run(ranked, rel, k=2)
    assert abs(res["mrr"] - (1.0 + 0.5) / 2) < 1e-9
    assert 0.0 <= res["ndcg@2"] <= 1.0
