import json
import os

import pytest

from raglab.chunking import Chunk
from raglab.cli import main
from raglab.harness import EvalHarness, best_alpha, load_jsonl, parse_alpha_list
from raglab.retrievers import BM25Retriever, HybridRetriever, TfidfRetriever

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


class FakeA:
    def search(self, q, k=5):
        return [("d1#c0", 9.0), ("d2#c0", 1.0), ("d3#c0", 0.5)]


class FakeB:
    def search(self, q, k=5):
        return [("d2#c0", 0.9), ("d1#c0", 0.1), ("d3#c0", 0.05)]


def test_weighted_rrf_alpha_one_reproduces_member_a():
    h = HybridRetriever([FakeA(), FakeB()], weights=[1.0, 0.0])
    assert [cid for cid, _ in h.search("q", k=3)] == ["d1#c0", "d2#c0", "d3#c0"]


def test_weighted_rrf_alpha_zero_reproduces_member_b():
    h = HybridRetriever([FakeA(), FakeB()], weights=[0.0, 1.0])
    assert [cid for cid, _ in h.search("q", k=3)] == ["d2#c0", "d1#c0", "d3#c0"]


def test_weighted_rrf_default_matches_equal_weights():
    plain = dict(HybridRetriever([FakeA(), FakeB()]).search("q", k=3))
    weighted = dict(
        HybridRetriever([FakeA(), FakeB()], weights=[1.0, 1.0]).search("q", k=3)
    )
    assert plain.keys() == weighted.keys()
    for cid in plain:
        assert plain[cid] == pytest.approx(weighted[cid])


def test_weighted_rrf_rejects_length_mismatch():
    with pytest.raises(ValueError, match="one weight per retriever"):
        HybridRetriever([FakeA(), FakeB()], weights=[1.0])


def test_weighted_rrf_rejects_negative_weights():
    with pytest.raises(ValueError, match="non-negative"):
        HybridRetriever([FakeA(), FakeB()], weights=[0.5, -0.5])


def test_weighted_rrf_rejects_all_zero_weights():
    with pytest.raises(ValueError, match="not all be zero"):
        HybridRetriever([FakeA(), FakeB()], weights=[0.0, 0.0])


def test_parse_alpha_list_ok():
    assert parse_alpha_list("0,0.25,0.5,0.75,1") == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert parse_alpha_list(" 0.5 ") == [0.5]


@pytest.mark.parametrize("bad", ["", "   ", "bogus", "0,one", "0,1.5", "-0.1", "2"])
def test_parse_alpha_list_rejects(bad):
    with pytest.raises(ValueError):
        parse_alpha_list(bad)


def test_best_alpha_picks_highest_mrr():
    sweep = {0.0: {"mrr": 0.90}, 0.5: {"mrr": 0.95}, 1.0: {"mrr": 0.80}}
    alpha, res = best_alpha(sweep)
    assert alpha == 0.5
    assert res["mrr"] == 0.95


def test_best_alpha_breaks_ties_toward_smaller_alpha():
    sweep = {0.25: {"mrr": 0.90}, 0.75: {"mrr": 0.90}}
    alpha, _ = best_alpha(sweep)
    assert alpha == 0.25


def test_best_alpha_rejects_empty():
    with pytest.raises(ValueError):
        best_alpha({})


def load_data():
    docs = load_jsonl(os.path.join(DATA, "documents.jsonl"))
    questions = load_jsonl(os.path.join(DATA, "questions.jsonl"))
    return docs, questions


def test_alpha_sweep_endpoints_match_member_configs():
    # alpha=1 (all weight on BM25) must score exactly like the bm25 ladder rung
    docs, questions = load_data()
    harness = EvalHarness(docs, questions)
    sweep = harness.run_alpha_sweep([0.0, 1.0], k=5)
    ladder = harness.run(k=5)
    assert sweep[1.0]["mrr"] == ladder["bm25"]["mrr"]
    assert sweep[1.0]["ndcg@5"] == ladder["bm25"]["ndcg@5"]
    assert sweep[0.0]["mrr"] == ladder["tfidf"]["mrr"]


def test_alpha_sweep_returns_metric_dicts():
    docs, questions = load_data()
    harness = EvalHarness(docs, questions)
    sweep = harness.run_alpha_sweep([0.25, 0.5, 0.75], k=5)
    assert list(sweep) == [0.25, 0.5, 0.75]
    for res in sweep.values():
        assert set(res) == {"recall@5", "precision@5", "mrr", "ndcg@5", "map"}
        for v in res.values():
            assert 0.0 <= v <= 1.0


def test_sweep_section_in_markdown_report():
    docs, questions = load_data()
    harness = EvalHarness(docs, questions)
    results = harness.run(k=5)
    sweep = harness.run_alpha_sweep([0.0, 0.5, 1.0], k=5)
    report = harness.markdown_report(results, k=5, sweep_results=sweep)
    assert "alpha sweep" in report.lower()
    best_a, _ = best_alpha(sweep)
    assert f"Best alpha on this benchmark: {best_a:g}" in report
    # without a sweep the section is absent
    assert "alpha sweep" not in harness.markdown_report(results, k=5).lower()


def test_cli_rejects_bad_alpha_before_loading_data(tmp_path):
    # no data files are touched: the grid fails validation first
    rc = main(["eval", "--alpha-sweep", "bogus",
               "--docs", "/nonexistent.jsonl",
               "--questions", "/nonexistent.jsonl",
               "--report", str(tmp_path / "r.md")])
    assert rc == 2
    assert not (tmp_path / "r.md").exists()


def test_cli_sweep_writes_json_report(tmp_path):
    report_md = tmp_path / "report.md"
    report_json = tmp_path / "report.json"
    rc = main(["eval", "--alpha-sweep", "0,1", "--report", str(report_md),
               "--report-json", str(report_json)])
    assert rc == 0
    payload = json.loads(report_json.read_text())
    assert set(payload["alpha_sweep"]["results"]) == {"0.0", "1.0"}
    assert payload["alpha_sweep"]["best_alpha"] in (0.0, 1.0)
    assert "alpha sweep" in report_md.read_text().lower()
