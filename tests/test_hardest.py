"""Tests for per-question results and hardest-questions diagnostics."""

import pytest

from raglab.harness import EvalHarness, EvalConfig
from raglab.retrievers import TfidfRetriever


def _tiny_harness():
    docs = [
        {"id": "d01", "title": "Cats", "text": "cats dogs pets animals furry companions"},
        {"id": "d02", "title": "Space", "text": "quantum physics particles atoms orbit stars"},
    ]
    questions = [
        {"id": "q01", "question": "tell me about cats and dogs",
         "relevant_ids": ["d01"], "reference_answer": "cats and dogs"},
        # Gold doc shares no vocabulary with the question, so no retriever
        # can surface it at k=1: a genuine zero-recall diagnostic case.
        {"id": "q02", "question": "tell me about cats and dogs",
         "relevant_ids": ["d02"], "reference_answer": "quantum physics"},
    ]
    configs = [EvalConfig("tfidf", lambda chunks: TfidfRetriever(chunks))]
    return EvalHarness(docs, questions, configs=configs)


def test_per_query_returns_one_row_per_question():
    harness = _tiny_harness()
    rows = harness.per_query("tfidf", k=1)
    assert [r["id"] for r in rows] == ["q01", "q02"]
    for r in rows:
        assert 0.0 <= r["recall@1"] <= 1.0
        assert 0.0 <= r["reciprocal_rank"] <= 1.0
    assert rows[0]["recall@1"] == 1.0
    assert rows[1]["recall@1"] == 0.0


def test_per_query_unknown_config_fails_closed():
    with pytest.raises(KeyError):
        _tiny_harness().per_query("nope", k=1)


def test_hardest_questions_lists_only_zero_recall():
    harness = _tiny_harness()
    results = harness.run(k=1)
    best, count, rows = harness.hardest_questions(results, k=1, n=10)
    assert best == "tfidf"
    assert count == 1
    assert [r["id"] for r in rows] == ["q02"]


def test_hardest_questions_respects_n():
    harness = _tiny_harness()
    results = harness.run(k=1)
    best, count, rows = harness.hardest_questions(results, k=1, n=0)
    assert count == 1 and rows == []


def test_markdown_report_includes_hardest_section():
    harness = _tiny_harness()
    results = harness.run(k=1)
    hardest = harness.hardest_questions(results, k=1, n=10)
    report = harness.markdown_report(results, k=1, hardest=hardest)
    assert "## Hardest questions" in report
    assert "q02" in report
    assert "retrieves nothing relevant for 1 of 2 questions" in report


def test_markdown_report_without_hardest_omits_section():
    harness = _tiny_harness()
    report = harness.markdown_report(harness.run(k=1), k=1)
    assert "Hardest questions" not in report
