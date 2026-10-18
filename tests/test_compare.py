import os

import pytest

from raglab.compare import (
    DEFAULT_COMPARE_CONFIGS,
    VALID_CONFIGS,
    compare_configs,
    format_report,
    overlap_summary,
)
from raglab.harness import load_jsonl

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def load_docs():
    return load_jsonl(os.path.join(DATA, "documents.jsonl"))


def test_compare_runs_two_configs_with_structured_results():
    docs = load_docs()
    results = compare_configs(docs, "how do I reverse a list in Python?",
                              ["bm25", "tfidf"], k=3)
    assert len(results) == 2
    assert [r["config"] for r in results] == ["bm25", "tfidf"]
    for r in results:
        assert len(r["hits"]) == 3
        for i, hit in enumerate(r["hits"], 1):
            assert hit["rank"] == i
            assert hit["doc_id"]
            assert hit["chunk_id"].startswith(hit["doc_id"])
            assert isinstance(hit["score"], float)
            assert isinstance(hit["title"], str)
            assert hit["text"]
    # ranks must be 1..k in order
    assert [h["rank"] for h in results[0]["hits"]] == [1, 2, 3]


def test_compare_supports_at_least_three_configs():
    docs = load_docs()
    results = compare_configs(docs, "what is a python dictionary?",
                              list(DEFAULT_COMPARE_CONFIGS), k=2)
    assert len(results) == 3
    assert all(len(r["hits"]) == 2 for r in results)
    for name in DEFAULT_COMPARE_CONFIGS:
        assert name in VALID_CONFIGS


def test_compare_invalid_config_errors_cleanly():
    docs = load_docs()
    with pytest.raises(ValueError) as exc:
        compare_configs(docs, "hello", ["bm25", "not-a-config"], k=3)
    assert "not-a-config" in str(exc.value)
    assert "unknown config" in str(exc.value)


def test_overlap_summary_small_fixture():
    results = [
        {"config": "bm25",
         "hits": [{"doc_id": "d01"}, {"doc_id": "d02"}, {"doc_id": "d03"}]},
        {"config": "tfidf",
         "hits": [{"doc_id": "d02"}, {"doc_id": "d04"}]},
        {"config": "hybrid",
         "hits": [{"doc_id": "d01"}, {"doc_id": "d04"}, {"doc_id": "d05"}]},
    ]
    overlap = overlap_summary(results)
    # d02 and d04 shared by exactly two; d01 by two as well; d03/d05 alone
    assert overlap == {
        "d01": ["bm25", "hybrid"],
        "d02": ["bm25", "tfidf"],
        "d04": ["hybrid", "tfidf"],
    }


def test_overlap_summary_empty_when_no_shared_docs():
    results = [
        {"config": "bm25", "hits": [{"doc_id": "d01"}]},
        {"config": "tfidf", "hits": [{"doc_id": "d02"}]},
    ]
    assert overlap_summary(results) == {}


def test_format_report_shows_all_configs_and_overlap():
    docs = load_docs()
    results = compare_configs(docs, "reverse a list", ["bm25", "tfidf"], k=2)
    report = format_report("reverse a list", results, k=2)
    assert 'Q: "reverse a list"' in report
    for name in ["bm25", "tfidf"]:
        assert f"=== {name} (top 2) ===" in report
    assert "=== overlap" in report


def test_compare_query_with_no_matching_terms_returns_empty_hits():
    docs = load_docs()
    results = compare_configs(docs, "xqzvwj plugh zzz", ["bm25"], k=3)
    assert results[0]["hits"] == []
