import json
import os

import pytest

from raglab.harness import EvalHarness, load_jsonl, default_configs
from raglab.pipeline import RagPipeline, Document

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def load_data():
    docs = load_jsonl(os.path.join(DATA, "documents.jsonl"))
    questions = load_jsonl(os.path.join(DATA, "questions.jsonl"))
    return docs, questions


def test_bundled_data_loads():
    docs, questions = load_data()
    assert len(docs) == 38
    assert len(questions) == 33
    for q in questions:
        assert q["relevant_ids"]
        for rid in q["relevant_ids"]:
            assert rid in {d["id"] for d in docs}


def test_eval_runs_all_configs():
    docs, questions = load_data()
    harness = EvalHarness(docs, questions)
    results = harness.run(k=5)
    assert set(results) == {c.name for c in default_configs()}
    for res in results.values():
        for metric, value in res.items():
            assert 0.0 <= value <= 1.0, (metric, value)


def test_ablation_ladder_is_monotonic():
    # each rung of the ladder should do no worse than the last
    docs, questions = load_data()
    harness = EvalHarness(docs, questions)
    results = harness.run(k=5)
    ladder = ["tfidf", "bm25", "bm25 + phrases", "bm25 + phrases + rerank"]
    mrrs = [results[name]["mrr"] for name in ladder]
    assert mrrs == sorted(mrrs), dict(zip(ladder, mrrs))


def test_report_contains_table():
    docs, questions = load_data()
    harness = EvalHarness(docs, questions)
    results = harness.run(k=5)
    report = harness.markdown_report(results, k=5)
    assert "| config |" in report
    assert "bm25 + phrases + rerank" in report
    assert "Lift over tfidf" in report


def test_pipeline_ask_returns_contexts():
    docs, _ = load_data()
    pipe = RagPipeline(docs, config="bm25+phrases+rerank")
    ans = pipe.ask("Which planet is the Red Planet?", k=2)
    assert len(ans.contexts) == 2
    top_id = ans.contexts[0][0]
    assert top_id.startswith("d03"), f"expected Mars planet doc, got {top_id}"


def test_pipeline_all_configs_answer():
    docs, _ = load_data()
    for config in ["bm25", "tfidf", "hybrid", "bm25+phrases+rerank"]:
        pipe = RagPipeline(docs, config=config)
        ans = pipe.ask("What is photosynthesis?", k=1)
        assert ans.contexts, config


def test_pipeline_rejects_unknown_config():
    docs, _ = load_data()
    with pytest.raises(ValueError):
        RagPipeline(docs, config="nope")


def test_pipeline_save_and_load(tmp_path):
    docs, _ = load_data()
    pipe = RagPipeline(docs[:3])
    out = str(tmp_path / "index.json")
    pipe.save_index(out)
    loaded = RagPipeline.load_documents(out)
    assert len(loaded) == len(pipe.chunks)
    assert all(isinstance(d, Document) for d in loaded)


def test_eval_is_deterministic():
    docs, questions = load_data()
    h1 = EvalHarness(docs, questions).run(k=5)
    h2 = EvalHarness(docs, questions).run(k=5)
    assert h1 == h2
