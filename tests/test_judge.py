"""Tests for raglab.judge: LLM-as-judge answer metrics."""

import pytest

from raglab.judge import (
    HeuristicJudge,
    _coerce_score,
    answer_relevancy,
    evaluate_judge_metrics,
    faithfulness,
    split_claims,
)

QUESTION = "How do you reverse a list in Python?"
CONTEXTS = [
    "Use the reverse() method to reverse a list in place.",
    "Slicing [::-1] returns a reversed copy of the list.",
]
FAITHFUL_ANSWER = (
    "Use the reverse() method to reverse a list in place. "
    "Slicing [::-1] returns a reversed copy."
)
HALLUCINATED_ANSWER = (
    "The sort() method reverses lists in descending order "
    "using quicksort internally."
)
IRRELEVANT_ANSWER = "Ocean tides are caused by the Moon's gravitational pull."


# --- claim splitting --------------------------------------------------------


def test_split_claims_splits_sentences():
    claims = split_claims("First claim. Second claim! Third claim?")
    assert claims == ["First claim.", "Second claim!", "Third claim?"]


def test_split_claims_empty_answer():
    assert split_claims("") == []
    assert split_claims("   ") == []


# --- score coercion ---------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [(1.0, 1.0), (0.0, 0.0), (True, 1.0), (False, 0.0),
     ("yes", 1.0), ("No", 0.0), ("true", 1.0), ("0.75", 0.75),
     (2.0, 1.0), (-1.0, 0.0), ("nonsense", 0.0)],
)
def test_coerce_score(value, expected):
    assert _coerce_score(value) == expected


# --- faithfulness ------------------------------------------------------------


def test_faithfulness_faithful_answer_high():
    score = faithfulness(QUESTION, FAITHFUL_ANSWER, CONTEXTS,
                         judge=HeuristicJudge())
    assert score >= 0.9


def test_faithfulness_hallucinated_answer_low():
    score = faithfulness(QUESTION, HALLUCINATED_ANSWER, CONTEXTS,
                         judge=HeuristicJudge())
    assert score <= 0.1


def test_faithfulness_hallucinated_scores_below_faithful():
    j = HeuristicJudge()
    bad = faithfulness(QUESTION, HALLUCINATED_ANSWER, CONTEXTS, judge=j)
    good = faithfulness(QUESTION, FAITHFUL_ANSWER, CONTEXTS, judge=j)
    assert bad < good


def test_faithfulness_empty_answer_is_vacuous():
    assert faithfulness(QUESTION, "", CONTEXTS, judge=HeuristicJudge()) == 1.0


def test_faithfulness_empty_contexts_is_zero():
    assert faithfulness(QUESTION, FAITHFUL_ANSWER, [], judge=HeuristicJudge()) == 0.0


# --- answer relevancy --------------------------------------------------------


def test_relevancy_relevant_answer_high():
    score = answer_relevancy(QUESTION, FAITHFUL_ANSWER, judge=HeuristicJudge())
    assert score >= 0.5


def test_relevancy_irrelevant_answer_low():
    score = answer_relevancy(QUESTION, IRRELEVANT_ANSWER, judge=HeuristicJudge())
    assert score < 0.5


def test_relevancy_empty_answer_is_zero():
    assert answer_relevancy(QUESTION, "", judge=HeuristicJudge()) == 0.0


# --- pluggable judge ---------------------------------------------------------


class RecordingJudge:
    """Custom judge that honors a per-claim verdict map and records calls."""

    def __init__(self, verdicts):
        self.verdicts = verdicts  # claim -> score
        self.calls = []

    def __call__(self, task, **kwargs):
        self.calls.append(task)
        if task == "claim_supported":
            return self.verdicts.get(kwargs["claim"], 0.0)
        return 0.5


def test_custom_judge_is_honored():
    claims = split_claims(FAITHFUL_ANSWER)
    judge = RecordingJudge({claims[0]: 1.0, claims[1]: 0.0})
    score = faithfulness(QUESTION, FAITHFUL_ANSWER, CONTEXTS, judge=judge)
    assert score == 0.5
    assert judge.calls == ["claim_supported", "claim_supported"]


def test_custom_judge_receives_prompt_fields():
    seen = {}

    def judge(task, **kwargs):
        seen.update(kwargs)
        return "yes"

    faithfulness(QUESTION, "One claim.", CONTEXTS, judge=judge)
    assert seen["question"] == QUESTION
    assert seen["claim"] == "One claim."
    assert seen["contexts"] == CONTEXTS


def test_judge_verdict_strings_coerced():
    def judge(task, **kwargs):
        return "no" if task == "answer_relevant" else "yes"

    assert faithfulness(QUESTION, "One claim.", CONTEXTS, judge=judge) == 1.0
    assert answer_relevancy(QUESTION, "One claim.", judge=judge) == 0.0


def test_heuristic_judge_rejects_unknown_task():
    with pytest.raises(ValueError):
        HeuristicJudge()("nonsense_task")


# --- aggregation --------------------------------------------------------------


def test_evaluate_judge_metrics_averages():
    j = HeuristicJudge()
    items = [
        {"question": QUESTION, "answer": FAITHFUL_ANSWER, "contexts": CONTEXTS},
        {"question": QUESTION, "answer": HALLUCINATED_ANSWER, "contexts": CONTEXTS},
    ]
    res = evaluate_judge_metrics(items, judge=j)
    assert res["faithfulness"] == pytest.approx(
        (faithfulness(QUESTION, FAITHFUL_ANSWER, CONTEXTS, j)
         + faithfulness(QUESTION, HALLUCINATED_ANSWER, CONTEXTS, j)) / 2
    )
    assert 0.0 <= res["faithfulness"] <= 1.0
    assert 0.0 <= res["answer_relevancy"] <= 1.0


def test_evaluate_judge_metrics_accepts_tuples():
    res = evaluate_judge_metrics(
        [(QUESTION, FAITHFUL_ANSWER, CONTEXTS)], judge=HeuristicJudge()
    )
    assert res["faithfulness"] >= 0.9


def test_evaluate_judge_metrics_empty():
    assert evaluate_judge_metrics([], judge=HeuristicJudge()) == {
        "faithfulness": 0.0, "answer_relevancy": 0.0
    }


# --- harness wiring -------------------------------------------------------------


def test_harness_run_judge_metrics_smoke():
    from raglab.harness import EvalHarness, load_jsonl

    docs = load_jsonl("data/documents.jsonl")
    questions = load_jsonl("data/questions.jsonl")[:5]
    harness = EvalHarness(docs, questions)
    results = harness.run_judge_metrics(judge=HeuristicJudge(), k=3)
    assert set(results) == {c.name for c in harness.configs}
    for res in results.values():
        assert 0.0 <= res["faithfulness"] <= 1.0
        assert 0.0 <= res["answer_relevancy"] <= 1.0
    # reference answers are written from the ground-truth docs, so
    # faithfulness of the gold answer vs retrieved context should be solid
    # for the best retrieval config
    best = results["bm25 + phrases + rerank"]
    assert best["faithfulness"] >= 0.5


def test_markdown_report_includes_judge_section():
    from raglab.harness import EvalHarness, load_jsonl

    docs = load_jsonl("data/documents.jsonl")
    questions = load_jsonl("data/questions.jsonl")[:3]
    harness = EvalHarness(docs, questions)
    results = harness.run(k=3)
    judge_results = harness.run_judge_metrics(judge=HeuristicJudge(), k=3)
    report = harness.markdown_report(results, k=3, judge_results=judge_results,
                                     judge_label="heuristic baseline")
    assert "## LLM-as-judge" in report
    assert "faithfulness" in report
    assert "heuristic baseline" in report
    for name in results:
        assert name in report
