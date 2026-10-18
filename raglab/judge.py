"""LLM-as-judge answer metrics: faithfulness + answer relevancy.

The retrieval metrics in ``metrics.py`` score *rankings*; these score
*answers*. Both metrics delegate the actual judgment to a pluggable
``Judge`` so you can swap in a real LLM judge (or your own heuristic)
without touching the metric math.

Judge protocol
--------------
A judge is any callable with this signature::

    judge(task, **kwargs) -> score

where ``task`` is one of:

- ``"claim_supported"`` — is this claim supported by the contexts?
  kwargs: ``question``, ``claim``, ``contexts`` (list of str)
- ``"answer_relevant"`` — does this answer address the question?
  kwargs: ``question``, ``answer``

``score`` may be a float in [0, 1] (a bool or int works too) or a
verdict string like ``"yes"`` / ``"no"`` / ``"true"`` / ``"false"``;
anything else is clamped to [0, 1] via ``float()`` when possible.

The repo ships a deterministic offline baseline, ``HeuristicJudge``,
used by the CLI demo, the README examples, and all tests. It scores by
token overlap — honest, reproducible, but coarse: it cannot reason about
paraphrase, negation, or implication the way a real LLM judge can. Plug
a real judge for numbers you would publish.
"""

import re

from .text import tokenize


_CLAIM_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

_YES = {"yes", "y", "true", "supported", "1", "supported_claim"}
_NO = {"no", "n", "false", "unsupported", "0"}


def _coerce_score(value):
    """Normalize a judge's return value to a float in [0, 1]."""
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _YES:
            return 1.0
        if lowered in _NO:
            return 0.0
        try:
            return max(0.0, min(1.0, float(lowered)))
        except ValueError:
            return 0.0
    return 0.0


def split_claims(answer):
    """Split an answer into claims (sentences), one judge unit each."""
    return [c.strip() for c in _CLAIM_SPLIT_RE.split(answer or "") if c.strip()]


class HeuristicJudge:
    """Deterministic offline baseline judge: token-overlap heuristics.

    - ``claim_supported``: fraction of the claim's content tokens found in
      the retrieved contexts. Supported if coverage >= 0.5.
    - ``answer_relevant``: fraction of the question's content tokens that
      also appear in the answer.

    Clearly labeled a baseline everywhere it appears: it mistakes
    paraphrase for absence, ignores negation, and says nothing about the
    quality of a real LLM judge. Useful for tests, demos, and keeping
    the repo zero-dependency — not for decisions.
    """

    def __init__(self, support_threshold=0.5):
        self.support_threshold = support_threshold

    def __call__(self, task, **kwargs):
        if task == "claim_supported":
            return self._claim_supported(
                kwargs.get("claim", ""), kwargs.get("contexts", [])
            )
        if task == "answer_relevant":
            return self._answer_relevant(
                kwargs.get("question", ""), kwargs.get("answer", "")
            )
        raise ValueError(f"unknown judge task: {task!r}")

    @staticmethod
    def _coverage(tokens, pool):
        if not tokens:
            return 0.0
        return len(set(tokens) & set(pool)) / len(set(tokens))

    def _claim_supported(self, claim, contexts):
        claim_tokens = tokenize(claim)
        if not claim_tokens:
            return 1.0  # nothing checkable in the claim
        pool = []
        for ctx in contexts:
            pool.extend(tokenize(ctx))
        coverage = self._coverage(claim_tokens, pool)
        return 1.0 if coverage >= self.support_threshold else 0.0

    def _answer_relevant(self, question, answer):
        question_tokens = tokenize(question)
        if not question_tokens or not answer.strip():
            return 0.0
        return self._coverage(question_tokens, tokenize(answer))


def faithfulness(question, answer, contexts, judge=None):
    """Share of answer claims supported by the retrieved contexts (0..1).

    ``contexts`` is a list of chunk-text strings. An answer with no
    checkable claims scores 1.0 (vacuous: nothing to contradict).
    """
    judge = judge or HeuristicJudge()
    claims = split_claims(answer)
    if not claims:
        return 1.0
    scores = [
        _coerce_score(judge("claim_supported", question=question,
                            claim=claim, contexts=contexts))
        for claim in claims
    ]
    return sum(scores) / len(scores)


def answer_relevancy(question, answer, judge=None):
    """How well the answer addresses the question (0..1)."""
    judge = judge or HeuristicJudge()
    return _coerce_score(judge("answer_relevant", question=question, answer=answer))


def evaluate_judge_metrics(items, judge=None):
    """Aggregate judge metrics over many (question, answer, contexts) items.

    ``items`` is an iterable of dicts or tuples with question/answer/contexts.
    Returns ``{"faithfulness": mean, "answer_relevancy": mean}``.
    """
    judge = judge or HeuristicJudge()
    items = list(items)
    if not items:
        return {"faithfulness": 0.0, "answer_relevancy": 0.0}
    tot_f, tot_r = 0.0, 0.0
    for item in items:
        if isinstance(item, dict):
            question, answer, contexts = (
                item["question"], item["answer"], item["contexts"])
        else:
            question, answer, contexts = item
        tot_f += faithfulness(question, answer, contexts, judge=judge)
        tot_r += answer_relevancy(question, answer, judge=judge)
    n = len(items)
    return {"faithfulness": tot_f / n, "answer_relevancy": tot_r / n}
