"""Evaluation harness: run retrieval configs head-to-head on labeled Q&A data."""

import json
from dataclasses import dataclass, field

from . import metrics
from .chunking import chunk_documents
from .expansion import ExpandedRetriever
from .judge import HeuristicJudge, evaluate_judge_metrics
from .retrievers import BM25Retriever, TfidfRetriever, HybridRetriever
from .rerank import FeatureReranker


@dataclass
class EvalConfig:
    name: str
    build: callable  # (chunks) -> object with .search(query, k)
    rerank: bool = False


def parse_alpha_list(text):
    """Parse a comma-separated alpha grid like "0,0.25,0.5,0.75,1".

    Fail closed: raises ValueError on empty input, non-numeric entries,
    or values outside [0, 1].
    """
    if not text or not text.strip():
        raise ValueError("alpha list is empty; expected e.g. '0,0.25,0.5,0.75,1'")
    alphas = []
    for part in text.split(","):
        part = part.strip()
        try:
            a = float(part)
        except ValueError:
            raise ValueError(f"alpha {part!r} is not a number") from None
        if not 0.0 <= a <= 1.0:
            raise ValueError(f"alpha {a} is outside [0, 1]")
        alphas.append(a)
    if not alphas:
        raise ValueError("alpha list is empty; expected e.g. '0,0.25,0.5,0.75,1'")
    return alphas


def best_alpha(sweep_results):
    """Pick the alpha with the highest MRR; ties break toward the smaller
    alpha. Returns (alpha, metrics). Diagnostic only — the sweep is tuned
    and measured on the same questions."""
    if not sweep_results:
        raise ValueError("no sweep results to choose from")
    return max(sweep_results.items(), key=lambda kv: (kv[1]["mrr"], -kv[0]))


def default_configs():
    """The ablation ladder: each rung adds one idea, so the report shows
    exactly what each technique contributes."""
    return [
        EvalConfig("tfidf", lambda chunks: TfidfRetriever(chunks)),
        EvalConfig("bm25", lambda chunks: BM25Retriever(chunks)),
        EvalConfig(
            "bm25 + phrases",
            lambda chunks: BM25Retriever(chunks, use_bigrams=True),
        ),
        EvalConfig(
            "bm25 + phrases + rm3",
            lambda chunks: ExpandedRetriever(
                BM25Retriever(chunks, use_bigrams=True)
            ),
        ),
        EvalConfig(
            "bm25 + phrases + rerank",
            lambda chunks: BM25Retriever(chunks, use_bigrams=True),
            rerank=True,
        ),
    ]


class EvalHarness:
    def __init__(self, documents, questions, configs=None, chunk_size=400, overlap=80):
        self.documents = documents
        self.questions = questions
        self.configs = configs or default_configs()
        self.chunks = chunk_documents(documents, chunk_size, overlap)
        # map chunk -> doc for doc-level scoring
        self.relevant = {q["id"]: set(q["relevant_ids"]) for q in questions}

    def run(self, k=5):
        results = {}
        for cfg in self.configs:
            retriever = cfg.build(self.chunks)
            reranker = FeatureReranker(self.chunks) if cfg.rerank else None
            ranked_per_query = {}
            for q in self.questions:
                cands = retriever.search(q["question"], k=max(k, 20) if reranker else k)
                if reranker:
                    cands = reranker.rerank(q["question"], cands, k=k)
                ranked_per_query[q["id"]] = metrics.to_doc_ids([cid for cid, _ in cands])
            results[cfg.name] = metrics.evaluate_run(ranked_per_query, self.relevant, k=k)
        return results

    def run_alpha_sweep(self, alphas, k=5):
        """Sweep the hybrid RRF member weight over the benchmark.

        alpha weights the BM25 member, 1 - alpha the TF-IDF member
        (weighted RRF, still rank-based — no score calibration needed).
        Returns {alpha: metrics} in the order given. Diagnostic: the same
        questions tune and measure the weights, so the winner is a
        description of this benchmark, not a generalization claim.
        """
        results = {}
        for a in alphas:
            hybrid = HybridRetriever(
                [BM25Retriever(self.chunks), TfidfRetriever(self.chunks)],
                weights=[a, 1.0 - a],
            )
            ranked_per_query = {}
            for q in self.questions:
                cands = hybrid.search(q["question"], k=k)
                ranked_per_query[q["id"]] = metrics.to_doc_ids(
                    [cid for cid, _ in cands]
                )
            results[a] = metrics.evaluate_run(ranked_per_query, self.relevant, k=k)
        return results

    def run_judge_metrics(self, judge=None, k=5):
        """Score reference answers with LLM-as-judge metrics, per config.

        For each retrieval config, the top-k retrieved chunks become the
        context and each question's ``reference_answer`` becomes the
        answer. Faithfulness then measures whether the retrieved context
        supports the gold answer — a retrieval-quality signal that needs
        no LLM and no generated answers.
        """
        judge = judge or HeuristicJudge()
        chunk_map = {c.chunk_id: c.text for c in self.chunks}
        results = {}
        for cfg in self.configs:
            retriever = cfg.build(self.chunks)
            reranker = FeatureReranker(self.chunks) if cfg.rerank else None
            items = []
            for q in self.questions:
                cands = retriever.search(q["question"], k=max(k, 20) if reranker else k)
                if reranker:
                    cands = reranker.rerank(q["question"], cands, k=k)
                contexts = [chunk_map[cid] for cid, _ in cands[:k]]
                items.append({
                    "question": q["question"],
                    "answer": q.get("reference_answer", ""),
                    "contexts": contexts,
                })
            results[cfg.name] = evaluate_judge_metrics(items, judge=judge)
        return results

    def per_query(self, config_name, k=5):
        """Per-question retrieval for one config: each entry carries the
        question, its recall@k, its reciprocal rank (0 when nothing
        relevant is retrieved), and the retrieved / relevant doc ids.
        Raises KeyError on an unknown config name."""
        cfg = next((c for c in self.configs if c.name == config_name), None)
        if cfg is None:
            raise KeyError(f"unknown config {config_name!r}")
        retriever = cfg.build(self.chunks)
        reranker = FeatureReranker(self.chunks) if cfg.rerank else None
        rows = []
        for q in self.questions:
            cands = retriever.search(q["question"], k=max(k, 20) if reranker else k)
            if reranker:
                cands = reranker.rerank(q["question"], cands, k=k)
            ranked = metrics.to_doc_ids([cid for cid, _ in cands])
            relevant = self.relevant[q["id"]]
            rows.append({
                "id": q["id"],
                "question": q["question"],
                f"recall@{k}": metrics.recall_at_k(ranked, relevant, k),
                "reciprocal_rank": metrics.mrr(ranked, relevant),
                "retrieved_ids": ranked[:k],
                "relevant_ids": sorted(relevant),
            })
        return rows

    def hardest_questions(self, results, k=5, n=10):
        """Questions the best-MRR config retrieves nothing relevant for.

        Returns (best_config_name, zero_recall_count, rows[:n]) where rows
        are per-query entries with recall@k == 0, in question order.
        A zero-recall question is a diagnostic, not a verdict: it flags
        where to look next — a retrieval gap, a chunking gap, or a
        mislabeled gold doc — not which of those it is."""
        if not results:
            raise ValueError("no eval results to pick a best config from")
        best_name = max(results, key=lambda name: results[name]["mrr"])
        rows = self.per_query(best_name, k=k)
        zero = [r for r in rows if r[f"recall@{k}"] == 0.0]
        return best_name, len(zero), zero[:max(0, n)]

    def markdown_report(self, results, k=5, judge_results=None, judge_label="",
                        sweep_results=None, hardest=None):
        metric_names = [f"recall@{k}", f"precision@{k}", "mrr", f"ndcg@{k}", "map"]
        lines = ["# raglab evaluation report", ""]
        lines.append(f"Questions: {len(self.questions)} | Documents: {len(self.documents)} | Chunks: {len(self.chunks)} | k={k}")
        lines.append("")
        header = "| config | " + " | ".join(metric_names) + " |"
        lines.append(header)
        lines.append("|" + "---|" * (len(metric_names) + 1))
        for name, res in results.items():
            row = "| " + name + " | " + " | ".join(f"{res[m]:.3f}" for m in metric_names) + " |"
            lines.append(row)
        lines.append("")
        # deltas vs the first (weakest) config in the ladder
        baseline_name = self.configs[0].name
        if baseline_name in results:
            lines.append(f"## Lift over {baseline_name} baseline (MRR)")
            base = results[baseline_name]["mrr"] or 1e-9
            for name, res in results.items():
                if name == baseline_name:
                    continue
                lift = (res["mrr"] - results[baseline_name]["mrr"]) / base * 100
                lines.append(f"- **{name}**: {lift:+.1f}% MRR")
        if judge_results is not None:
            lines.append("")
            lines.append("## LLM-as-judge: reference-answer metrics"
                         f" ({judge_label or 'judge'})")
            lines.append("")
            lines.append("| config | faithfulness | answer_relevancy |")
            lines.append("|---|---|---|")
            for name, res in judge_results.items():
                lines.append(
                    f"| {name} | {res['faithfulness']:.3f} |"
                    f" {res['answer_relevancy']:.3f} |"
                )
        if sweep_results is not None:
            lines.append("")
            lines.append("## Hybrid RRF alpha sweep (diagnostic)")
            lines.append("")
            lines.append(
                "alpha = BM25 member weight in weighted RRF "
                "(1 - alpha = TF-IDF weight). Same questions tune and measure "
                "the weights, so the winner describes this benchmark — not a "
                "generalization claim."
            )
            lines.append("")
            lines.append(f"| alpha | recall@{k} | mrr | ndcg@{k} | map |")
            lines.append("|---|---|---|---|---|")
            for a, res in sweep_results.items():
                lines.append(
                    f"| {a:g} | {res[f'recall@{k}']:.3f} | {res['mrr']:.3f} |"
                    f" {res[f'ndcg@{k}']:.3f} | {res['map']:.3f} |"
                )
            best_a, best_res = best_alpha(sweep_results)
            lines.append("")
            lines.append(
                f"**Best alpha on this benchmark: {best_a:g}** "
                f"(mrr={best_res['mrr']:.3f}, ndcg@{k}={best_res[f'ndcg@{k}']:.3f})"
            )
        if hardest is not None:
            best_name, zero_count, rows = hardest
            lines.append("")
            lines.append("## Hardest questions (diagnostic)")
            lines.append("")
            lines.append(
                f"Best-MRR config {best_name!r} retrieves nothing relevant "
                f"for {zero_count} of {len(self.questions)} questions "
                f"(recall@{k} = 0). These are where to look next: a "
                "retrieval gap, a chunking gap, or a mislabeled gold doc."
            )
            if rows:
                lines.append("")
                for r in rows:
                    lines.append(f"- {r['id']}: {r['question']}")
            else:
                lines.append("")
                lines.append(f"- none: every question retrieves at least one relevant doc at k={k}.")
        return "\n".join(lines)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
