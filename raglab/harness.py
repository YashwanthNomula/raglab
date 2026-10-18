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

    def markdown_report(self, results, k=5, judge_results=None, judge_label=""):
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
        return "\n".join(lines)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
