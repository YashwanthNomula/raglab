"""Side-by-side comparison of retrieval configs on a single query.

Scores are comparable only *within* a config, never across configs, so the
report shows per-config ranked lists plus an overlap summary: which documents
showed up in the top-k of more than one config.
"""

from collections import defaultdict

from .pipeline import RagPipeline

# Same config names as `raglab ask --config`.
VALID_CONFIGS = [
    "tfidf",
    "bm25",
    "bm25+phrases",
    "bm25+phrases+rerank",
    "bm25+rm3",
    "bm25+phrases+rm3",
    "bm25+phrases+rm3+rerank",
    "hybrid",
    "hybrid+rerank",
]

# Dense retriever is optional (needs sentence-transformers) and not wired
# into the compare default set for that reason.
DEFAULT_COMPARE_CONFIGS = ["bm25", "tfidf", "hybrid"]


def compare_configs(documents, query, configs, k=5):
    """Run each retrieval config on the same query over the same documents.

    Returns a list of dicts, one per config:
        {"config": name, "hits": [{"rank", "chunk_id", "doc_id",
                                   "title", "text", "score"}]}
    """
    unknown = [c for c in configs if c not in VALID_CONFIGS]
    if unknown:
        raise ValueError(
            f"unknown config(s): {', '.join(unknown)}; "
            f"valid configs: {', '.join(VALID_CONFIGS)}"
        )
    results = []
    for name in configs:
        pipe = RagPipeline(documents, config=name)
        answer = pipe.ask(query, k=k)
        hits = []
        for rank, (cid, title, text, score) in enumerate(answer.contexts, 1):
            chunk = pipe.chunk_map[cid]
            hits.append(
                {
                    "rank": rank,
                    "chunk_id": cid,
                    "doc_id": chunk.doc_id,
                    "title": title,
                    "text": text,
                    "score": score,
                }
            )
        results.append({"config": name, "hits": hits})
    return results


def overlap_summary(results):
    """Which doc ids appear in the top-k of more than one config.

    Returns {doc_id: [config, ...]} sorted by config, only for docs
    retrieved by at least two configs.
    """
    owners = defaultdict(set)
    for result in results:
        for hit in result["hits"]:
            owners[hit["doc_id"]].add(result["config"])
    return {
        doc_id: sorted(cfgs)
        for doc_id, cfgs in owners.items()
        if len(cfgs) > 1
    }


def format_report(query, results, k=5, width=100):
    """Render the comparison as stacked per-config tables + overlap summary."""
    lines = [f'Q: "{query}"', ""]
    for result in results:
        lines.append(f"=== {result['config']} (top {k}) ===")
        lines.append(f"{'rank':<5}{'doc_id':<10}{'score':>9}  title")
        lines.append("-" * width)
        for hit in result["hits"]:
            title = (hit["title"] or hit["chunk_id"])[:40]
            lines.append(
                f"{hit['rank']:<5}{hit['doc_id']:<10}{hit['score']:>9.3f}  {title}"
            )
            snippet = hit["text"].replace("\n", " ")[:width]
            lines.append(f"      {snippet}")
        lines.append("")
    overlap = overlap_summary(results)
    lines.append("=== overlap: docs in the top-k of multiple configs ===")
    if overlap:
        for doc_id in sorted(overlap):
            lines.append(f"  {doc_id}: {', '.join(overlap[doc_id])}")
    else:
        lines.append("  (no shared documents)")
    return "\n".join(lines)
