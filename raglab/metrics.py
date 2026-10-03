"""Retrieval metrics: the honest numbers behind every RAG claim.

All metrics take:
    ranked_ids  - chunk/doc ids in rank order (best first)
    relevant    - set of relevant chunk/doc ids (ground truth)
"""

import math


def _doc_id(chunk_id):
    return chunk_id.split("#c")[0]


def to_doc_ids(ranked_chunk_ids):
    """Collapse chunk ranking to doc ranking, keeping first occurrence order."""
    seen, docs = set(), []
    for cid in ranked_chunk_ids:
        d = _doc_id(cid)
        if d not in seen:
            seen.add(d)
            docs.append(d)
    return docs


def recall_at_k(ranked, relevant, k):
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & set(relevant)) / len(relevant)


def precision_at_k(ranked, relevant, k):
    if k == 0:
        return 0.0
    return len(set(ranked[:k]) & set(relevant)) / k


def mrr(ranked, relevant):
    """Mean Reciprocal Rank (single query): 1 / rank of first relevant hit."""
    for i, cid in enumerate(ranked):
        if cid in relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranked, relevant, k):
    """NDCG@k with binary relevance."""
    dcg = 0.0
    for i, cid in enumerate(ranked[:k]):
        if cid in relevant:
            dcg += 1.0 / math.log2(i + 2)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg else 0.0


def average_precision(ranked, relevant):
    if not relevant:
        return 0.0
    hits, total = 0, 0.0
    for i, cid in enumerate(ranked):
        if cid in relevant:
            hits += 1
            total += hits / (i + 1)
    return total / len(relevant)


def evaluate_run(ranked_per_query, relevant_per_query, k=5):
    """Aggregate metrics over many queries. Returns dict of metric -> mean."""
    queries = list(ranked_per_query)
    agg = {
        f"recall@{k}": 0.0,
        f"precision@{k}": 0.0,
        "mrr": 0.0,
        f"ndcg@{k}": 0.0,
        "map": 0.0,
    }
    for qid in queries:
        ranked = ranked_per_query[qid]
        rel = relevant_per_query[qid]
        agg[f"recall@{k}"] += recall_at_k(ranked, rel, k)
        agg[f"precision@{k}"] += precision_at_k(ranked, rel, k)
        agg["mrr"] += mrr(ranked, rel)
        agg[f"ndcg@{k}"] += ndcg_at_k(ranked, rel, k)
        agg["map"] += average_precision(ranked, rel)
    n = max(1, len(queries))
    return {m: v / n for m, v in agg.items()}
