"""Rerankers: second-stage scoring of a candidate list.

Interface: rerank(query, candidates, k) -> [(chunk_id, score), ...]
where candidates is [(chunk_id, base_score), ...].
"""

from collections import Counter

from .text import tokenize, bigrams


class FeatureReranker:
    """Lightweight, from-scratch reranker over interpretable features.

    Features per candidate:
      1. query term coverage  (fraction of distinct query terms present)
      2. bigram overlap       (phrase-level match, rewards exact phrases)
      3. title match          (query terms appearing in the chunk title)
      4. base score rank      (reciprocal of the first-stage rank)

    The weighted sum is deliberately simple and deterministic: it fixes the
    classic first-stage failure where a lexically similar but irrelevant chunk
    outranks the true match.
    """

    def __init__(self, chunks, weights=(0.4, 0.3, 0.15, 0.15)):
        self.chunks = {c.chunk_id: c for c in chunks}
        self.weights = weights

    def _features(self, query, chunk, rank):
        qtokens = tokenize(query)
        ctokens = set(tokenize(chunk.text))
        qtokens_set = set(qtokens)
        coverage = len(qtokens_set & ctokens) / max(1, len(qtokens_set))
        qbig = set(bigrams(qtokens))
        cbig = set(bigrams(tokenize(chunk.text)))
        bigram_overlap = len(qbig & cbig) / max(1, len(qbig))
        title_tokens = set(tokenize(chunk.title))
        title_match = len(qtokens_set & title_tokens) / max(1, len(qtokens_set))
        rank_feature = 1.0 / (rank + 1)
        return (coverage, bigram_overlap, title_match, rank_feature)

    def rerank(self, query, candidates, k=5):
        scored = []
        for rank, (cid, _) in enumerate(candidates):
            chunk = self.chunks[cid]
            feats = self._features(query, chunk, rank)
            score = sum(w * f for w, f in zip(self.weights, feats))
            scored.append((cid, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]


class CrossEncoderReranker:
    """Cross-encoder reranking via sentence-transformers (optional)."""

    def __init__(self, chunks, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "CrossEncoderReranker needs `pip install sentence-transformers`."
            ) from exc
        self.chunks = {c.chunk_id: c for c in chunks}
        self.model = CrossEncoder(model_name)

    def rerank(self, query, candidates, k=5):
        pairs = [(query, self.chunks[cid].text) for cid, _ in candidates]
        scores = self.model.predict(pairs)
        ranked = sorted(
            zip([cid for cid, _ in candidates], scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return [(cid, float(s)) for cid, s in ranked[:k]]
