"""Query expansion via pseudo-relevance feedback (RM3-lite).

The idea: the user's query is often a poor description of what the relevant
documents look like. So we run the query once, *assume* the top-ranked chunks
are relevant (pseudo-relevance feedback), harvest their most characteristic
terms, and search again with the expanded query.

RM3-lite estimates a relevance model from the feedback set:

    P(w|R) = sum_d P(w|d) * P(d|Q)

where P(w|d) is the chunk's term distribution and P(d|Q) is the initial
retrieval score normalized over the feedback set. Candidate terms are ranked
by P(w|R) * idf(w) (corpus idf, so corpus-frequent filler never wins), the top
`fb_terms` terms not already in the query are kept, and their weights are
scaled by `lambda_` (the RM3 interpolation weight) with the strongest term
at exactly `lambda_`. Original query terms keep weight 1.0.

All term handling uses the shared tokenizer (lowercase, stopword removal,
stemming), so expansion terms always match the index vocabulary.
"""

import inspect
import math
from collections import Counter, defaultdict

from .text import tokenize


class RM3Expander:
    """Selects RM3-style expansion terms from pseudo-relevant chunks.

    chunks:   the indexed chunks (same objects the retriever was built on)
    fb_docs:  how many top-ranked chunks count as pseudo-relevant feedback
    fb_terms: how many expansion terms to add to the query (0 = none)
    lambda_:  interpolation weight for expansion terms, in [0, 1];
              0 disables expansion (rankings identical to the base retriever)
    """

    def __init__(self, chunks, fb_docs=10, fb_terms=20, lambda_=0.6):
        if not isinstance(fb_docs, int) or isinstance(fb_docs, bool) or fb_docs < 1:
            raise ValueError(f"fb_docs must be a positive int, got {fb_docs!r}")
        if not isinstance(fb_terms, int) or isinstance(fb_terms, bool) or fb_terms < 0:
            raise ValueError(
                f"fb_terms must be a non-negative int, got {fb_terms!r}"
            )
        if not isinstance(lambda_, (int, float)) or not 0.0 <= lambda_ <= 1.0:
            raise ValueError(f"lambda_ must be in [0, 1], got {lambda_!r}")
        self.chunks = {c.chunk_id: c for c in chunks}
        self.fb_docs = fb_docs
        self.fb_terms = fb_terms
        self.lambda_ = float(lambda_)
        # corpus idf with the same +1 smoothing BM25Retriever uses
        df = Counter()
        for c in chunks:
            for t in set(tokenize(c.text)):
                df[t] += 1
        n = len(chunks)
        self.idf = {
            t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()
        }

    def expand(self, query, ranked, query_terms=None):
        """Build the expanded query model.

        ranked: [(chunk_id, score), ...] from the initial retrieval pass.
        query_terms: ordered unique query terms; defaults to tokenize(query).
            Pass the base retriever's own tokenization here so the expanded
            model matches what the base would have scored (e.g. BM25's
            phrase bigrams are kept as query terms at weight 1.0).
        Returns {term: weight}: original query terms at 1.0 plus expansion
        terms at weight <= lambda_. Deterministic for identical inputs.
        """
        if query_terms is None:
            query_terms = list(dict.fromkeys(tokenize(query)))
        qset = set(query_terms)
        weights = {t: 1.0 for t in query_terms}
        feedback = [
            (cid, score)
            for cid, score in ranked[: self.fb_docs]
            if cid in self.chunks
        ]
        if feedback and self.fb_terms > 0 and self.lambda_ > 0:
            rel_model = self._relevance_model(feedback, exclude=qset)
            if rel_model:
                peak = max(rel_model.values())
                for term, w in rel_model.items():
                    weights[term] = self.lambda_ * w / peak
        return weights

    def _relevance_model(self, feedback, exclude):
        # P(d|Q): normalized initial scores; uniform when scores are all
        # non-positive (e.g. zero-score or negative-score feedback).
        total = sum(max(score, 0.0) for _, score in feedback)
        if total > 0:
            p_d = {cid: max(score, 0.0) / total for cid, score in feedback}
        else:
            p_d = {cid: 1.0 / len(feedback) for cid, _ in feedback}
        model = defaultdict(float)
        for cid, _ in feedback:
            tokens = tokenize(self.chunks[cid].text)
            if not tokens:
                continue
            tf = Counter(tokens)
            length = len(tokens)
            p = p_d[cid]
            for term, freq in tf.items():
                if term in exclude:
                    continue
                model[term] += (freq / length) * p * self.idf.get(term, 0.0)
        # deterministic top-fb_terms: weight desc, then term asc
        ranked_terms = sorted(model.items(), key=lambda kv: (-kv[1], kv[0]))
        return dict(ranked_terms[: self.fb_terms])


def _supports_query_weights(retriever):
    try:
        params = inspect.signature(retriever.search).parameters
    except (TypeError, ValueError):
        return False
    return "query_weights" in params


class ExpandedRetriever:
    """Wraps a base retriever with RM3-lite query expansion.

    search() runs the base retriever once to collect pseudo-relevant
    feedback chunks, expands the query with RM3Expander, then runs the base
    retriever again with the weighted query. Base retrievers that accept a
    `query_weights` keyword (BM25Retriever, TfidfRetriever) get term-level
    weights; anything else falls back to a plain expanded query string
    (expansion terms appended to the original query).
    """

    def __init__(self, base, fb_docs=10, fb_terms=20, lambda_=0.6):
        if not hasattr(base, "search"):
            raise ValueError("base must expose .search(query, k)")
        chunks = getattr(base, "chunks", None)
        if not chunks:
            raise ValueError("base retriever must expose a non-empty .chunks map")
        self.base = base
        self.expander = RM3Expander(
            list(chunks.values()),
            fb_docs=fb_docs,
            fb_terms=fb_terms,
            lambda_=lambda_,
        )
        self.fb_docs = self.expander.fb_docs
        self.fb_terms = self.expander.fb_terms
        self.lambda_ = self.expander.lambda_

    def expand_query(self, query):
        """Run the feedback pass; return the weighted expanded query model."""
        initial = self.base.search(query, k=self.fb_docs)
        # score the same query terms the base would (keeps BM25 bigrams,
        # and keeps rankings identical to base when expansion is disabled)
        tokenize_query = getattr(self.base, "_tokens", None)
        if tokenize_query is not None:
            qterms = list(dict.fromkeys(tokenize_query(query)))
        else:
            qterms = list(dict.fromkeys(tokenize(query)))
        return self.expander.expand(query, initial, query_terms=qterms)

    def search(self, query, k=5):
        weights = self.expand_query(query)
        if _supports_query_weights(self.base):
            return self.base.search(query, k=k, query_weights=weights)
        # plain-text fallback: original query plus expansion terms
        extra = " ".join(sorted(t for t, w in weights.items() if w < 1.0))
        expanded = f"{query} {extra}".strip()
        return self.base.search(expanded, k=k)
