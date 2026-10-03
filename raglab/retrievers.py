"""Retrievers: BM25, TF-IDF, dense (optional), and hybrid RRF fusion.

All retrievers share one interface:

    retriever.search(query, k=5) -> [(chunk_id, score), ...]

Scores are comparable only within a retriever, never across retrievers;
HybridRetriever fuses *ranks*, not raw scores, for exactly this reason.
"""

import math
from collections import Counter, defaultdict

from .text import tokenize, bigrams


class BM25Retriever:
    """Okapi BM25, implemented from scratch. k1=1.5, b=0.75 (standard).

    With use_bigrams=True, phrase bigrams ("red_planet") are indexed alongside
    unigrams, so exact phrase matches score higher than scattered terms.
    """

    def __init__(self, chunks, k1=1.5, b=0.75, use_bigrams=False):
        self.chunks = {c.chunk_id: c for c in chunks}
        self.k1, self.b = k1, b
        self.use_bigrams = use_bigrams
        self.doc_tf = {}
        self.doc_len = {}
        df = Counter()
        for c in chunks:
            tokens = self._tokens(c.text)
            tf = Counter(tokens)
            self.doc_tf[c.chunk_id] = tf
            self.doc_len[c.chunk_id] = len(tokens)
            for t in tf:
                df[t] += 1
        self.n_docs = len(chunks)
        self.avgdl = sum(self.doc_len.values()) / max(1, self.n_docs)
        # idf with the standard +1 smoothing so common terms never go negative
        self.idf = {
            t: math.log(1 + (self.n_docs - f + 0.5) / (f + 0.5))
            for t, f in df.items()
        }

    def _tokens(self, text):
        tokens = tokenize(text)
        if self.use_bigrams:
            tokens = tokens + ["_".join(b) for b in bigrams(tokens)]
        return tokens

    def search(self, query, k=5):
        qtf = Counter(self._tokens(query))
        scores = defaultdict(float)
        for term in qtf:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for cid, tf in self.doc_tf.items():
                f = tf.get(term, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[cid] / self.avgdl)
                scores[cid] += idf * f * (self.k1 + 1) / denom
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:k]


class TfidfRetriever:
    """Classic TF-IDF vector space model with cosine similarity."""

    def __init__(self, chunks):
        self.chunks = {c.chunk_id: c for c in chunks}
        df = Counter()
        self.doc_tf = {}
        for c in chunks:
            tf = Counter(tokenize(c.text))
            self.doc_tf[c.chunk_id] = tf
            for t in tf:
                df[t] += 1
        n = len(chunks)
        self.idf = {t: math.log(n / f) for t, f in df.items()}
        self.doc_vec = {}
        for cid, tf in self.doc_tf.items():
            total = sum(tf.values())
            vec = {t: (f / total) * self.idf[t] for t, f in tf.items()}
            self.doc_vec[cid] = vec
        self.doc_norm = {
            cid: math.sqrt(sum(v * v for v in vec.values())) or 1.0
            for cid, vec in self.doc_vec.items()
        }

    def _query_vec(self, query):
        tf = Counter(tokenize(query))
        total = sum(tf.values()) or 1
        return {t: (f / total) * self.idf[t] for t, f in tf.items() if t in self.idf}

    def search(self, query, k=5):
        qvec = self._query_vec(query)
        qnorm = math.sqrt(sum(v * v for v in qvec.values())) or 1.0
        scores = []
        for cid, dvec in self.doc_vec.items():
            dot = sum(qvec[t] * dvec.get(t, 0.0) for t in qvec)
            scores.append((cid, dot / (qnorm * self.doc_norm[cid])))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


class DenseRetriever:
    """Dense embeddings via sentence-transformers (optional dependency).

    Raises a clear error if the package is missing, so the zero-dependency
    core (BM25 / TF-IDF / hybrid) always works.
    """

    def __init__(self, chunks, model_name="all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "DenseRetriever needs `pip install sentence-transformers`. "
                "Use BM25Retriever, TfidfRetriever, or HybridRetriever for the zero-dependency path."
            ) from exc
        self.chunks = {c.chunk_id: c for c in chunks}
        self.model = SentenceTransformer(model_name)
        texts = [c.text for c in chunks]
        self.ids = [c.chunk_id for c in chunks]
        self.embeddings = self.model.encode(texts, normalize_embeddings=True)

    def search(self, query, k=5):
        import numpy as np

        q = self.model.encode([query], normalize_embeddings=True)[0]
        sims = self.embeddings @ q
        top = np.argsort(sims)[::-1][:k]
        return [(self.ids[i], float(sims[i])) for i in top]


class HybridRetriever:
    """Reciprocal Rank Fusion over member retrievers.

    RRF fuses ranks (not raw scores), which is why it can combine a BM25
    lexical retriever with a dense semantic one without score calibration:
        rrf(d) = sum( 1 / (c + rank_i(d)) )
    """

    def __init__(self, retrievers, c=60):
        self.retrievers = retrievers
        self.c = c

    def search(self, query, k=5):
        fused = defaultdict(float)
        for r in self.retrievers:
            # ask each member for a deep candidate list, then fuse ranks
            for rank, (cid, _) in enumerate(r.search(query, k=50)):
                fused[cid] += 1.0 / (self.c + rank + 1)
        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        return ranked[:k]
