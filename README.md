# raglab — a RAG evaluation lab

Everyone builds a RAG chatbot. Almost nobody measures whether the retrieval actually works. **raglab** is a zero-dependency RAG toolkit with a built-in evaluation harness: it benchmarks retrieval strategies head-to-head on labeled Q&A data and tells you, with numbers, what each technique contributes.

## The headline result

On the bundled 38-document, 33-question benchmark (with adversarial traps like *Mars the planet vs. Mars the candy company*):

| config | recall@5 | mrr | ndcg@5 | map |
|---|---|---|---|---|
| tfidf | 1.000 | 0.934 | 0.951 | 0.934 |
| bm25 | 1.000 | 0.955 | 0.966 | 0.955 |
| bm25 + phrases | 1.000 | 0.970 | 0.978 | 0.970 |
| bm25 + phrases + rerank | 1.000 | 0.985 | 0.989 | 0.985 |

**+5.4% MRR** from TF-IDF to the full stack. Each rung of the ladder earns its place — run `raglab eval` to reproduce it.

## What's inside

**Retrievers** (all from scratch, zero dependencies):
- `BM25Retriever` — Okapi BM25 with Porter-style stemming and optional phrase (bigram) indexing
- `TfidfRetriever` — classic vector space model, cosine similarity
- `DenseRetriever` — sentence-transformers embeddings *(optional)*
- `HybridRetriever` — Reciprocal Rank Fusion over any member retrievers (rank-based, so lexical + dense fuse without score calibration)

**Reranking:**
- `FeatureReranker` — interpretable second-stage scorer (term coverage, bigram overlap, title match, first-stage rank). Fixes the classic failure where a lexically similar distractor outranks the true match.

**Evaluation:**
- Standard IR metrics: Recall@k, Precision@k, MRR, NDCG@k, MAP — all tested against hand-computed values
- `EvalHarness` — runs every config over the Q&A set and emits a Markdown ablation report with lift-over-baseline

## Quickstart

```bash
pip install raglab
raglab ask "Which planet is known as the Red Planet?"
raglab eval --report report.md        # reproduce the benchmark above
raglab ingest ./my-docs --out index.json
```

```python
from raglab import RagPipeline

pipe = RagPipeline(documents, config="bm25+phrases+rerank")
answer = pipe.ask("How do neural networks learn?")
for chunk_id, title, text, score in answer.contexts:
    print(title, round(score, 3))
```

## Design notes

- **Honest ablations.** RRF-hybrid of two lexical rankers adds ~nothing on this corpus (it's in the codebase, tested, and documented) — the measurable wins come from phrase matching and reranking. Hybrid fusion earns its keep when members are diverse (lexical + dense).
- **Traps, not just questions.** The benchmark includes lexical distractors (Mercury the planet vs. the element, Amazon the rainforest vs. the company) so scores reflect disambiguation, not just keyword overlap.
- **No LLM required.** Retrieval quality is measurable without generation; the pipeline returns ranked contexts and a `Generator` hook is left open for your LLM.

## Tests

35 tests, all passing: `pytest tests/`
