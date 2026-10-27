# raglab — a RAG evaluation lab

Everyone builds a RAG chatbot. Almost nobody measures whether the retrieval actually works. **raglab** is a zero-dependency RAG toolkit with a built-in evaluation harness: it benchmarks retrieval strategies head-to-head on labeled Q&A data and tells you, with numbers, what each technique contributes.

## The headline result

On the bundled 106-document, 53-question benchmark (with adversarial traps like *Mars the planet vs. Mars the candy company*):

| config | recall@5 | mrr | ndcg@5 | map |
|---|---|---|---|---|
| tfidf | 1.000 | 0.959 | 0.970 | 0.959 |
| bm25 | 1.000 | 0.962 | 0.972 | 0.962 |
| bm25 + phrases | 1.000 | 0.972 | 0.979 | 0.972 |
| bm25 + phrases + rm3 | 1.000 | 0.953 | 0.965 | 0.953 |
| bm25 + phrases + rerank | 1.000 | 0.981 | 0.986 | 0.981 |

**+2.3% MRR** from TF-IDF to the full stack. Each rung of the ladder earns its place — run `raglab eval` to reproduce it. (RM3 expansion is the honest exception: on this already-saturated corpus it scores −2.0% vs. the phrases rung — see design notes.)

The benchmark grew from 38 docs / 33 questions to 106 docs / 53 questions on 2026-10-08: 34 new lexical trap pairs (Turkey country/bird, crane bird/machine, Java island/language, Rust language/corrosion, git branch/tree branch, ML feature/film feature, ...) plus production-ML topics (concept drift, bias, weights, feature stores) alongside the original set.

## What's inside

**Retrievers** (all from scratch, zero dependencies):
- `BM25Retriever` — Okapi BM25 with Porter-style stemming and optional phrase (bigram) indexing
- `TfidfRetriever` — classic vector space model, cosine similarity
- `DenseRetriever` — sentence-transformers embeddings *(optional)*
- `HybridRetriever` — Reciprocal Rank Fusion over any member retrievers (rank-based, so lexical + dense fuse without score calibration)

**Reranking:**
- `FeatureReranker` — interpretable second-stage scorer (term coverage, bigram overlap, title match, first-stage rank). Fixes the classic failure where a lexically similar distractor outranks the true match.

**Query expansion:**
- `RM3Expander` / `ExpandedRetriever` — RM3-lite pseudo-relevance feedback. Runs the base retriever once, estimates a relevance model `P(w|R)` from the top-`fb_docs` chunks (term distribution weighted by normalized initial score, times corpus idf), and re-searches with the top-`fb_terms` new terms interpolated at weight `lambda_` (original query terms stay at 1.0). Composes with any retriever: `ExpandedRetriever(BM25Retriever(chunks))`, or via pipeline configs `"bm25+rm3"`, `"bm25+phrases+rm3"`, `"bm25+phrases+rm3+rerank"`. `lambda_=0` (or `fb_terms=0`) reproduces the base ranking exactly.

**Evaluation:**
- Standard IR metrics: Recall@k, Precision@k, MRR, NDCG@k, MAP — all tested against hand-computed values
- `EvalHarness` — runs every config over the Q&A set and emits a Markdown ablation report with lift-over-baseline
- **LLM-as-judge answer metrics** (`raglab.judge`): `faithfulness(question, answer, contexts)` scores the share of answer claims supported by the retrieved contexts; `answer_relevancy(question, answer)` scores how well the answer addresses the question. Both delegate judgment to a pluggable judge — any callable `judge(task, **kwargs) -> score` with `task` in `{"claim_supported", "answer_relevant"}`. Ships with `HeuristicJudge`, a deterministic offline token-overlap baseline for tests and demos. `raglab eval --judge heuristic` appends both metrics to the ablation report.
- **Hybrid weight (alpha) sweep** (`raglab eval --alpha-sweep 0,0.25,0.5,0.75,1`): re-scores the bundled benchmark at each hybrid interpolation weight — alpha weights the BM25 member, `1 - alpha` the TF-IDF member (weighted RRF, still rank-based, so no score calibration is needed). Prints a per-alpha MRR/NDCG table to the terminal, appends it to the report, and names the best alpha; `--report-json` adds a structured `alpha_sweep` section. Malformed grids (non-numbers, values outside [0, 1]) fail closed with an error. The sweep is diagnostic: the same questions tune and measure the weights, so the winner describes this benchmark, not a generalization claim. On the bundled corpus both members are lexical, so the sweep is nearly flat (0.959–0.962 MRR) — fusion earns its keep with diverse members (lexical + dense).
- **Hardest questions** (`raglab eval --hardest N`): lists up to N questions the best-MRR config retrieves nothing relevant for (recall@k = 0), in the report and in `--report-json` as `hardest`. A zero-recall question is where to look next — a retrieval gap, a chunking gap, or a mislabeled gold doc — not a verdict on which one.

## Quickstart

```bash
pip install raglab
raglab ask "Which planet is known as the Red Planet?"
raglab eval --report report.md        # reproduce the benchmark above
raglab eval --judge heuristic         # add faithfulness / answer relevancy (offline baseline judge)
raglab eval --alpha-sweep 0,0.25,0.5,0.75,1 --report-json report.json  # sweep hybrid member weights (diagnostic)
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
- **RM3 is neutral-to-negative here, and that's the point.** On this corpus recall@5 is already 1.000 and MRR sits at 0.972 before expansion, so harvesting terms from already-correct top docs mostly shuffles near-ties (−2.0% MRR at the default `fb_docs=10, fb_terms=20, lambda_=0.6`; neutral with fewer terms). RM3 earns its keep on vocabulary mismatch — the test suite includes a fixture where the query term appears in only one doc and an expansion bridging term retrieves a second, otherwise-invisible doc. `raglab compare --configs bm25,bm25+phrases+rm3` shows the expanded ranking side-by-side on any query.
- **Traps, not just questions.** The benchmark includes lexical distractors (Mercury the planet vs. the element, Amazon the rainforest vs. the company) so scores reflect disambiguation, not just keyword overlap.
- **No LLM required.** Retrieval quality is measurable without generation; the pipeline returns ranked contexts and a `Generator` hook is left open for your LLM.
- **The heuristic judge is a baseline, not a verdict.** `HeuristicJudge` scores faithfulness by token coverage of each claim in the retrieved contexts and relevancy by question-term overlap in the answer. It cannot see paraphrase, negation, or implication — a claim reworded with no overlapping terms scores 0 even if the context fully supports it. It also saturates on easy ground: retrieval on this benchmark is so strong (recall@5 = 1.000) that every config scores 1.000 on gold-answer faithfulness (mismatched contexts do score 0.0, so the metric discriminates when retrieval actually fails). Use it to keep tests and demos offline; plug a real LLM judge for numbers you'd publish:

```python
def llm_judge(task, **kwargs):
    # build a prompt from task + kwargs, call your LLM, parse "yes"/"no"
    verdict = call_llm(prompt_for(task, **kwargs))
    return 1.0 if verdict.strip().lower().startswith("yes") else 0.0

from raglab import faithfulness
faithfulness(question, answer, context_texts, judge=llm_judge)
```

## Tests

108 tests, all passing: `pytest tests/`
