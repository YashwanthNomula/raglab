# raglab evaluation report

Questions: 53 | Documents: 106 | Chunks: 123 | k=5

| config | recall@5 | precision@5 | mrr | ndcg@5 | map |
|---|---|---|---|---|---|
| tfidf | 1.000 | 0.204 | 0.959 | 0.970 | 0.959 |
| bm25 | 1.000 | 0.204 | 0.962 | 0.972 | 0.962 |
| bm25 + phrases | 1.000 | 0.204 | 0.972 | 0.979 | 0.972 |
| bm25 + phrases + rm3 | 1.000 | 0.204 | 0.953 | 0.965 | 0.953 |
| bm25 + phrases + rerank | 1.000 | 0.204 | 0.981 | 0.986 | 0.981 |

## Lift over tfidf baseline (MRR)
- **bm25**: +0.3% MRR
- **bm25 + phrases**: +1.3% MRR
- **bm25 + phrases + rm3**: -0.7% MRR
- **bm25 + phrases + rerank**: +2.3% MRR

## LLM-as-judge: reference-answer metrics (heuristic baseline)

| config | faithfulness | answer_relevancy |
|---|---|---|
| tfidf | 1.000 | 0.662 |
| bm25 | 1.000 | 0.662 |
| bm25 + phrases | 1.000 | 0.662 |
| bm25 + phrases + rm3 | 1.000 | 0.662 |
| bm25 + phrases + rerank | 1.000 | 0.662 |
