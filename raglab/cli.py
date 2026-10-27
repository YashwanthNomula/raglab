"""raglab CLI: ingest, ask, eval."""

import argparse
import json
import os
import sys

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def cmd_ask(args):
    from .harness import load_jsonl
    from .pipeline import RagPipeline

    docs_path = args.docs or os.path.join(DATA_DIR, "documents.jsonl")
    documents = load_jsonl(docs_path)
    pipe = RagPipeline(documents, config=args.config)
    answer = pipe.ask(args.question, k=args.k)
    print(f"\nQ: {answer.question}")
    print(f"retrieval: {answer.config}\n")
    for i, (cid, title, text, score) in enumerate(answer.contexts, 1):
        print(f"[{i}] {title} ({cid}, score={score:.3f})")
        print(f"    {text[:300]}{'...' if len(text) > 300 else ''}\n")


def cmd_eval(args):
    from .harness import EvalHarness, best_alpha, load_jsonl, parse_alpha_list
    from .judge import HeuristicJudge

    # fail closed before touching data: a malformed alpha grid exits here
    alphas = None
    if args.alpha_sweep is not None:
        try:
            alphas = parse_alpha_list(args.alpha_sweep)
        except ValueError as e:
            print(f"error: invalid --alpha-sweep: {e}", file=sys.stderr)
            return 2

    docs = load_jsonl(args.docs or os.path.join(DATA_DIR, "documents.jsonl"))
    questions = load_jsonl(args.questions or os.path.join(DATA_DIR, "questions.jsonl"))
    harness = EvalHarness(docs, questions)
    results = harness.run(k=args.k)
    hardest = None
    if args.hardest is not None:
        if args.hardest < 0:
            print("error: --hardest must be >= 0", file=sys.stderr)
            return 2
        hardest = harness.hardest_questions(results, k=args.k, n=args.hardest)
    judge_results, judge_label = None, ""
    if args.judge == "heuristic":
        judge = HeuristicJudge()
        judge_results = harness.run_judge_metrics(judge=judge, k=args.k)
        judge_label = "heuristic baseline"
    sweep_results = None
    if alphas is not None:
        sweep_results = harness.run_alpha_sweep(alphas, k=args.k)
    report = harness.markdown_report(results, k=args.k,
                                     judge_results=judge_results,
                                     judge_label=judge_label,
                                     sweep_results=sweep_results,
                                     hardest=hardest)
    print(report)
    if args.report:
        with open(args.report, "w") as f:
            f.write(report + "\n")
        print(f"\nreport written to {args.report}")
    if args.report_json:
        payload = {
            "k": args.k,
            "questions": len(questions),
            "documents": len(docs),
            "chunks": len(harness.chunks),
            "ablation": results,
            "judge": judge_results,
            "alpha_sweep": _sweep_payload(sweep_results, k=args.k),
            "hardest": (
                {
                    "best_config": hardest[0],
                    "zero_recall_count": hardest[1],
                    "questions": hardest[2],
                }
                if hardest is not None
                else None
            ),
        }
        with open(args.report_json, "w") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        print(f"json report written to {args.report_json}")
    return 0


def _sweep_payload(sweep_results, k=5):
    """Structured sweep section for the JSON report (None when no sweep)."""
    from .harness import best_alpha

    if sweep_results is None:
        return None
    best_a, best_res = best_alpha(sweep_results)
    return {
        "alpha_is_bm25_weight": True,
        "note": "diagnostic: same questions tune and measure the weights, "
                "so the winner describes this benchmark, not a generalization "
                "claim.",
        "results": {str(a): res for a, res in sweep_results.items()},
        "best_alpha": best_a,
        f"best_mrr": round(best_res["mrr"], 4),
        f"best_ndcg@{k}": round(best_res[f"ndcg@{k}"], 4),
    }


def cmd_compare(args):
    from .compare import (
        DEFAULT_COMPARE_CONFIGS,
        VALID_CONFIGS,
        compare_configs,
        format_report,
    )
    from .harness import load_jsonl

    docs_path = args.docs or os.path.join(DATA_DIR, "documents.jsonl")
    documents = load_jsonl(docs_path)
    configs = [c.strip() for c in args.configs.split(",") if c.strip()] or list(
        DEFAULT_COMPARE_CONFIGS
    )
    try:
        results = compare_configs(documents, args.question, configs, k=args.k)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        print(f"valid configs: {', '.join(VALID_CONFIGS)}", file=sys.stderr)
        return 2
    print(format_report(args.question, results, k=args.k))
    return 0


def cmd_ingest(args):
    from .pipeline import RagPipeline, Document

    docs = []
    for fname in sorted(os.listdir(args.dir)):
        if not fname.endswith((".txt", ".md")):
            continue
        with open(os.path.join(args.dir, fname)) as f:
            docs.append(
                Document(id=os.path.splitext(fname)[0], title=fname, text=f.read())
            )
    if not docs:
        print(f"no .txt/.md files found in {args.dir}")
        return 1
    pipe = RagPipeline(docs)
    pipe.save_index(args.out)
    print(f"indexed {len(docs)} documents -> {args.out}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="raglab", description="RAG evaluation lab")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ask = sub.add_parser("ask", help="ask a question over the document set")
    p_ask.add_argument("question")
    p_ask.add_argument("--docs", default=None)
    p_ask.add_argument("--config", default="bm25+phrases+rerank",
                       choices=["tfidf", "bm25", "bm25+phrases", "bm25+phrases+rerank",
                                "bm25+rm3", "bm25+phrases+rm3",
                                "bm25+phrases+rm3+rerank",
                                "hybrid", "hybrid+rerank"])
    p_ask.add_argument("-k", type=int, default=3)
    p_ask.set_defaults(func=cmd_ask)

    p_eval = sub.add_parser("eval", help="run the retrieval ablation benchmark")
    p_eval.add_argument("--docs", default=None)
    p_eval.add_argument("--questions", default=None)
    p_eval.add_argument("-k", type=int, default=5)
    p_eval.add_argument("--report", default="eval_report.md")
    p_eval.add_argument("--judge", default=None, choices=["heuristic"],
                        help="also score reference answers with judge metrics "
                             "(faithfulness, answer_relevancy); 'heuristic' uses "
                             "the built-in offline baseline judge")
    p_eval.add_argument("--alpha-sweep", default=None, metavar="A,B,C",
                        help="sweep the hybrid RRF member weight over the benchmark, "
                             "e.g. '0,0.25,0.5,0.75,1'; alpha weights the BM25 member, "
                             "1-alpha the TF-IDF member; values must be in [0,1]")
    p_eval.add_argument("--hardest", default=None, type=int, metavar="N",
                        help="also list up to N questions the best-MRR config "
                             "retrieves nothing relevant for (recall@k = 0), "
                             "as a 'Hardest questions' section in the report")
    p_eval.add_argument("--report-json", default=None, metavar="PATH",
                        help="also write the report as JSON, including an "
                             "'alpha_sweep' section when --alpha-sweep is used")
    p_eval.set_defaults(func=cmd_eval)

    p_cmp = sub.add_parser(
        "compare",
        help="compare retrieval configs side-by-side on a single query",
    )
    p_cmp.add_argument("question")
    p_cmp.add_argument("--docs", default=None)
    p_cmp.add_argument(
        "--configs",
        default="bm25,tfidf,hybrid",
        help="comma-separated config names, e.g. 'bm25,tfidf,hybrid'",
    )
    p_cmp.add_argument("-k", type=int, default=5)
    p_cmp.set_defaults(func=cmd_compare)

    p_ing = sub.add_parser("ingest", help="index a directory of .txt/.md files")
    p_ing.add_argument("dir")
    p_ing.add_argument("--out", default="index.json")
    p_ing.set_defaults(func=cmd_ingest)

    args = parser.parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
