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
    from .harness import EvalHarness, load_jsonl

    docs = load_jsonl(args.docs or os.path.join(DATA_DIR, "documents.jsonl"))
    questions = load_jsonl(args.questions or os.path.join(DATA_DIR, "questions.jsonl"))
    harness = EvalHarness(docs, questions)
    results = harness.run(k=args.k)
    report = harness.markdown_report(results, k=args.k)
    print(report)
    if args.report:
        with open(args.report, "w") as f:
            f.write(report + "\n")
        print(f"\nreport written to {args.report}")


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
                                "hybrid", "hybrid+rerank"])
    p_ask.add_argument("-k", type=int, default=3)
    p_ask.set_defaults(func=cmd_ask)

    p_eval = sub.add_parser("eval", help="run the retrieval ablation benchmark")
    p_eval.add_argument("--docs", default=None)
    p_eval.add_argument("--questions", default=None)
    p_eval.add_argument("-k", type=int, default=5)
    p_eval.add_argument("--report", default="eval_report.md")
    p_eval.set_defaults(func=cmd_eval)

    p_ing = sub.add_parser("ingest", help="index a directory of .txt/.md files")
    p_ing.add_argument("dir")
    p_ing.add_argument("--out", default="index.json")
    p_ing.set_defaults(func=cmd_ingest)

    args = parser.parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
