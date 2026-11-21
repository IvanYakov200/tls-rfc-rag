"""Command-line interface for live generation and offline experiment review."""
import argparse
import json
import sys

from .corpus import load, questions
from .evaluation import verify, read_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("setup", "index", "metrics", "verify"):
        commands.add_parser(name)
    for name in ("review-template", "score"):
        sub = commands.add_parser(name)
        sub.add_argument("--split", choices=("dev", "test", "test-v1", "holdout", "challenge", "blind"), required=True)
        sub.add_argument("--answers", required=True)
        sub.add_argument("--review", required=True)
    replay = commands.add_parser("replay")
    replay.add_argument("id")
    methods = ("dense", "hybrid", "scoped", "reranked", "final", "multi", "faceted", "direct", "single_no_repeat")
    replay.add_argument("--method", choices=methods, default="direct")
    ask = commands.add_parser("ask")
    ask.add_argument("question")
    batch = commands.add_parser("run")
    batch.add_argument("--split", choices=("dev", "test", "test-v1", "holdout", "challenge", "blind"), default="test")
    for sub in (ask, batch):
        sub.add_argument("--method", choices=methods, default="direct")
    args = parser.parse_args()
    if args.command in ("review-template", "score"):
        from .run_review import review_run
        try:
            review_run(args)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            parser.error(str(exc))
        return
    if args.command in ("metrics", "verify"):
        verify()
        return
    if args.command == "replay":
        if args.method == "direct":
            rows = read_rows("all_direct/answers.jsonl")
        else:
            rows = read_rows("comparisons.jsonl")
        for row in rows:
            if row["id"] == args.id and row["method"] == args.method:
                print(json.dumps(row, ensure_ascii=False, indent=2))
                return
        parser.error("No saved answer for this question and method")
    from .retrieval import Search, check_assets
    if args.command == "setup":
        check_assets(download=True)
        return
    raw, passages = load()
    search = Search(passages)
    if args.command == "index":
        print(f"Indexed {len(passages)} passages")
        return
    from .llm_client import api_key
    from .pipeline import run
    api_key()
    batch = questions(args.split) if args.command == "run" else [{"id": "live", "question": args.question}]
    failed = False
    for q in batch:
        row = run(search, passages, raw, q["question"], args.method)
        print(json.dumps({"id": q["id"], **row}, ensure_ascii=False), flush=True)
        if row["status"] == "generation_failed":
            failed = True
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
