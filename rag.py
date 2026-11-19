"""Command-line interface for live generation and offline experiment review."""
import argparse
import json
import sys

from corpus import load, questions
from evaluation import metrics, read_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("setup", "index", "metrics", "verify"):
        commands.add_parser(name)
    replay = commands.add_parser("replay")
    replay.add_argument("id")
    methods = ("dense", "hybrid", "scoped", "reranked", "final")
    ask = commands.add_parser("ask")
    ask.add_argument("question")
    batch = commands.add_parser("run")
    batch.add_argument("--split", choices=("dev", "test", "holdout"), default="test")
    for sub in (ask, batch):
        sub.add_argument("--method", choices=methods, default="final")
    args = parser.parse_args()
    if args.command in ("metrics", "verify"):
        metrics()
        return
    if args.command == "replay":
        for row in read_rows("final.jsonl"):
            if row["id"] == args.id:
                print(json.dumps(row, ensure_ascii=False, indent=2))
                return
        parser.error("Unknown question ID")
    from retrieval import Search, check_assets
    if args.command == "setup":
        check_assets(download=True)
        return
    raw, passages = load()
    search = Search(passages)
    if args.command == "index":
        print(f"Indexed {len(passages)} passages")
        return
    from generator import api_key
    from pipeline import run
    api_key()
    batch = questions(args.split) if args.command == "run" else [{"id": "live", "question": args.question}]
    for q in batch:
        row = run(search, passages, raw, q["question"], args.method)
        print(json.dumps({"id": q["id"], **row}, ensure_ascii=False), flush=True)
        if row["status"] == "generation_failed":
            sys.exit(1)


if __name__ == "__main__":
    main()
