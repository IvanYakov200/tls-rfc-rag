"""Run from any directory. Replay and validation require only Python's stdlib."""
import argparse
import hashlib
import os
import tempfile
import json
from pathlib import Path

from corpus import CACHE, ROOT, load, questions, slices

K = 5
RESULTS = ROOT / "results"


def dump(obj):
    return json.dumps(obj, ensure_ascii=False)


def coverage(group, spans, raw):
    def covered(s):
        start, end = s["start_byte"], s["end_byte"]
        blob = raw[s["rfc"]]
        cursor = start
        for rfc, left, right in sorted(spans):
            if rfc != s["rfc"] or right <= cursor or left >= end:
                continue
            if blob[cursor:min(end, left)].strip():
                return False
            cursor = max(cursor, right)
        return not blob[min(cursor, end):end].strip()
    return sum(covered(s) for s in group) / len(group)


def record(q, method, hits, raw):
    spans = [(p.rfc, p.start, p.end) for p in hits]
    groups = q.get("gold_evidence_sets", [])
    return {"id": q["id"], "question": q["question"], "method": method,
            "retrieved": [p.key for p in hits],
            "recall_at_5": max((coverage(g, spans, raw) for g in groups), default=None)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("verify", "setup", "index"):
        sub.add_parser(name)
    for name in ("run", "replay", "metrics"):
        p = sub.add_parser(name)
        p.add_argument("--split", choices=["dev", "test"], default="dev")
        if name == "run":
            p.add_argument("--methods", nargs="+", choices=["A", "B", "C"], default=["A","B"])
        if name == "replay":
            p.add_argument("--id", default=None)
            p.add_argument("--method", default="A")
    p = sub.add_parser("ask")
    p.add_argument("question")
    p.add_argument("--method", choices=["A", "B", "C"], default="B")
    args = parser.parse_args()
    raw, passages = load()
    if args.cmd == "verify":
        print(f"Verified {len(raw)} RFC checksums; {len(passages)} passages")
        for path in sorted((ROOT / "data").glob("*.jsonl")):
            qs = questions(path.stem)
            assert len(set(q["id"] for q in qs)) == len(qs)
            for q in qs:
                assert bool(q["gold_evidence_sets"]) == q["answerable"]
                for group in q["gold_evidence_sets"]:
                    assert group
                    for s in group:
                        assert 0 <= s["start_byte"] < s["end_byte"] <= len(raw[s["rfc"]])
                    assert all(x.strip() for x in slices(raw, group))
            print(f"Verified {len(qs)} question records in {path.name}")
        for path in sorted(RESULTS.glob("*-?.jsonl")):
            for line in path.read_text().splitlines():
                row = json.loads(line)
                for c in row["claims"]:
                    assert slices(raw, [c])[0] == c["quote"]
            print(f"Verified citation bytes in {path.name}")
        return
    if args.cmd in ("metrics", "replay"):
        files = sorted(RESULTS.glob(f"{args.split}-?.jsonl"))
        if not files:
            parser.error("No saved runs for this split")
        for path in files:
            rows = [json.loads(s) for s in path.read_text().splitlines()]
            if args.cmd == "metrics":
                scored = [r["recall_at_5"] for r in rows if r["recall_at_5"] is not None]
                print(path.stem, f"Recall@5={sum(scored)/len(scored):.3f}, complete={scored.count(1)}/{len(scored)}, answered={sum(r['status']=='answered' for r in rows)}/{len(rows)}")
            else:
                for r in rows:
                    if r["method"] == args.method and (not args.id or r["id"] == args.id):
                        print(r["id"], r["question"], r["status"])
                        for c in r["claims"]:
                            quote = slices(raw, [c])[0]
                            if quote != c["quote"]:
                                raise ValueError("Saved quote differs from source")
                            print(f"  {c['text']}\n  RFC {c['rfc']} §{c['section']}, bytes [{c['start_byte']}, {c['end_byte']})\n  {quote.strip()}")
        return
    from retrieval import Search, check_assets
    if args.cmd == "setup":
        check_assets(download=True)
        return
    search = Search(passages)
    if args.cmd == "index":
        print(f"Indexed {len(passages)} passages")
        return
    from local_llm import answer, digest, translate
    model_sha = digest()
    methods = args.methods if args.cmd == "run" else [args.method]
    qs = questions(args.split) if args.cmd == "run" else [{"id": "live", "question": args.question}]
    CACHE.mkdir(exist_ok=True)
    run_cache = CACHE / "runs"
    run_cache.mkdir(exist_ok=True)
    def save(path, content):
        with tempfile.NamedTemporaryFile(dir=path.parent, mode="w", encoding="utf-8", delete=False) as f:
            f.write(content)
        os.replace(f.name, path)
    signature = hashlib.sha256(b"".join((ROOT/name).read_bytes() for name in (
        "rag.py", "corpus.py", "retrieval.py", "local_llm.py", "data/manifest.json"))).hexdigest()
    records = {m: [] for m in methods}
    for q in qs:
        question = q["question"]
        en = None
        if "C" in methods:
            key = hashlib.sha256((signature + model_sha + question).encode()).hexdigest()
            cached = run_cache / f"{key}.en"
            if not cached.exists():
                save(cached, translate(question))
            en = cached.read_text()
        ranks = search.rank(question, en)
        for m in methods:
            hits = [passages[int(i)] for i in ranks[m][:K]]
            row = record(q, m, hits, raw)
            key = hashlib.sha256(dump([signature, model_sha, question, row["retrieved"]]).encode()).hexdigest()
            cached = run_cache / f"{key}.json"
            if not cached.exists():
                save(cached, dump(answer(question, hits, raw)))
            row.update(json.loads(cached.read_text()), model_digest=model_sha, code_digest=signature)
            if m == "C":
                row["translation"] = en
            if args.cmd == "ask":
                print(json.dumps(row, ensure_ascii=False, indent=2))
            else:
                records[m].append(row)
                print(q["id"], m, row["status"], row["duration_s"], flush=True)
    if args.cmd == "run":
        RESULTS.mkdir(exist_ok=True)
        for m in methods:
            path = RESULTS / f"{args.split}-{m}.jsonl"
            save(path, "".join(dump(r) + "\n" for r in records[m]))


if __name__ == "__main__":
    main()
