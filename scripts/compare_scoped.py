"""Offline retrieval diagnostic. Run from the repo root as a Python module."""
import json
import re
from collections import defaultdict
from itertools import islice, zip_longest

from corpus import ROOT, load, questions
from rag import K, record
from retrieval import Search


def compare():
    raw, passages = load()
    search = Search(passages)
    by_rfc = defaultdict(list)
    for i, passage in enumerate(passages):
        by_rfc[passage.rfc].append(i)
    saved = {r["id"]: r for r in map(json.loads, (ROOT / "results/test-B.jsonl").read_text().splitlines())}
    for q in questions("test"):
        question = q["question"]
        baseline = search.rank(question)["B"][:K]
        if [passages[i].key for i in baseline] != saved[q["id"]]["retrieved"]:
            raise ValueError(f"Baseline differs from saved experiment for {q['id']}")
        refs = list(dict.fromkeys(int(n) for n in re.findall(r"\bRFC\s*(\d+)\b", question, re.I)))
        scoped = baseline
        if len(refs) >= 2 and all(r in by_rfc for r in refs):
            rankings = [search.rank(question, candidates=by_rfc[r])["B"] for r in refs]
            merged = (i for group in zip_longest(*rankings) for i in group if i is not None)
            scoped = list(islice(merged, K))
        row = record(q, "scoped", [passages[i] for i in scoped], raw)
        yield {**row, "category": q["category"], "baseline_recall_at_5": saved[q["id"]]["recall_at_5"]}


if __name__ == "__main__":
    for row in compare():
        print(json.dumps(row, ensure_ascii=False))
