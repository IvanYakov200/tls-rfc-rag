"""Measure direct on every distinct question and report scores from saved answers."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from math import ceil
from pathlib import Path
from statistics import mean, median
from time import monotonic

from .judge import judge
from .corpus import ROOT, load, questions
from .evaluation import ranking_at_5, recall
from .run_review import fingerprint, read_index, validate_answer

SPLITS = ("dev", "test", "holdout", "challenge", "blind", "test-v1")


def dataset():
    seen = set()
    rows = []
    for split in SPLITS:
        for q in questions(split):
            if q["question"] in seen:
                continue
            seen.add(q["question"])
            key = "legacy-" + q["id"] if split == "test-v1" else q["id"]
            rows.append(dict(q, id=key, split=split))
    return rows


def collect(path, qs, operation):
    done = read_index(path) if path.exists() else {}
    pending = [q for q in qs if q["id"] not in done]
    def attempt(q):
        try:
            return operation(q)
        except (OSError, ValueError) as exc:
            print(q["id"], str(exc), flush=True)
            return None
    with path.open("a", encoding="utf-8") as out, ThreadPoolExecutor(6) as pool:
        for row in pool.map(attempt, pending):
            if row is not None:
                print(json.dumps(row, ensure_ascii=False), file=out, flush=True)


def report(qs, rows, grades, raw, passages):
    by_key = {p.key: p for p in passages}
    assert set(rows) == {q["id"] for q in qs}
    assert set(grades) == {k for k, r in rows.items() if r["status"] != "generation_failed"}
    ranking = {q["id"]: ranking_at_5(q, rows[q["id"]]["retrieved"], passages, raw)
               for q in qs if q["answerable"]}
    for group in (*SPLITS, "current", "all"):
        subset = [q for q in qs if group == "all" or q["split"] == group or
                  (group == "current" and q["split"] != "test-v1")]
        recalls, pool_recalls, times = [], [], []
        citations = 0
        correct = strict = supported = claims = refusals = failed = 0
        for q in subset:
            row = rows[q["id"]]
            times.append(row["total_s"])
            assert row["method"] == "direct" and row["question"] == q["question"]
            assert len(row["pool"]) == len(set(row["pool"])) == 40
            assert set(row["retrieved"]) <= set(row["pool"])
            if q["answerable"]:
                pool_recalls.append(recall(q, [by_key[k] for k in row["pool"]], raw))
            if row["status"] == "generation_failed":
                failed += 1
                value = recall(q, [by_key[k] for k in row["retrieved"]], raw)
                if value is not None:
                    recalls.append(value)
                continue
            score = validate_answer(row, q, by_key, raw)
            if score is not None:
                recalls.append(score)
            citations += sum(len(c["citations"]) for c in row["claims"])
            g = grades[q["id"]]
            assert g["answer_sha256"] == fingerprint(row)
            n = len(row["claims"])
            bad = g["unsupported"]
            assert type(g["correct"]) is bool and isinstance(bad, list) and len(set(bad)) == len(bad)
            assert all(type(i) is int and 1 <= i <= n for i in bad)
            refused = row["status"] == "insufficient"
            if refused or not q["answerable"]:
                assert g["correct"] == (refused and not q["answerable"])
            correct += g["correct"]
            strict += g["correct"] and not bad
            claims += n
            supported += n - len(bad)
            refusals += refused and not q["answerable"]
        yield {
            "group": group,
            "questions": len(subset),
            "correct": correct,
            "strict": strict,
            "evidence_recall_at_5": mean(recalls),
            "ndcg_at_5": mean(ranking[q["id"]][0] for q in subset if q["answerable"]),
            "mrr_at_5": mean(ranking[q["id"]][1] for q in subset if q["answerable"]),
            "evidence_recall_at_40": mean(pool_recalls),
            "verified_citations": citations,
            "full_coverage": recalls.count(1),
            "answerable": len(recalls),
            "supported": supported,
            "claims": claims,
            "correct_refusals": refusals,
            "technical_failures": failed,
            "mean_s": round(mean(times), 2),
            "median_s": round(median(times), 2),
            "p95_s": sorted(times)[ceil(0.95 * len(times)) - 1],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "judge", "score"))
    parser.add_argument("--out", type=Path, default=ROOT / "results/all_direct")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    answers, review = args.out / "answers.jsonl", args.out / "review.jsonl"
    qs = dataset()
    raw, passages = load()
    if args.action == "run":
        from .pipeline import run
        from .retrieval import Search
        search = Search(passages)
        def generate(q):
            start = monotonic()
            row = run(search, passages, raw, q["question"], "direct")
            return dict(id=q["id"], total_s=round(monotonic() - start, 3), **row)
        collect(answers, qs, generate)
    else:
        rows = read_index(answers)
        if args.action == "judge":
            collect(review, [q for q in qs if q["id"] in rows and
                             rows[q["id"]]["status"] != "generation_failed"],
                    lambda q: judge(q, rows[q["id"]], raw))
        else:
            summary = list(report(qs, rows, read_index(review), raw, passages))
            print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
