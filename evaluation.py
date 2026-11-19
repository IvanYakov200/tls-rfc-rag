"""Byte-range evidence coverage and validation of the frozen experiment."""
import json
from statistics import mean
from corpus import ROOT, load, questions
from generator import evidence_context

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


def read_rows(name):
    return [json.loads(line) for line in (ROOT / "results" / name).read_text().splitlines()]


def recall(question, hits, raw):
    spans = [(p.rfc, p.start, p.end) for p in hits]
    return max((coverage(group, spans, raw) for group in question["gold_evidence_sets"]), default=None)


def metrics():
    raw, passages = load()
    by_key = {p.key: p for p in passages}
    qs = {q["id"]: q for split in ("test", "holdout") for q in questions(split)}
    grades = {g["id"]: g for g in read_rows("review.jsonl")}
    rows = read_rows("final.jsonl")
    assert len(qs) == len(rows) == len(grades) and set(qs) == set(grades) == {r["id"] for r in rows}
    scores = []
    correct = strict = claims = unsupported = citations = refusals = 0
    for row in rows:
        q = qs[row["id"]]
        grade = grades[q["id"]]
        assert row["question"] == q["question"]
        assert len(set(row["pool"])) == 40 and len(set(row["retrieved"])) == 5
        assert set(row["retrieved"]) <= set(row["pool"]) <= by_key.keys()
        hits = [by_key[key] for key in row["retrieved"]]
        score = recall(q, hits, raw)
        assert score == row["recall_at_5"]
        if score is not None:
            scores.append(score)
        _, excerpts = evidence_context(hits, raw)
        assert row["status"] == ("answered" if row["claims"] else "insufficient")
        for claim in row["claims"]:
            ids = claim["evidence_ids"]
            assert ids and len(set(ids)) == len(ids) and all(1 <= n <= len(excerpts) for n in ids)
            assert claim["citations"] == [excerpts[n - 1] for n in ids]
            citations += len(ids)
        assert all(1 <= n <= len(row["claims"]) for n in grade["unsupported"])
        correct += grade["correct"]
        strict += grade["correct"] and not grade["unsupported"]
        claims += len(row["claims"])
        unsupported += len(grade["unsupported"])
        refusals += not q["answerable"] and row["status"] == "insufficient"
    comparisons = read_rows("ablations.jsonl")
    for method in ("hybrid", "scoped", "reranked"):
        subset = [r for r in comparisons if r["method"] == method]
        assert {r["id"] for r in subset} == {q for q in qs if q.startswith("test-")} and len(subset) == 20
        for r in subset:
            _, excerpts = evidence_context([by_key[key] for key in r["retrieved"]], raw)
            for claim in r["claims"]:
                assert {k: v for k, v in claim.items() if k != "text"} in excerpts
        print(f"{method:8} test: correct {sum(r['correct'] for r in subset)}/20, strict {sum(r['strict'] for r in subset)}/20")
    for method in ("dense", "hybrid", "scoped"):
        subset = [r for r in read_rows("retrieval.jsonl") if r["method"] == method]
        assert len(subset) == 20
        values = [recall(qs[r["id"]], [by_key[k] for k in r["retrieved"]], raw) for r in subset]
        print(f"{method:8} Recall@5 (test): {mean(v for v in values if v is not None):.1%}")
    print(f"final    Recall@5 (test + holdout): {mean(scores):.1%}, complete {scores.count(1)}/{len(scores)}")
    print(f"Accuracy {correct}/{len(rows)}, strict {strict}/{len(rows)}, correct refusals {refusals}")
    print(f"Supported claims {claims-unsupported}/{claims}, unsupported {unsupported}/{claims}, verified quotes {citations}")
