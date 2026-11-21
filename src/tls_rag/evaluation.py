"""Byte-range evidence coverage and validation of the frozen experiment."""
import json
from statistics import mean
from .corpus import ROOT, load, questions


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


def verify():
    from collections import defaultdict
    from .all_questions import dataset, report
    from .run_review import read_index, validate_answer, fingerprint

    raw, passages = load()
    by_key = {p.key: p for p in passages}
    root = ROOT / "results/all_direct"
    summary = list(report(dataset(), read_index(root / "answers.jsonl"),
                          read_index(root / "review.jsonl"), raw, passages))
    if summary != json.loads((root / "metrics.json").read_text()):
        raise ValueError("Saved metrics differ from recomputed metrics")
    for r in summary:
        print(f"{r['group']:10} correct {r['correct']}/{r['questions']}, strict {r['strict']}/{r['questions']}, "
              f"Recall@5 {r['evidence_recall_at_5']:.1%}, faithfulness {r['supported']}/{r['claims']}")

    qs = {q["id"]: q for split in ("test", "holdout", "challenge") for q in questions(split)}
    original = {q["id"]: q for q in questions("test-v1")}
    groups = defaultdict(list)
    for r in read_rows("comparisons.jsonl"):
        experiment = r["experiment"]
        q = (original if experiment == "test-v1" else qs)[r["id"]]
        value = validate_answer(dict(r, question=q["question"]), q, by_key, raw)
        if "unsupported" in r:
            assert r["strict"] == (r["correct"] and not r["unsupported"])
        groups[(experiment, r["method"])].append((r, value))
    expected = {"test-v1": {"dense", "hybrid", "scoped", "reranked", "single_no_repeat", "multi", "final"},
                "selection": {"multi", "faceted", "direct"}}
    assert set(groups) == {(group, m) for group, methods in expected.items() for m in methods}
    for (experiment, method), values in groups.items():
        ids = original.keys() if experiment == "test-v1" else qs.keys()
        assert {r["id"] for r, v in values} == ids and len(values) == len(ids)
        scores = [v for _, v in values if v is not None]
        print(f"{experiment} {method}: Recall@5 {mean(scores):.1%}, correct "
              f"{sum(r['correct'] for r, _ in values)}/{len(values)}, strict "
              f"{sum(r['strict'] for r, _ in values)}/{len(values)}")

    blind = {q["id"]: q for q in questions("blind")}
    first = read_rows("first_blind.jsonl")
    assert len(first) == len(blind) and {r["answer"]["id"] for r in first} == blind.keys()
    correct = strict = 0
    for pair in first:
        r, g = pair["answer"], pair["grade"]
        validate_answer(r, blind[r["id"]], by_key, raw)
        assert g["id"] == r["id"] and g["answer_sha256"] == fingerprint(r)
        correct += g["correct"]
        strict += g["correct"] and not g["unsupported"]
    print(f"First blind: correct {correct}/{len(first)}, strict {strict}/{len(first)}")
