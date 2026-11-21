"""Offline manual review of one complete generation run."""
import hashlib
import json
from pathlib import Path
from statistics import mean

from .corpus import load, questions
from .evaluation import recall
from .generation import K, evidence_context


def read_index(path):
    rows = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        key = row["id"]
        if key in rows:
            raise ValueError(f"{path}: duplicate ID {key}")
        rows[key] = row
    return rows


def fingerprint(row):
    data = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode()).hexdigest()


def validate_answer(row, q, by_key, raw):
    key = q["id"]
    if row["question"] != q["question"]:
        raise ValueError(f"{key}: question differs from dataset")
    claims = row["claims"]
    if not isinstance(claims, list) or row["status"] != ("answered" if claims else "insufficient"):
        raise ValueError(f"{key}: invalid status or technical failure; complete the run first")
    keys = row["retrieved"]
    if len(keys) != K or len(set(keys)) != K or not set(keys) <= by_key.keys():
        raise ValueError(f"{key}: expected {K} distinct corpus passages")
    hits = [by_key[k] for k in keys]
    _, excerpts = evidence_context(hits, raw)
    for number, claim in enumerate(claims, 1):
        valid = isinstance(claim.get("text"), str) and claim["text"].strip()
        if "citations" in claim or "evidence_ids" in claim:
            ids = claim.get("evidence_ids")
            valid = valid and isinstance(ids, list) and ids and all(
                type(i) is int and 1 <= i <= len(excerpts) for i in ids
            ) and len(set(ids)) == len(ids) and claim.get("citations") == [excerpts[i-1] for i in ids]
        else:
            valid = valid and {k: v for k, v in claim.items() if k != "text"} in excerpts
        if not valid:
            raise ValueError(f"{key}, claim {number}: invalid claim or citation")
    return recall(q, hits, raw)


def review_run(args):
    rows = read_index(args.answers)
    dataset = questions(args.split)
    expected = {q["id"] for q in dataset}
    if rows.keys() != expected:
        raise ValueError(f"answers differ from {args.split}: missing {sorted(expected - rows.keys())}, "
                         f"unexpected {sorted(rows.keys() - expected)}")
    raw, passages = load()
    by_key = {p.key: p for p in passages}
    scores = [validate_answer(rows[q["id"]], q, by_key, raw) for q in dataset]
    if args.command == "review-template":
        with Path(args.review).open("x", encoding="utf-8") as out:
            for q in dataset:
                row = rows[q["id"]]
                grade = dict(id=q["id"], answer_sha256=fingerprint(row), correct=None, unsupported=None)
                print(json.dumps(grade, ensure_ascii=False), file=out)
        return
    grades = read_index(args.review)
    if grades.keys() != expected:
        raise ValueError("review IDs must match the complete run")
    correct = strict = claims = unsupported = refusals = 0
    for q in dataset:
        key = q["id"]
        row, grade = rows[key], grades[key]
        if grade.get("answer_sha256") != fingerprint(row):
            raise ValueError(f"{key}: review belongs to another answer")
        ok, bad = grade.get("correct"), grade.get("unsupported")
        count = len(row["claims"])
        if type(ok) is not bool or not isinstance(bad, list) or not all(
            type(i) is int and 1 <= i <= count for i in bad
        ) or len(set(bad)) != len(bad):
            raise ValueError(f"{key}: complete correct and unsupported with valid grades")
        refused = row["status"] == "insufficient"
        if (refused or not q["answerable"]) and ok != (refused and not q["answerable"]):
            raise ValueError(f"{key}: correctness inconsistent with refusal rubric")
        correct += ok
        strict += ok and not bad
        claims += count
        unsupported += len(bad)
        refusals += refused and not q["answerable"]
    recalls = [v for v in scores if v is not None]
    print(f"Evidence Recall@5: {mean(recalls):.1%} ({len(recalls)} answerable questions)")
    print(f"Accuracy: {correct}/{len(rows)}, strict: {strict}/{len(rows)}")
    faith = f"{(claims-unsupported)/claims:.1%}" if claims else "N/A"
    print(f"Faithfulness: {claims-unsupported}/{claims} ({faith})")
    print(f"Correct refusals: {refusals}/{sum(not q['answerable'] for q in dataset)}")
