"""Expand Russian questions into scoped English retrieval queries."""
import json
from itertools import chain, islice, zip_longest

from .llm_client import structured_completion

INSTRUCTION = """Write English search queries for the RFC corpus, one per requested fact or protocol version (at most three).
Use RFC terminology, expand the name of the operation, preserve field identifiers and distinctions such as sending versus receiving.
Focus on locating the relevant rule, without guessing its answer. Provide the applicable RFC number for each query when explicit
or identifiable from the named protocol version; use 0 for corpus-wide search. For a comparison use one query per side."""


def rewrite(question):
    schema = {"type": "object", "additionalProperties": False, "required": ["queries"], "properties": {
        "queries": {"type": "array", "minItems": 1, "maxItems": 3, "items": {
            "type": "object", "additionalProperties": False, "required": ["rfc", "text"], "properties": {
                "rfc": {"type": "integer"}, "text": {"type": "string"}}}}}}
    response = structured_completion([{"role": "system", "content": INSTRUCTION},
                                      {"role": "user", "content": question}], schema, 400)
    queries = json.loads(response["message"]["content"])["queries"]
    if response["done_reason"] != "stop" or not isinstance(queries, list) or not 1 <= len(queries) <= 3:
        raise ValueError("Incomplete retrieval queries")
    for q in queries:
        if not isinstance(q, dict) or type(q["rfc"]) is not int or not isinstance(q["text"], str) or not q["text"].strip():
            raise ValueError("Invalid retrieval query")
    return queries


def merge_queries(search, passages, baseline, queries, limit):
    lists = []
    for q in queries:
        scope = [i for i, p in enumerate(passages) if p.rfc == q["rfc"]]
        lists.append(search.rank(q["text"], candidates=scope or None)["B"])
    # Keep half the slots for the original ranking, then interleave English queries.
    english = (i for group in zip_longest(*lists) for i in group if i is not None)
    return list(islice(dict.fromkeys(chain(baseline[:limit // 2], english, baseline)), limit))
