"""Russian query, scoped hybrid recall, listwise selection and cited generation."""
import re
from collections import defaultdict
from http.client import HTTPException
from time import monotonic
from itertools import islice, zip_longest

from .faceted import merge_queries, rewrite
from .generation import DIRECT_PROMPT, K, POOL_SIZE, answer, select, generation_prompt


def scoped_ranking(search, passages, question, limit):
    baseline = search.rank(question)["B"][:limit]
    by_rfc = defaultdict(list)
    for i, passage in enumerate(passages):
        by_rfc[passage.rfc].append(i)
    refs = list(dict.fromkeys(int(n) for n in re.findall(r"\bRFC\s*(\d+)\b", question, re.I)))
    scoped = baseline
    if len(refs) >= 2 and all(r in by_rfc for r in refs):
        rankings = [search.rank(question, candidates=by_rfc[r])["B"] for r in refs]
        merged = (i for group in zip_longest(*rankings) for i in group if i is not None)
        scoped = list(islice(merged, limit))
        if len(scoped) < limit:
            # Small RFCs can exhaust the scoped pool. Preserve its order and
            # fill remaining slots from the original corpus-wide ranking.
            seen = set(scoped)
            scoped.extend(islice((i for i in baseline if i not in seen), limit - len(scoped)))
    return scoped


def run(search, passages, raw, question, method="direct"):
    if method in ("dense", "hybrid"):
        indices = search.rank(question)["A" if method == "dense" else "B"][:K]
    else:
        indices = scoped_ranking(search, passages, question, POOL_SIZE)
    diagnostics = {}
    if method in ("faceted", "direct"):
        try:
            queries = rewrite(question)
        except (OSError, HTTPException, ValueError, TypeError, KeyError, AttributeError) as exc:
            diagnostics["rewrite_error"] = str(exc)
        else:
            indices = merge_queries(search, passages, indices, queries, POOL_SIZE)
            diagnostics["queries"] = queries
    return {**finish(question, [passages[i] for i in indices], raw, method), **diagnostics}


def finish(question, pool, raw, method):
    row = {"question": question, "method": method, "pool": [p.key for p in pool]}
    if method in ("final", "single_no_repeat", "multi", "reranked", "faceted", "direct"):
        started = monotonic()
        try:
            hits, _ = select(question, pool)
        except (OSError, HTTPException, ValueError, TypeError, KeyError, AttributeError, RuntimeError) as exc:
            return {**row, "retrieved": [], "status": "generation_failed",
                    "error_stage": "selection", "error": f"{type(exc).__name__}: {exc}",
                    "claims": [], "raw_output": "", "attempts": [],
                    "duration_s": round(monotonic() - started, 2)}
    else:
        hits = pool[:K]
    if method == "direct":
        result = answer(question, hits, raw, multiple=True, system=DIRECT_PROMPT)
    elif method in ("multi", "faceted"):
        result = answer(question, hits, raw, multiple=True, system=generation_prompt(True, False))
    elif method == "single_no_repeat":
        result = answer(question, hits, raw, system=generation_prompt(False, True))
    else:
        result = answer(question, hits, raw, multiple=method == "final")
    if result["status"] == "generation_failed" and "error_stage" not in result:
        result["error_stage"] = "answer"
    return {**row, "retrieved": [p.key for p in hits], **result}
