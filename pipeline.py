"""Russian query, scoped hybrid recall, listwise selection and cited generation."""
import re
from collections import defaultdict
from itertools import islice, zip_longest

from generator import K, POOL_SIZE, answer, select


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


def run(search, passages, raw, question, method="final"):
    if method in ("dense", "hybrid"):
        indices = search.rank(question)["A" if method == "dense" else "B"][:K]
    else:
        indices = scoped_ranking(search, passages, question, POOL_SIZE)
    pool = [passages[i] for i in indices]
    if method in ("final", "reranked"):
        hits, _ = select(question, pool)
    else:
        hits = pool[:K]
    result = answer(question, hits, raw, multiple=method == "final")
    return {"question": question, "method": method, "pool": [p.key for p in pool],
            "retrieved": [p.key for p in hits], **result}
