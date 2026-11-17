"""Local generator with verifiable references to unmodified source bytes."""
import json
import os
import re
import urllib.request

MODEL = "qwen3:4b-instruct-2507-q4_K_M"
HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
if "://" not in HOST:
    HOST = "http://" + HOST
DIGEST = "0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0"
OPTIONS = {"temperature": 0, "seed": 42, "num_ctx": 4096, "num_predict": 300}
SYSTEM = """Ответь на русском по предоставленным выдержкам из RFC. Для сравнения редакций нужны доказательства для обеих сторон. Если на существенную часть вопроса нельзя ответить, верни пустой список claims. Иначе разбей ответ на проверяемые утверждения. Для каждого утверждения выбери один номер evidence (число в квадратных скобках), который его подтверждает. Для разных источников создай отдельные утверждения. Не копируй цитату. Пиши кратко. Сохраняй различие MUST и SHOULD. Не добавляй сведений из памяти."""
SCHEMA = {"type": "object", "required": ["claims"], "properties": {"claims": {"type": "array",
          "items": {"type": "object", "required": ["text", "evidence"], "properties": {
          "text": {"type": "string"}, "evidence": {"type": "integer"}}}}}}


def api(path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(HOST + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=240) as res:
        return json.load(res)


def digest():
    for model in api("/api/tags")["models"]:
        if model["name"].lower() == MODEL.lower():
            if model["digest"] != DIGEST:
                raise ValueError("Qwen registry digest differs from the reference run")
            return model["digest"]
    raise ValueError("Pull the pinned Qwen model with Ollama first.")


def evidence_context(hits, raw):
    excerpts, blocks = [], []
    boundary = rb'(?<=[.!?])\s+(?=[A-Z*(\["]|$)|\n[ \t]*\n|(?<=\|)\n'
    for p in hits:
        blob = raw[p.rfc][p.start:p.end]
        lines = [f"RFC {p.rfc} §{p.section} {p.heading}"]
        left = 0
        for sep in [*re.finditer(boundary, blob), None]:
            right = len(blob) if sep is None else sep.start()
            part = blob[left:right]
            if part.strip():
                start = p.start + left + len(part) - len(part.lstrip())
                end = p.start + left + len(part.rstrip())
                quote = raw[p.rfc][start:end].decode()
                excerpts.append(dict(rfc=p.rfc, section=p.section, start_byte=start, end_byte=end, quote=quote))
                lines.append(f"[{len(excerpts)}] " + " ".join(quote.split()))
            left = right if sep is None else sep.end()
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks), excerpts


def answer(question, hits, raw):
    context, excerpts = evidence_context(hits, raw)
    res = api("/api/chat", {"model": MODEL, "stream": False, "format": SCHEMA, "options": OPTIONS,
              "messages": [{"role": "system", "content": SYSTEM},
                           {"role": "user", "content": question + "\n\n" + context}]})
    output = res["message"]["content"]
    citations = []
    status = "answered"
    try:
        if res.get("done_reason") == "length":
            raise ValueError("truncated")
        claims = json.loads(output)["claims"]
        if not isinstance(claims, list):
            raise ValueError("invalid claims")
        if not claims:
            status = "insufficient"
        for claim in claims:
            n = claim["evidence"]
            if type(n) is not int or not 1 <= n <= len(excerpts) or not claim["text"].strip():
                raise ValueError("invalid evidence or claim")
            citations.append({"text": claim["text"], **excerpts[n - 1]})
    except (KeyError, TypeError, ValueError, AttributeError):
        citations, status = [], "invalid_evidence"
    return {"status": status, "claims": citations, "raw_output": output,
            "duration_s": round(res["total_duration"] / 1e9, 2)}


def translate(question):
    res = api("/api/chat", {"model": MODEL, "stream": False, "options": OPTIONS,
              "messages": [{"role": "system", "content": "Translate the question to English for document search. Preserve all RFC numbers, identifiers, dates, negation, and version numbers. Return only the translation."},
                           {"role": "user", "content": question}]})
    return res["message"]["content"].strip()
