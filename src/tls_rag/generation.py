"""Structured passage selection and cited answer generation."""
import json
import re
from http.client import HTTPException
from time import monotonic

from .llm_client import structured_completion

K = 5
POOL_SIZE = 40
INSTRUCTION = """Select exactly five distinct numbered passages as evidence for the user's question.
First identify each fact requested and the relevant RFC/version. Choose passages that explicitly support
each part, then conditions, exceptions, numeric values and normative levels. Prefer complementary
evidence to redundant near-matches. If evidence is incomplete choose the best available passages.
Use only candidate indices; return them in priority order. Do not answer the question."""


def select(question, candidates):
    texts = "\n\n".join(f"[{i}] {p.document}" for i, p in enumerate(candidates, 1))
    allowed = list(range(1, len(candidates) + 1))
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["ids"],
        "properties": {
            "ids": {
                "type": "array", "minItems": K, "maxItems": K,
                "items": {"type": "integer", "enum": allowed},
            }
        },
    }
    messages = [
        {"role": "system", "content": INSTRUCTION},
        {"role": "user", "content": question + "\n\n" + texts},
    ]
    response = structured_completion(messages, schema, 600)
    ids = json.loads(response["message"]["content"])["ids"]
    if response["done_reason"] != "stop" or not isinstance(ids, list) or len(ids) != K:
        raise ValueError("Incomplete passage selection")
    if not all(type(i) is int and i in allowed for i in ids) or len(set(ids)) != K:
        raise ValueError(f"Invalid passage indices {ids}")
    return [candidates[i - 1] for i in ids], response


INTRO = "Ответь на русском по предоставленным выдержкам из RFC. Для сравнения редакций нужны доказательства для обеих сторон. Если на существенную часть вопроса нельзя ответить, верни пустой список claims. Иначе разбей ответ на проверяемые утверждения."
SINGLE_CITATION = "Для каждого утверждения выбери один номер evidence (число в квадратных скобках), который его подтверждает. Для разных источников создай отдельные утверждения."
MULTIPLE_CITATIONS = "Для каждого утверждения укажи в evidence_ids номера выдержек, которые вместе подтверждают все его части. Выбирай только необходимые выдержки, без повторяющихся номеров."
NO_REPEAT = "Не повторяй уже подтверждённые утверждения в итоговой сводке."
RUSSIAN_STYLE = (
    " В тексте каждого утверждения передавай силу требования естественными русскими словами:"
    " MUST — обязан, MUST NOT — запрещено, SHOULD — рекомендуется, SHOULD NOT — не рекомендуется,"
    " MAY — может или разрешается. Согласуй слова по роду и числу."
    " Не дублируй английское ключевое слово в скобках. Переводи и обычные технические термины:"
    " handshake — рукопожатие, alert — оповещение, cipher suite — набор шифров."
    " Имена сообщений, полей, алгоритмов и коды оповещений сохраняй точно:"
    " например, сообщение ClientHello или оповещение insufficient_security."
)
ENDING = "Не копируй цитату. Пиши кратко. Не добавляй сведений из памяти." + RUSSIAN_STYLE


def generation_prompt(multiple, no_repeat):
    parts = [INTRO, MULTIPLE_CITATIONS if multiple else SINGLE_CITATION]
    if no_repeat:
        parts.append(NO_REPEAT)
    return " ".join([*parts, ENDING])


DIRECT_PROMPT = (
    "Ты отвечаешь по RFC на русском. Сначала для каждой части вопроса найди точное правило в выдержках."
    " Сформулируй его как короткий прямой ответ. Верни claims: один элемент на каждый запрошенный факт"
    " или сторону сравнения. В evidence_ids перечисли номера всех выдержек, из которых следует именно"
    " этот текст. Переводи границы требования буквально: кто, при каком условии,"
    " что именно разрешено или запрещено. Включи необходимые оговорки источника."
    " Для вопроса о значении дай значение; для вопроса о правиле дай правило."
    " Закончи сразу после ответа на последнюю часть. Подробности реализации и повторные выводы опусти."
    " Если существенная часть не подтверждается выдержками, верни пустой claims."
    + RUSSIAN_STYLE
)

SYSTEM = generation_prompt(False, False)
MULTI_PROMPT = generation_prompt(True, True)


def answer_schema(count, multiple=False):
    claim = {
        "type": "object",
        "additionalProperties": False,
        "required": ["text", "evidence"],
        "properties": {
            "text": {"type": "string"},
            "evidence": {"type": "integer", "enum": list(range(1, count + 1))},
        },
    }
    if multiple:
        index = claim["properties"].pop("evidence")
        claim["required"] = ["text", "evidence_ids"]
        claim["properties"]["evidence_ids"] = {"type": "array", "minItems": 1, "items": index}
    claims = {"type": "array", "items": claim} if count else {"const": []}
    return {"type": "object", "additionalProperties": False, "required": ["claims"], "properties": {"claims": claims}}


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


def answer(question, hits, raw, *, transport=structured_completion, system=None, multiple=False):
    if system is None:
        system = MULTI_PROMPT if multiple else SYSTEM
    context, excerpts = evidence_context(hits, raw)
    attempts = []
    start = monotonic()
    output = ""
    citations = []
    status = "generation_failed"
    error = None
    for limit in (600, 900):
        attempt = {"num_predict": limit, "done_reason": None, "prompt_eval_count": None,
                   "eval_count": None}
        attempts.append(attempt)
        output = ""
        try:
            messages = [{"role": "system", "content": system},
                        {"role": "user", "content": question + "\n\n" + context}]
            res = transport(messages, answer_schema(len(excerpts), multiple), limit)
            for key in ("done_reason", "prompt_eval_count", "eval_count"):
                attempt[key] = res.get(key)
            output = res["message"]["content"]
            if attempt["done_reason"] == "length":
                attempt["raw_output"] = output
                error = "output_limit"
                continue
            if res.get("done") is False or attempt["done_reason"] not in (None, "stop"):
                raise ValueError("unexpected completion state")
            parsed = json.loads(output)
            claims = parsed["claims"]
            if not isinstance(claims, list):
                raise ValueError("invalid claims")
            for claim in claims:
                ids = claim["evidence_ids"] if multiple else [claim["evidence"]]
                if not isinstance(ids, list) or not ids or not isinstance(claim["text"], str) or not claim["text"].strip():
                    raise ValueError("missing evidence or claim")
                if any(type(n) is not int or not 1 <= n <= len(excerpts) for n in ids) or len(ids) != len(set(ids)):
                    raise ValueError("invalid evidence or claim")
                refs = [excerpts[n - 1] for n in ids]
                if multiple:
                    citations.append({"text": claim["text"], "evidence_ids": ids, "citations": refs})
                else:
                    citations.append({"text": claim["text"], **refs[0]})
            status = "answered" if claims else "insufficient"
            error = None
        except (OSError, HTTPException) as exc:
            error = f"request failed: {exc}"
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            error = f"invalid response: {exc}"
            citations = []
        break
    return {"status": status, "claims": citations, "raw_output": output, "error": error,
            "attempts": attempts, "duration_s": round(monotonic() - start, 2)}
