"""Grade answers against references and each claim against its own citations."""
import json

from .llm_client import structured_completion
from .run_review import fingerprint

JUDGE = "cx/gpt-6.2-sol"
RUBRIC = """Оцени ответ на вопрос по RFC. correct=true только если все существенные части
вопроса получили верный ответ без смысловой ошибки. Допускай эквивалентные формулировки.
Если answerable=false, зачти только статус insufficient; при answerable=true отказ ошибочен.
Отдельно перечисли в unsupported номера утверждений (с 1), для которых совокупность
ИХ СОБСТВЕННЫХ цитат не подтверждает хотя бы один существенный факт, условие, версию
или силу требования. Эталон помогает проверить содержание, но не заменяет цитату.
Для пустого списка утверждений unsupported пуст. Кратко обоснуй оценку по-русски."""
SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["correct", "unsupported", "note"],
    "properties": {"correct": {"type": "boolean"},
                   "unsupported": {"type": "array", "items": {"type": "integer"}},
                   "note": {"type": "string"}}}


def judge(q, row, raw):
    payload = {k: q[k] for k in ("question", "reference_answer", "answerable")}
    payload["reference_sources"] = [
        [dict(rfc=s["rfc"], section=s["section"],
              quote=raw[s["rfc"]][s["start_byte"]:s["end_byte"]].decode()) for s in group]
        for group in q["gold_evidence_sets"]]
    payload["status"] = row["status"]
    payload["claims"] = [{k: c[k] for k in ("text", "citations")} for c in row["claims"]]
    response = structured_completion([{"role": "system", "content": RUBRIC},
                                      {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                                     SCHEMA, 1600, model=JUDGE)
    if response["done_reason"] != "stop":
        raise ValueError("Judge response incomplete")
    return dict(id=q["id"], answer_sha256=fingerprint(row), judge=JUDGE,
                **json.loads(response["message"]["content"]))
