"""Loopback-only browser interface for saved examples and optional live questions."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from time import monotonic

from .corpus import ROOT, load
from .run_review import read_index

ASSETS = Path(__file__).with_name("web")
EXAMPLES = {"test-01": "Запрет RC4", "dev-01": "Две редакции TLS", "dev-05": "Граница корпуса"}

# Russian display text for the frozen demo. Keep the recorded outputs and evidence intact.
EXAMPLE_TEXT = {
    "test-01": (
        "Если клиент предлагает только наборы шифров RC4, сервер обязан прервать рукопожатие TLS. "
        "Выбирать набор шифров RC4 запрещено.",
        "Сервер может отправить оповещение insufficient_security. Отправлять его необязательно.",
    ),
    "dev-01": (
        "В мае 2015 года реализациям не рекомендовалось согласовывать TLS 1.0. Исключение допускалось, "
        "если при согласовании была недоступна более новая версия.",
        "В марте 2021 года исключение отменили. Согласовывать TLS 1.0 и выполнять откат к этой версии запрещено.",
    ),
}


def public_answer(row):
    return {key: row[key] for key in ("question", "status", "claims", "total_s")}


class Chat:
    def __init__(self, generate=None):
        rows = read_index(ROOT / "results/all_direct/answers.jsonl")
        self.examples = [dict(title=title, **public_answer(rows[key])) for key, title in EXAMPLES.items()]
        for key, example in zip(EXAMPLES, self.examples):
            if key in EXAMPLE_TEXT:
                example["claims"] = [dict(claim, text=text) for claim, text in
                                     zip(example["claims"], EXAMPLE_TEXT[key], strict=True)]
        self.lock = Lock()
        self.generate = generate

    def ask(self, question):
        if not isinstance(question, str) or not 1 <= len(question.strip()) <= 2000:
            return 400, {"error": "Введите вопрос длиной от 1 до 2000 символов."}
        if not self.lock.acquire(blocking=False):
            return 409, {"error": "Дождитесь завершения текущего запроса."}
        try:
            if self.generate is None:
                return 403, {"error": "Включён режим сохранённых примеров."}
            start = monotonic()
            answer = self.generate(question.strip())
            if answer["status"] == "generation_failed":
                return 502, {"error": "Не удалось получить ответ модели. Проверьте соединение и доступ к модели."}
            return 200, public_answer(dict(answer, total_s=round(monotonic() - start, 2)))
        except (OSError, ValueError, RuntimeError):
            return 502, {"error": "Ошибка обработки вопроса. Проверьте доступ к модели."}
        finally:
            self.lock.release()


class ChatHandler(BaseHTTPRequestHandler):
    def send_body(self, status, content, mime):
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; frame-ancestors 'none'; object-src 'none'")
        self.end_headers()
        self.wfile.write(content)

    def send_json(self, status, data):
        self.send_body(status, json.dumps(data, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def local_request(self):
        port = self.server.server_port
        return self.headers.get("Host") in (f"127.0.0.1:{port}", f"localhost:{port}")

    def do_GET(self):
        if not self.local_request():
            return self.send_error(403)
        if self.path == "/api/examples":
            chat = self.server.chat
            return self.send_json(200, {"examples": chat.examples, "live": chat.generate is not None})
        pages = {"/": ("index.html", "text/html"), "/chat.css": ("chat.css", "text/css"),
                 "/chat.js": ("chat.js", "text/javascript")}
        if self.path not in pages:
            return self.send_error(404)
        name, mime = pages[self.path]
        self.send_body(200, (ASSETS / name).read_bytes(), mime + "; charset=utf-8")

    def do_POST(self):
        if not self.local_request() or self.headers.get("Origin") != "http://" + self.headers.get("Host", ""):
            return self.send_error(403)
        if self.path != "/api/ask":
            return self.send_error(404)
        if self.headers.get_content_type() != "application/json":
            return self.send_error(415)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 16384:
                return self.send_error(413)
            payload = json.loads(self.rfile.read(length))
        except (UnicodeDecodeError, ValueError):
            return self.send_error(400)
        chat = self.server.chat
        status, result = chat.ask(payload.get("question") if isinstance(payload, dict) else None)
        self.send_json(status, result)


def serve(port, *, offline=False):
    generate = None
    if not offline:
        from .llm_client import api_key
        from .pipeline import run
        from .retrieval import Search
        api_key()
        raw, passages = load()
        search = Search(passages)

        def generate(question):
            return run(search, passages, raw, question, "direct")

    with ThreadingHTTPServer(("127.0.0.1", port), ChatHandler) as server:
        server.chat = Chat(generate)
        print(f"http://127.0.0.1:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
