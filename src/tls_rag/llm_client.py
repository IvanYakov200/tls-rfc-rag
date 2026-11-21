"""AnyModel API credentials and structured streaming transport."""
import json
import os
import subprocess

from .corpus import ROOT

MODEL = "cx/gpt-6.1-sol"
URL = "https://anymodel.org/v1/chat/completions"


def structured_completion(messages, schema, limit, *, model=None):
    body = {"model": model or MODEL, "messages": messages, "stream": True,
            "max_completion_tokens": limit,
            "stream_options": {"include_usage": True},
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "cited_answer", "strict": True, "schema": schema}}}
    # Keep the bearer token off the command line. curl reads headers from stdin.
    response = subprocess.run(
        ["curl", "--silent", "--show-error", "--max-time", "240", "--header", "@-",
         "--header", "Content-Type: application/json", "--data-binary", json.dumps(body),
         "--write-out", "\n%{http_code}", URL],
        input="Authorization: Bearer " + api_key() + "\n",
        text=True, capture_output=True)
    if response.returncode:
        raise OSError(f"curl exited with code {response.returncode}")
    stream, _, status = response.stdout.rpartition("\n")
    if status != "200":
        raise OSError(f"AnyModel HTTP {status}")
    pieces, reason, usage = [], None, {}
    for line in stream.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        event = json.loads(data)
        if "error" in event:
            raise ValueError("Gateway returned an error during streaming")
        usage = event.get("usage") or usage
        for choice in event.get("choices", []):
            pieces.append(choice.get("delta", {}).get("content") or "")
            reason = choice.get("finish_reason") or reason
    if reason is None:
        raise ValueError("Missing stream finish_reason")
    return {"message": {"content": "".join(pieces)}, "done_reason": reason,
            "prompt_eval_count": usage.get("prompt_tokens"), "eval_count": usage.get("completion_tokens")}


def api_key():
    key = os.getenv("ANYMODEL_API_KEY")
    if not key:
        local = ROOT / ".anymodel-key"
        if local.exists():
            key = local.read_text().strip()
    if not key:
        raise ValueError("Set ANYMODEL_API_KEY before running a live request")
    return key
