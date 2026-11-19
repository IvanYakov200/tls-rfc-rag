"""Stable evidence coordinates in the frozen RFC Editor text files."""
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CACHE = ROOT / ".runtime"
HEADING = re.compile(r"^((?:[0-9]+(?:\.[0-9]+)*)|(?:Appendix [A-Z])|(?:[A-Z](?:\.[0-9]+)*))\.\s+(\S.*)")
FOOTER = re.compile(r"\[Page \d+\]\s*$|^RFC \d+\s{2,}")


@dataclass(frozen=True)
class Passage:
    rfc: int
    section: str
    heading: str
    start: int
    end: int
    text: str

    @property
    def key(self):
        return f"{self.rfc}:{self.start}-{self.end}"

    @property
    def document(self):
        return f"RFC {self.rfc} §{self.section} {self.heading}\n{self.text}"


def read_corpus():
    records = json.loads((DATA / "manifest.json").read_text())["documents"]
    raw = {}
    for record in records:
        blob = (DATA / record["path"]).read_bytes()
        if hashlib.sha256(blob).hexdigest() != record["sha256"]:
            raise ValueError(f"Checksum mismatch in {record['path']}")
        raw[record["rfc"]] = blob
    return raw


def passages(rfc, blob, *, paragraphs=False):
    section = heading = ""
    start = end = offset = words = 0
    blank = False

    def emit():
        raw = blob[start:end].decode()
        return Passage(rfc, section, heading, start, end, " ".join(raw.split()))

    for line in blob.split(b"\n"):
        pos, offset = offset, offset + len(line) + 1
        s = line.decode().rstrip()
        match = HEADING.match(s)
        boundary = s in {"Abstract", "Table of Contents", "Acknowledgements",
                         "Copyright Notice", "Status of This Memo"} or match
        if boundary or FOOTER.search(s) or "\f" in s:
            if words:
                yield emit()
                words = 0
            if boundary:
                if match:
                    section, heading = match.groups()
                else:
                    section = s if s == "Abstract" else ""
                    heading = ""
            continue
        if not s.strip():
            blank = True
            continue
        if not section:
            continue
        if words and (blank if paragraphs else ((blank and words >= 67) or words >= 145)):
            yield emit()
            words = 0
        if not words:
            start = pos
        words += len(s.split())
        end = pos + len(line)
        blank = False
    if words:
        yield emit()


def load():
    raw = read_corpus()
    return raw, [p for rfc, blob in raw.items() for p in passages(rfc, blob)]


def questions(split):
    return [json.loads(line) for line in (DATA / f"{split}.jsonl").read_text().splitlines() if line]


def slices(raw, spans):
    return [raw[s["rfc"]][s["start_byte"]:s["end_byte"]].decode() for s in spans]
