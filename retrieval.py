"""Cross-lingual dense baseline and a one-variable lexical ablation."""
import hashlib
import json
import math
import os
import re
import tempfile
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from corpus import CACHE

REV = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
MODEL = "model_qint8_avx512_vnni.onnx"
ASSETS = {MODEL: "dd476dd0c2514e9b9be83aeb3853fac0763e0bdf4a71645407587d77c48a2d88",
          "tokenizer.json": "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39"}
ASSET_DIR = CACHE / "e5"


def check_assets(download=False):
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    for name, expected in ASSETS.items():
        target = ASSET_DIR / name
        if not target.exists() and download:
            url = f"https://huggingface.co/intfloat/multilingual-e5-small/resolve/{REV}/onnx/{name}"
            with tempfile.NamedTemporaryFile(dir=ASSET_DIR, delete=False) as f:
                temp = Path(f.name)
            try:
                urllib.request.urlretrieve(url, temp)
                if hashlib.sha256(temp.read_bytes()).hexdigest() != expected:
                    raise ValueError(f"Download checksum mismatch for {name}")
                temp.replace(target)
            finally:
                temp.unlink(missing_ok=True)
        if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Missing or damaged E5 asset {target}. Run the setup command.")


class Encoder:
    def __init__(self):
        check_assets()
        self.tokenizer = Tokenizer.from_file(str(ASSET_DIR / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=256)
        self.tokenizer.enable_padding(pad_id=1, pad_token="<pad>")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 3
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(ASSET_DIR / MODEL), opts, providers=["CPUExecutionProvider"])

    def encode(self, texts):
        enc = self.tokenizer.encode_batch(texts)
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        hidden = self.session.run(None, {"input_ids": ids, "attention_mask": mask,
                                        "token_type_ids": np.zeros_like(ids)})[0]
        pooled = (hidden * mask[:, :, None]).sum(axis=1) / mask.sum(axis=1)[:, None]
        return (pooled / np.linalg.norm(pooled, axis=1, keepdims=True)).astype(np.float32)


def dense_index(encoder, passages):
    signature = hashlib.sha256(json.dumps(
        [REV, ASSETS[MODEL], "mean-v1-256", [(p.key, p.document) for p in passages]],
        ensure_ascii=False).encode()).hexdigest()[:20]
    target = CACHE / f"dense-{signature}.npy"
    if target.exists():
        return np.load(target, allow_pickle=False)
    order = sorted(range(len(passages)), key=lambda i: len(passages[i].document))
    vectors = np.empty((len(order), 384), dtype=np.float32)
    for i in range(0, len(order), 12):
        batch = order[i:i + 12]
        vectors[batch] = encoder.encode(["passage: " + passages[j].document for j in batch])
    with tempfile.NamedTemporaryFile(dir=CACHE, suffix=".npy", delete=False) as f:
        np.save(f, vectors)
    os.replace(f.name, target)
    return vectors


def tokens(s):
    return re.findall(r"[\w]+(?:[.-][\w]+)*", s.lower())


class Search:
    def __init__(self, passages):
        self.encoder = Encoder()
        self.matrix = dense_index(self.encoder, passages)
        self.counts = [Counter(tokens(p.document)) for p in passages]
        df = Counter(t for d in self.counts for t in d)
        n = len(passages)
        self.idf = {t: math.log1p((n - f + .5) / (f + .5)) for t, f in df.items()}
        dl = np.array([sum(d.values()) for d in self.counts])
        self.norm = 1.2 * (.25 + .75 * dl / dl.mean())

    def rank(self, question, translated=None):
        vec = self.encoder.encode(["query: " + question])[0]
        dense = np.argsort(-(self.matrix @ vec), kind="stable")
        def fuse(query):
            hits = np.zeros(len(dense))
            for term in set(tokens(query)):
                if term in self.idf:
                    freq = np.array([c[term] for c in self.counts])
                    hits += self.idf[term] * freq * 2.2 / (freq + self.norm)
            lexical = np.argsort(-hits, kind="stable")
            scores = Counter()
            for ranking in (dense[:40], [i for i in lexical[:40] if hits[i] > 0]):
                for rank, i in enumerate(ranking, 1):
                    scores[int(i)] += 1 / (60 + rank)
            return sorted(scores, key=scores.get, reverse=True)
        result = {"A": dense, "B": fuse(question)}
        if translated:
            result["C"] = fuse(translated)
        return result
