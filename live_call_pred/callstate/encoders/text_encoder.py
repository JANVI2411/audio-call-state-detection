"""
Pluggable text embedding branch for the partial ASR transcript.

Default `HashedNGramEncoder` is deterministic, dependency-free and instant —
a signed hashing trick over word unigrams/bigrams. It gives the fusion model
a usable lexical signal and makes tests reproducible on any machine.

`MiniLMEncoder` (sentence-transformers all-MiniLM-L6-v2) is the upgrade when
semantic generalisation matters: it will place "let me get you over to
billing" near "I'll connect you with a specialist" even though they share no
content words, which a hashing encoder cannot do. Opt in with
`--text-encoder minilm`.
"""
from __future__ import annotations

import hashlib
import re
from typing import List

import numpy as np

_WORD = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> List[str]:
    return _WORD.findall(text.lower())


class TextEncoder:
    dim: int

    def encode(self, text: str) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class HashedNGramEncoder(TextEncoder):
    def __init__(self, dim: int = 64):
        self.dim = dim

    def _h(self, s: str) -> int:
        return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little")

    def encode(self, text: str) -> np.ndarray:
        toks = tokenize(text)
        v = np.zeros(self.dim, dtype=np.float32)
        if not toks:
            return v
        grams = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        for g in grams:
            h = self._h(g)
            v[h % self.dim] += 1.0 if (h >> 32) & 1 else -1.0
        n = np.linalg.norm(v)
        return v / n if n > 0 else v


class MiniLMEncoder(TextEncoder):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def encode(self, text: str) -> np.ndarray:
        if not text.strip():
            return np.zeros(self.dim, dtype=np.float32)
        return self.model.encode(text, normalize_embeddings=True).astype(np.float32)


def build_text_encoder(kind: str = "hashed", dim: int = 64) -> TextEncoder:
    if kind in ("hashed", "default", "auto"):
        return HashedNGramEncoder(dim=dim)
    if kind == "minilm":
        return MiniLMEncoder()
    raise ValueError(f"unknown text encoder: {kind}")
