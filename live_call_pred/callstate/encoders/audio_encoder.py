"""
Pluggable audio embedding branch.

Default is `LogMelStatsEncoder`: mean/std pooled log-mel, pure numpy, ~0.3 ms
per 6 s window on one core, no download, no GPU. It is the honest baseline —
good enough to separate music from speech from tone, weak at the subtler
"is this voice scripted or spontaneous" judgment.

`WavLMEncoder` is the production upgrade: a pretrained self-supervised speech
model whose representations carry far more speaker and paralinguistic detail.
It is opt-in (`--audio-encoder wavlm`) because it needs torch, transformers,
a ~360 MB download, 16 kHz input, and roughly two orders of magnitude more
compute per window. Both satisfy the same interface, so switching is a flag,
not a refactor — that separation is the point of this module.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..audio.features import logmel
from ..audio.codecs import resample_linear


class AudioEncoder:
    dim: int

    def encode(self, x: np.ndarray, sr: int) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class LogMelStatsEncoder(AudioEncoder):
    def __init__(self, n_mels: int = 24):
        self.n_mels = n_mels
        self.dim = 2 * n_mels

    def encode(self, x: np.ndarray, sr: int) -> np.ndarray:
        lm = logmel(x, sr, self.n_mels)
        if len(lm) == 0:
            return np.zeros(self.dim, dtype=np.float32)
        return np.concatenate([lm.mean(axis=0), lm.std(axis=0)]).astype(np.float32)


class WavLMEncoder(AudioEncoder):
    """
    Mean-pooled WavLM hidden states. Lazy-loads so importing this module never
    costs anything unless the encoder is actually selected.
    """

    def __init__(self, model_name: str = "microsoft/wavlm-base-plus", device: str = "cpu"):
        import torch  # noqa: F401  (import here so torch stays optional)
        from transformers import AutoModel, AutoFeatureExtractor

        self._torch = __import__("torch")
        self.device = device
        self.fe = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.dim = int(self.model.config.hidden_size)

    def encode(self, x: np.ndarray, sr: int) -> np.ndarray:
        torch = self._torch
        # WavLM is trained on 16 kHz; telephony audio must be upsampled even
        # though it carries no information above 4 kHz. That mismatch is a real
        # accuracy cost and the reason a telephony-finetuned checkpoint beats
        # stock WavLM on this data.
        wide = resample_linear(x, sr, 16000)
        if len(wide) < 400:
            return np.zeros(self.dim, dtype=np.float32)
        inputs = self.fe(wide, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            out = self.model(inputs.input_values.to(self.device)).last_hidden_state
        return out.mean(dim=1).squeeze(0).cpu().numpy().astype(np.float32)


def build_audio_encoder(kind: str = "logmel", n_mels: int = 24) -> AudioEncoder:
    if kind in ("logmel", "default", "auto"):
        return LogMelStatsEncoder(n_mels=n_mels)
    if kind == "wavlm":
        return WavLMEncoder()
    raise ValueError(f"unknown audio encoder: {kind}")
