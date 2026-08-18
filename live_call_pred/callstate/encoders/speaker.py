"""
Streaming speaker branch: embedding, change detection, and an online registry.

This is the component that makes "a transfer happened" detectable rather than
guessed. `HUMAN_A → hold → HUMAN_B` is a near-conclusive transfer signature,
and it is only visible if we can tell HUMAN_B from HUMAN_A. Crucially this is
done *online* — we never wait for the call to finish and diarize it, we
compare each window against a registry of speakers seen so far.

Embeddings: MFCC mean/std by default (classical, zero dependencies). C0 is
dropped deliberately — it is log frame energy, so leaving it in makes the
"embedding" mostly a loudness measure and two utterances from the same person
at different volumes stop matching. That was a real bug, and dropping C0 is
the fix. ECAPA-TDNN is the production choice (`--speaker-encoder ecapa`);
it is trained for exactly this and is far more robust to codec and channel
mismatch, at the cost of torch plus a model download.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..audio.features import logmel
from ..types import SpeakerObs


def _dct_ii(x: np.ndarray, n_out: int) -> np.ndarray:
    n = x.shape[-1]
    k = np.arange(n_out)[:, None]
    i = np.arange(n)[None, :]
    basis = np.cos(np.pi * k * (2 * i + 1) / (2.0 * n))
    return x @ basis.T


def embed_mfcc(x: np.ndarray, sr: int, n_mfcc: int = 13, n_mels: int = 26) -> np.ndarray:
    """MFCC mean/std embedding with C0 dropped and per-vector L2 normalisation."""
    lm = logmel(x, sr, n_mels=n_mels, frame_ms=32)
    if len(lm) < 3:
        return np.zeros(2 * (n_mfcc - 1), dtype=np.float32)
    mfcc = _dct_ii(lm, n_mfcc)[:, 1:]  # drop C0 (log energy)
    v = np.concatenate([mfcc.mean(axis=0), mfcc.std(axis=0)]).astype(np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


@dataclass
class SpeakerRegistry:
    """
    Online speaker registry with hysteresis.

    Hysteresis exists because of a specific failure mode: the first window
    after hold music ends contains the tail of the music bleeding into the
    start of speech, which produces a garbage embedding that matches nobody
    and registers a phantom third speaker. Requiring `hysteresis` consecutive
    windows to agree before *committing* a brand-new identity removes it,
    mirroring the state tracker's own debouncing.
    """

    threshold: float = 0.86
    hysteresis: int = 2
    centroids: Dict[str, np.ndarray] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)
    active_id: Optional[str] = None
    _pending_emb: Optional[np.ndarray] = None
    _pending_count: int = 0

    def _new_id(self) -> str:
        return f"human_{len(self.centroids) + 1}"

    def observe(self, emb: np.ndarray) -> Tuple[Optional[str], float, bool]:
        """Returns (speaker_id, similarity, is_new_speaker)."""
        if emb is None or not np.any(emb):
            return self.active_id, 0.0, False

        best_id, best_sim = None, -1.0
        for sid, c in self.centroids.items():
            s = cosine(emb, c)
            if s > best_sim:
                best_id, best_sim = sid, s

        if best_id is not None and best_sim >= self.threshold:
            self._pending_emb, self._pending_count = None, 0
            self.counts[best_id] += 1
            c = self.centroids[best_id]
            k = self.counts[best_id]
            self.centroids[best_id] = ((k - 1) * c + emb) / k  # running mean
            self.active_id = best_id
            return best_id, best_sim, False

        # Candidate new speaker — must persist before we commit an identity.
        if self._pending_emb is not None and cosine(emb, self._pending_emb) >= self.threshold:
            self._pending_count += 1
        else:
            self._pending_emb, self._pending_count = emb, 1

        if self._pending_count >= self.hysteresis or not self.centroids:
            sid = self._new_id()
            self.centroids[sid] = emb.copy()
            self.counts[sid] = 1
            self.active_id = sid
            self._pending_emb, self._pending_count = None, 0
            return sid, max(best_sim, 0.0), True

        return self.active_id, max(best_sim, 0.0), False


class SpeakerBranch:
    """Wraps embedding + registry + change probability into one per-hop call."""

    def __init__(self, cfg, encoder_kind: str = "mfcc"):
        self.cfg = cfg
        self.kind = encoder_kind
        self.registry = SpeakerRegistry(
            threshold=cfg.speaker_similarity_threshold,
            hysteresis=cfg.speaker_change_hysteresis,
        )
        self.prev_emb: Optional[np.ndarray] = None
        self._ecapa = None
        if encoder_kind == "ecapa":
            from speechbrain.inference import EncoderClassifier  # noqa: F401

            self._ecapa = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb"
            )

    def _embed(self, x: np.ndarray, sr: int) -> np.ndarray:
        if self._ecapa is not None:
            import torch

            from ..audio.codecs import resample_linear

            wide = resample_linear(x, sr, 16000)
            with torch.no_grad():
                e = self._ecapa.encode_batch(torch.tensor(wide)[None, :])
            v = e.squeeze().cpu().numpy().astype(np.float32)
            n = np.linalg.norm(v)
            return v / n if n > 0 else v
        return embed_mfcc(x, sr)

    def observe(self, x: np.ndarray, sr: int, speech_prob: float,
                is_human_like: bool) -> SpeakerObs:
        """
        Only track identity where it means something.

        Running the registry on hold music or an IVR prompt pollutes it with
        non-speaker centroids and destroys the "did the person change" signal,
        so windows that are not human-like speech return a null observation
        and leave the registry untouched.
        """
        if not is_human_like or speech_prob < self.cfg.speaker_min_speech_prob:
            return SpeakerObs(None, 0.0, 0.0, self.registry.active_id, False)

        emb = self._embed(x, sr)
        prev_active = self.registry.active_id
        sid, sim, is_new = self.registry.observe(emb)

        if self.prev_emb is not None:
            change_prob = float(np.clip((1.0 - cosine(emb, self.prev_emb)) / 0.30, 0.0, 1.0))
        else:
            change_prob = 0.0
        if is_new or (sid is not None and prev_active is not None and sid != prev_active):
            change_prob = max(change_prob, 0.9)
        self.prev_emb = emb

        return SpeakerObs(
            embedding=emb, change_prob=change_prob, similarity_to_active=sim,
            speaker_id=sid, is_new_speaker=is_new,
        )
