"""
Streaming ASR with swappable backends.

`StreamingASR` owns the buffering policy that turns a batch recogniser into a
streaming one: transcribe the trailing `window_s` of audio each hop, then keep
only the words whose timestamps fall after the last hop's cut point. Without
that de-duplication step, overlapping windows re-emit the same words every hop
and the word-rate feature — a genuinely useful IVR-vs-human cue, since IVR
prompts have unnaturally even pacing — becomes meaningless.

Backends:
  FasterWhisperBackend  real recognition, self-hosted (no call audio leaves
                        the process — it matters for payer/PHI calls)
  ScriptedBackend       replays a known transcript against timestamps; this
                        is what makes the end-to-end tests deterministic and
                        runnable with no network
  NullBackend           acoustic-only operation, so the pipeline degrades
                        instead of failing when no ASR is available
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Word:
    start_s: float
    end_s: float
    text: str
    confidence: float = 1.0


@dataclass
class ASRResult:
    words: List[Word] = field(default_factory=list)
    confidence: float = 0.0

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words).strip()


class ASRBackend:
    name: str = "base"

    def transcribe(self, audio: np.ndarray, sr: int, window_start_s: float) -> ASRResult:
        raise NotImplementedError  # pragma: no cover


class NullBackend(ASRBackend):
    name = "null"

    def transcribe(self, audio, sr, window_start_s) -> ASRResult:
        return ASRResult([], 0.0)


class ScriptedBackend(ASRBackend):
    """Replays `(start_s, end_s, text)` utterances that overlap the window."""

    name = "scripted"

    def __init__(self, script: List[Tuple[float, float, str]]):
        self.script = sorted(script, key=lambda s: s[0])

    def transcribe(self, audio, sr, window_start_s) -> ASRResult:
        win_end = window_start_s + len(audio) / float(sr)
        words: List[Word] = []
        for s, e, text in self.script:
            if e <= window_start_s or s >= win_end:
                continue
            toks = text.split()
            if not toks:
                continue
            dur = max(e - s, 1e-3)
            step = dur / len(toks)
            for i, tok in enumerate(toks):
                ws = s + i * step
                if window_start_s <= ws < win_end:
                    words.append(Word(ws, ws + step, tok, 0.95))
        words.sort(key=lambda w: w.start_s)
        return ASRResult(words, 0.95 if words else 0.0)


class FasterWhisperBackend(ASRBackend):
    name = "faster_whisper"

    def __init__(self, model: str = "small.en", compute_type: str = "int8", device: str = "cpu"):
        from faster_whisper import WhisperModel

        self.model = WhisperModel(model, device=device, compute_type=compute_type)

    def transcribe(self, audio, sr, window_start_s) -> ASRResult:
        from ..audio.codecs import resample_linear

        wide = resample_linear(audio, sr, 16000)
        if len(wide) < 1600:
            return ASRResult([], 0.0)
        segments, _info = self.model.transcribe(
            wide, language="en", word_timestamps=True,
            # Whisper's own speech gate, ON. With it off, a window of hold
            # music or line noise makes the decoder hallucinate and then loop
            # on its own output: measured on a real payer call it emitted
            # "You", "You You", "Thank you." as the most frequent "transcripts"
            # and spent 30.1 SECONDS on a single 3-second window. That one
            # setting was responsible for half of all hops breaching the
            # latency budget, and for feeding the IVR word-matcher garbage
            # instead of the prompt text it needs.
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
            condition_on_previous_text=False,
        )
        words: List[Word] = []
        confs: List[float] = []
        for seg in segments:
            confs.append(float(np.exp(getattr(seg, "avg_logprob", -1.0))))
            for w in (getattr(seg, "words", None) or []):
                words.append(Word(window_start_s + w.start, window_start_s + w.end,
                                  w.word.strip(), float(getattr(w, "probability", 0.9))))
        return ASRResult(words, float(np.mean(confs)) if confs else 0.0)


def build_asr_backend(kind: str = "auto", model: str = "small.en",
                      compute_type: str = "int8",
                      script: Optional[List[Tuple[float, float, str]]] = None) -> ASRBackend:
    if kind == "scripted" or (kind == "auto" and script is not None):
        return ScriptedBackend(script or [])
    if kind == "null":
        return NullBackend()
    if kind in ("auto", "faster_whisper"):
        try:
            return FasterWhisperBackend(model=model, compute_type=compute_type)
        except Exception as e:
            if kind == "faster_whisper":
                raise
            logger.warning(
                "faster-whisper unavailable (%s: %s) -- falling back to acoustic-only ASR. "
                "Install with: pip install faster-whisper", type(e).__name__, e,
            )
            return NullBackend()
    raise ValueError(f"unknown ASR backend: {kind}")


class StreamingASR:
    """Buffers, de-duplicates, and exposes a rolling recent-text view."""

    def __init__(self, backend: ASRBackend, window_s: float = 6.0, keep_s: float = 12.0,
                 dedup_tolerance_s: float = 0.7):
        self.backend = backend
        self.window_s = window_s
        self.keep_s = keep_s
        self.dedup_tolerance_s = dedup_tolerance_s
        self.words: List[Word] = []
        self._emitted_until = -1.0
        self._recent_keys: List[Tuple[float, str]] = []
        self.last_confidence = 0.0

    def push(self, window_audio: np.ndarray, sr: int, window_start_s: float) -> List[Word]:
        """
        De-duplicate on both time *and* content.

        A timestamp cutoff alone is not enough with a real recogniser. Each
        window is decoded independently, so word timings drift by a few
        hundred milliseconds between overlapping decodes; a word emitted at
        5.90 s in one window reappears at 6.05 s in the next, clears a
        strictly-increasing cutoff, and is emitted twice. On the real call
        that produced transcripts like "for calling BlueCard. Card
        Eligibility. Eligibility." — which then breaks phrase matching,
        because "thank you for calling" no longer appears contiguously.

        So a word is also dropped if the same token was already emitted within
        `dedup_tolerance_s`. Content-plus-time is robust to the drift in a way
        that neither alone is.
        """
        res = self.backend.transcribe(window_audio, sr, window_start_s)
        self.last_confidence = res.confidence

        fresh: List[Word] = []
        for w in res.words:
            if w.start_s <= self._emitted_until - self.dedup_tolerance_s:
                continue
            key = w.text.strip().lower().strip(".,!?-")
            if any(k == key and abs(t - w.start_s) <= self.dedup_tolerance_s
                   for t, k in self._recent_keys):
                continue
            fresh.append(w)
            self._recent_keys.append((w.start_s, key))

        if fresh:
            self._emitted_until = max(self._emitted_until,
                                      max(w.start_s for w in fresh))
            self.words.extend(fresh)

        now_end = window_start_s + len(window_audio) / float(sr)
        cutoff = now_end - self.keep_s
        self.words = [w for w in self.words if w.end_s >= cutoff]
        self._recent_keys = [(t, k) for t, k in self._recent_keys
                             if t >= now_end - 2 * self.dedup_tolerance_s - 2.0]
        return fresh

    def recent_text(self, now_s: float, lookback_s: float = 8.0) -> str:
        lo = now_s - lookback_s
        return " ".join(w.text for w in self.words if w.end_s >= lo).strip()

    def word_rate(self, now_s: float, lookback_s: float = 5.0) -> float:
        lo = now_s - lookback_s
        n = sum(1 for w in self.words if w.end_s >= lo)
        return n / max(lookback_s, 1e-6)
