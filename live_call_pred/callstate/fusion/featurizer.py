"""
Turns one `Observation` into the fixed-length vector the state model consumes.

Two decisions worth stating explicitly:

1. **Named scalars stay named.** The 24 interpretable scalars (speech/music
   probabilities, lexical scores, speaker-change probability, telephony flags,
   dwell time) keep fixed indices and are exported by name, so a surprising
   prediction can be traced to the feature that drove it. `FEATURE_NAMES` is
   part of the model contract — appending is safe, reordering is not.

2. **Embeddings are randomly projected, not learned down.** The audio and text
   embeddings are compressed to 16 dims each through a seeded Gaussian
   projection. That is a Johnson-Lindenstrauss compression: it preserves
   relative distances well enough for a linear head while keeping the
   parameter count small enough to fit on the amount of labelled call data
   anyone realistically has at the start. Swapping the projection for a
   learned encoder is the natural upgrade once thousands of labelled calls
   exist.

3. **Context is summarised, not stacked flat.** Each hop is represented by
   [current, mean over the last `context_hops`, delta vs. that mean]. This
   gives the linear model temporal context at 3x the dimension instead of
   `context_hops`x, and the delta term is what lets it react to *changes*
   (music starting, speaker switching) rather than only to levels. All three
   terms are causal — they only ever read history.
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List

import numpy as np

from ..types import Observation

SCALAR_NAMES: List[str] = [
    # acoustic
    "speech_prob", "music_prob", "silence_prob", "tone_prob",
    "periodicity", "spectral_stability", "pitch_cv_valid", "pitch_cv",
    "syllable_mod", "slow_mod",
    # speaker
    "speaker_change_prob", "speaker_sim_active", "speaker_known",
    # asr / lexical
    "asr_confidence", "word_rate", "has_text",
    "ivr_prompt_prob", "transfer_phrase_prob", "hold_phrase_prob",
    "human_spontaneous_prob",
    # conversational / telephony / history
    "agent_recently_spoke", "time_since_agent_spoke", "dtmf_recent",
    "sip_leg_changed", "prev_state_is_hold", "dwell_s_norm",
]

EMB_DIM = 16


class Featurizer:
    def __init__(self, cfg, audio_emb_dim: int, text_emb_dim: int, seed: int = 1234):
        self.cfg = cfg
        rng = np.random.default_rng(seed)
        self.P_audio = rng.normal(0, 1.0 / np.sqrt(EMB_DIM), (audio_emb_dim, EMB_DIM)).astype(np.float32)
        self.P_text = rng.normal(0, 1.0 / np.sqrt(EMB_DIM), (text_emb_dim, EMB_DIM)).astype(np.float32)
        self.base_dim = len(SCALAR_NAMES) + 2 * EMB_DIM
        self.dim = 3 * self.base_dim
        self._hist: Deque[np.ndarray] = deque(maxlen=cfg.context_hops)

    def reset(self) -> None:
        self._hist.clear()

    def scalars(self, obs: Observation) -> np.ndarray:
        a, s, m, tel, h = obs.audio, obs.speaker, obs.semantic, obs.telephony, obs.history
        pitch_valid = 1.0 if a.pitch_cv >= 0 else 0.0
        v = [
            a.speech_prob, a.music_prob, a.silence_prob, a.tone_prob,
            a.periodicity, a.spectral_stability, pitch_valid,
            a.pitch_cv if a.pitch_cv >= 0 else 0.0,
            a.syllable_mod, a.slow_mod,
            s.change_prob, s.similarity_to_active, 1.0 if s.speaker_id else 0.0,
            m.asr_confidence, min(m.word_rate / 4.0, 1.5), 1.0 if m.text.strip() else 0.0,
            m.ivr_prompt_prob, m.transfer_phrase_prob, m.hold_phrase_prob,
            m.human_spontaneous_prob,
            tel.get("agent_recently_spoke", 0.0),
            min(tel.get("time_since_agent_spoke", 30.0) / 30.0, 1.0),
            tel.get("dtmf_recent", 0.0),
            tel.get("sip_leg_changed", 0.0),
            h.get("prev_state_is_hold", 0.0),
            min(h.get("dwell_s", 0.0) / 60.0, 1.0),
        ]
        return np.asarray(v, dtype=np.float32)

    def base_vector(self, obs: Observation) -> np.ndarray:
        a_emb = np.asarray(obs.audio.embedding, dtype=np.float32)
        t_emb = np.asarray(obs.semantic.text_embedding, dtype=np.float32)
        a_proj = (a_emb @ self.P_audio) if a_emb.size == self.P_audio.shape[0] else np.zeros(EMB_DIM, np.float32)
        t_proj = (t_emb @ self.P_text) if t_emb.size == self.P_text.shape[0] else np.zeros(EMB_DIM, np.float32)
        a_proj = np.tanh(a_proj / 8.0)  # log-mel stats are O(10); squash to a sane range
        t_proj = np.tanh(t_proj)
        return np.concatenate([self.scalars(obs), a_proj, t_proj]).astype(np.float32)

    def transform(self, obs: Observation) -> np.ndarray:
        """Causal: appends to history, then reads only current + past."""
        cur = self.base_vector(obs)
        self._hist.append(cur)
        ctx = np.mean(np.stack(self._hist), axis=0)
        return np.concatenate([cur, ctx, cur - ctx]).astype(np.float32)

    def named_scalars(self, obs: Observation) -> Dict[str, float]:
        return {n: float(v) for n, v in zip(SCALAR_NAMES, self.scalars(obs))}
