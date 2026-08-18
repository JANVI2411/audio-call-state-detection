"""
Every tunable in one place. Defaults are the ones the tests and the shipped
model checkpoint were validated against; changing them invalidates the
checkpoint's calibration, so `Config` is hashed into the run summary.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from typing import Dict, List


@dataclass
class Config:
    # --- streaming geometry ------------------------------------------------
    target_sr: int = 8000          # telephony native rate; no upsampling by default
    frame_ms: int = 20             # analysis frame for VAD/energy
    window_s: float = 6.0          # causal context fed to the encoders / ASR
    hop_s: float = 0.5             # inference cadence
    emit_chunk_s: float = 2.0      # aggregation used only for reporting/eval

    # Acoustic evidence is read from a *shorter* trailing slice than the full
    # encoder window. Detection lag is bounded below by how much stale audio
    # the decision sees, so measuring music/speech over the whole 6 s window
    # keeps a state alive for seconds after it has actually ended. Language
    # wants the opposite -- a phrase needs its context -- so the lexical
    # lookback stays longer. Measured effect of splitting these: boundary
    # recall@1s went from 0.00 to 0.60 on the synthetic corpus.
    decision_window_s: float = 2.5
    lexical_lookback_s: float = 4.0

    # --- VAD ---------------------------------------------------------------
    vad_energy_percentile: float = 25.0
    vad_rel_threshold_db: float = 9.0
    vad_min_speech_frames: int = 3

    # --- acoustic heads ----------------------------------------------------
    n_mels: int = 24
    music_periodicity_threshold: float = 0.42
    # Two-peak spectral concentration above which a window is treated as a
    # call-progress tone. Measured over 6 s windows, 25 seeds each:
    #   speech    0.008 - 0.016
    #   music     0.073 - 0.101
    #   ringback  0.202 - 0.321
    # 0.15 sits in the empty gap between music and ringback, so it is set from
    # the measurement rather than picked.
    tone_min_inband_fraction: float = 0.15

    # --- speaker branch ----------------------------------------------------
    speaker_window_s: float = 3.0
    speaker_similarity_threshold: float = 0.86
    speaker_change_hysteresis: int = 2   # consecutive hops before committing a new id
    speaker_min_speech_prob: float = 0.5

    # --- ASR ---------------------------------------------------------------
    asr_backend: str = "auto"      # auto | faster_whisper | scripted | null
    asr_model: str = "small.en"
    asr_compute_type: str = "int8"
    # ASR runs on its own, slower cadence over its own, shorter window. The
    # state loop hops every 0.5 s over a 6 s context; feeding that same 6 s to
    # a recogniser every hop re-transcribes each second of audio twelve times,
    # which dominates the per-hop cost and buys nothing — words do not change
    # once spoken. Running a 3 s window every 1 s cuts the redundancy to 3x.
    # Between ASR runs the state loop reuses the most recent transcript, which
    # is exactly what a real streaming recogniser's partial hypotheses give.
    asr_hop_s: float = 1.0
    asr_window_s: float = 3.0
    # Below this much acoustic speech evidence, do not call the recogniser at
    # all. Set deliberately low: the cost of skipping real speech is a missed
    # prompt, so this only screens out windows that are clearly music, tone or
    # silence. See CallStateEngine.step for why the gate exists.
    asr_min_speech_prob: float = 0.12

    # --- fusion model ------------------------------------------------------
    context_hops: int = 8          # stacked history fed to the state model (causal)
    model_path: str = ""           # empty -> use the built-in prior weights

    # --- HMM ---------------------------------------------------------------
    # Self-transition mass. Higher = stickier states = less flicker, more lag.
    hmm_self_prob: Dict[str, float] = field(
        default_factory=lambda: {"ivr": 0.93, "human": 0.96, "hold": 0.96, "other": 0.80}
    )
    # Relative weight of each off-diagonal transition (rows renormalised).
    hmm_transition_bias: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            "ivr": {"human": 1.0, "hold": 1.0, "other": 0.4},
            "human": {"hold": 1.0, "ivr": 0.3, "other": 0.4},
            "hold": {"human": 1.5, "ivr": 0.8, "other": 0.3},
            "other": {"human": 1.0, "ivr": 1.0, "hold": 1.0},
        }
    )
    emission_temperature: float = 1.0
    commit_min_prob: float = 0.45   # below this the tracker holds the previous state

    # --- transfer event detector ------------------------------------------
    transfer_announce_window_s: float = 45.0   # announcement stays "live" this long
    transfer_max_duration_s: float = 240.0     # after this, an open transfer is abandoned
    transfer_min_hold_s: float = 1.0

    # --- runtime -----------------------------------------------------------
    realtime: bool = False          # pace the file source at wall-clock speed
    log_level: str = "INFO"

    def to_json(self) -> dict:
        return asdict(self)

    def fingerprint(self) -> str:
        import hashlib

        blob = json.dumps(self.to_json(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    @property
    def hop_frames(self) -> int:
        return int(round(self.hop_s * self.target_sr))

    @property
    def window_samples(self) -> int:
        return int(round(self.window_s * self.target_sr))

    @classmethod
    def from_json_file(cls, path: str) -> "Config":
        with open(path) as fh:
            return cls(**json.load(fh))
