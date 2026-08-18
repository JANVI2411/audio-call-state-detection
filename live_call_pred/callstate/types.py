"""
Core data types shared across the pipeline.

The central modelling commitment of this package, and the reason it is not a
flat 4-class chunk classifier:

    IVR / HUMAN / HOLD / OTHER are *persistent states*.
    TRANSFER is an *event* — a transition inferred from state history,
    speaker change, language and telephony evidence.

So there are two output heads (`StateBelief`, `Event`), not one softmax.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional

import numpy as np


class State(str, Enum):
    """Persistent conversational state of the *remote* (counterparty) channel."""

    IVR = "ivr"
    HUMAN = "human"
    HOLD = "hold"
    OTHER = "other"

    @classmethod
    def order(cls) -> List["State"]:
        # Fixed index order. Everything (model weights, HMM transition matrix,
        # metrics tables) indexes states through this, so never reorder it
        # without retraining/regenerating persisted models.
        return [cls.IVR, cls.HUMAN, cls.HOLD, cls.OTHER]

    @classmethod
    def index(cls, s: "State | str") -> int:
        return cls.order().index(State(s))


class EventType(str, Enum):
    TRANSFER_START = "transfer_start"
    TRANSFER_END = "transfer_end"
    SPEAKER_CHANGED = "speaker_changed"
    HUMAN_JOINED = "human_joined"
    IVR_EXIT = "ivr_exit"


@dataclass
class TelephonyEvent:
    """Out-of-band signal from the carrier (SIP / Twilio / conference bridge)."""

    t_s: float
    kind: str  # sip_leg_changed | participant_changed | dtmf | amd_result | ...
    detail: str = ""


@dataclass
class AudioFeatures:
    """Cheap acoustic descriptors for one inference window."""

    speech_prob: float
    music_prob: float
    silence_prob: float
    tone_prob: float
    periodicity: float
    spectral_stability: float
    pitch_cv: float
    embedding: np.ndarray  # log-mel statistics (or WavLM pooled, if enabled)
    syllable_mod: float = 0.0   # envelope-modulation energy at 2.5-8 Hz
    slow_mod: float = 0.0       # envelope-modulation energy at 0.05-1.2 Hz

    def scalars(self) -> Dict[str, float]:
        return {
            "speech_prob": self.speech_prob,
            "music_prob": self.music_prob,
            "silence_prob": self.silence_prob,
            "tone_prob": self.tone_prob,
            "periodicity": self.periodicity,
            "spectral_stability": self.spectral_stability,
            "pitch_cv": self.pitch_cv,
            "syllable_mod": self.syllable_mod,
            "slow_mod": self.slow_mod,
        }


@dataclass
class SpeakerObs:
    """Output of the streaming speaker branch for one window."""

    embedding: Optional[np.ndarray]
    change_prob: float
    similarity_to_active: float
    speaker_id: Optional[str]
    is_new_speaker: bool


@dataclass
class SemanticObs:
    """Output of the ASR + language branch for one window."""

    text: str
    asr_confidence: float
    word_rate: float  # words per second over the window
    ivr_prompt_prob: float
    transfer_phrase_prob: float
    hold_phrase_prob: float
    human_spontaneous_prob: float
    text_embedding: np.ndarray


@dataclass
class Observation:
    """Everything the fusion layer sees at one hop. Strictly causal."""

    t_s: float
    window_start_s: float
    audio: AudioFeatures
    speaker: SpeakerObs
    semantic: SemanticObs
    telephony: Dict[str, float]
    history: Dict[str, float]


@dataclass
class StateBelief:
    """Posterior over states after HMM filtering, plus the committed state."""

    t_s: float
    probs: Dict[str, float]
    state: State
    raw_probs: Dict[str, float]  # pre-HMM, straight from the fusion model
    confidence: float


@dataclass
class Event:
    type: EventType
    t_s: float
    confidence: float
    evidence: str
    meta: Dict[str, str] = field(default_factory=dict)

    def to_json(self) -> dict:
        d = asdict(self)
        d["type"] = self.type.value
        return d


@dataclass
class TimelineRow:
    """One emitted hop of the timeline log."""

    t_s: float
    state: str
    confidence: float
    probs: Dict[str, float]
    speaker_id: Optional[str]
    text: str
    transfer_in_progress: bool
    latency_ms: float
    sub_label: Optional[str] = None
    evidence: str = ""

    def to_json(self) -> dict:
        return asdict(self)


@dataclass
class Segment:
    """Contiguous run of one state — the human-readable view of the timeline."""

    start_s: float
    end_s: float
    state: str
    speaker_id: Optional[str] = None
    mean_confidence: float = 0.0
    sub_label: Optional[str] = None
    evidence: str = ""

    def to_json(self) -> dict:
        return asdict(self)
