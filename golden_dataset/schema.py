"""
Shared types for the golden-dataset pipeline.

A "segment" is what one labeler proposes for one stretch of the call.
A "frame" is a fixed-width time bin used only to make cross-labeler
agreement well-defined (segments alone can't be compared directly since
boundaries never line up between models).
"""
from dataclasses import dataclass, field
from typing import Any, Optional

LABELS = ["ivr", "human", "survey", "hold", "unknown"]

FRAME_MS = 250  # grid resolution for agreement voting


@dataclass
class Segment:
    start: float          # seconds
    end: float
    label: str             # one of LABELS
    confidence: float       # 0-1, model-reported
    evidence: str = ""      # one-line justification from the model
    human_id: Optional[str] = None  # only meaningful when label == "human"


@dataclass
class LabelerResult:
    labeler: str            # "gemini" | "gpt_audio" | "asr_llm"
    call_id: str
    segments: list[Segment] = field(default_factory=list)
    raw_response: str = ""  # kept for audit, never shown to end users
    model: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
