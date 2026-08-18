"""
Turn one hop's internal evidence into a sub-label and a human-readable reason.

The state head answers "what is this" in four words. That is the right answer
for a policy to act on, but it is a poor answer for a person reading a
timeline and asking why. These two fields fill that gap:

  sub_label   the finer distinction inside a state -- hold that is music
              versus hold that is dead air, an IVR reading a keypad menu
              versus one asking a question in a full sentence. Different
              things to a caller, and worth separating in a dataset even
              though the pipeline does not act on them differently.

  evidence    a short phrase naming what actually drove the decision, taken
              from the signals that produced it rather than written after
              the fact.

Both are derived from values the hop already computed, so nothing here costs
extra time and nothing here can disagree with the decision it describes.
"""
from __future__ import annotations

from typing import Optional, Tuple

from .types import AudioFeatures, SemanticObs, State

# How each named feature reads in a sentence, when it turns out to be the
# strongest contributor to the winning state.
PHRASES = {
    "ivr_prompt_prob": "IVR prompt wording",
    "human_spontaneous_prob": "spontaneous speech markers",
    "hold_phrase_prob": "hold announcement wording",
    "transfer_phrase_prob": "transfer announcement",
    "music_prob": "music bed",
    "slow_mod": "slow, music-like rhythm",
    "syllable_mod": "syllable-rate speech rhythm",
    "speech_prob": "speech present",
    "silence_prob": "little or no energy",
    "tone_prob": "call-progress tone",
    "pitch_cv": "pitch variation",
    "periodicity": "repeating loop",
    "spectral_stability": "steady spectrum",
    "dtmf_recent": "keypad tone",
    "word_rate": "speaking rate",
    "has_text": "words recognised",
    "agent_recently_spoke": "our agent just spoke",
    "sip_leg_changed": "carrier reported a new call leg",
}


def sub_label(state: str, af: AudioFeatures, sem: SemanticObs) -> Optional[str]:
    """Finer category inside the committed state, or None when undivided."""
    if state == State.HOLD.value:
        if af.music_prob >= 0.35 and af.speech_prob >= 0.30:
            return "announcement_over_music"
        if af.music_prob >= 0.35:
            return "music"
        if af.silence_prob >= 0.55:
            return "dead_air"
        return "queue"

    if state == State.IVR.value:
        text = (sem.text or "").lower()
        if "press" in text or "keypad" in text or "pound" in text:
            return "keypad_menu"
        if sem.ivr_prompt_prob >= 0.5:
            return "speech_ivr"
        return "automated_prompt"

    if state == State.HUMAN.value:
        if sem.human_spontaneous_prob >= 0.5:
            return "conversational"
        if af.speech_prob >= 0.3:
            return "speech"
        return None

    if state == State.OTHER.value:
        if af.tone_prob >= 0.15:
            return "ringback_or_tone"
        if af.silence_prob >= 0.6:
            return "silence"
        return "unclassified"
    return None


def evidence(state: str, af: AudioFeatures, sem: SemanticObs,
             explain: Optional[dict] = None) -> str:
    """
    One short phrase describing why this state won.

    Prefers the state model's own top contributing feature when it is
    available (`PriorStateModel.explain`), because that is the actual cause
    rather than a plausible-sounding reconstruction. Falls back to the
    strongest raw signal when the model cannot explain itself -- the trained
    logistic head, for instance, has no per-feature story to tell.
    """
    if explain:
        contribs = explain.get(state) or []
        positive = [(n, v) for n, v in contribs if v > 0]
        if positive:
            name, _ = max(positive, key=lambda kv: kv[1])
            phrase = PHRASES.get(name, name)
            snippet = (sem.text or "").strip()
            if name in ("ivr_prompt_prob", "hold_phrase_prob",
                        "transfer_phrase_prob", "human_spontaneous_prob") and snippet:
                return f"{phrase}: \"{snippet[-60:].strip()}\""
            return phrase

    # Fallback: name the loudest raw signal for this state.
    if state == State.HOLD.value:
        return "music bed" if af.music_prob > 0.3 else "no speech in a queue"
    if state == State.IVR.value:
        return ("IVR prompt wording" if sem.ivr_prompt_prob > 0.3
                else "flat, recorded-sounding speech")
    if state == State.HUMAN.value:
        return ("spontaneous speech markers" if sem.human_spontaneous_prob > 0.3
                else "natural pitch variation")
    return "tone or silence"


def describe(state: str, af: AudioFeatures, sem: SemanticObs,
             explain: Optional[dict] = None) -> Tuple[Optional[str], str]:
    return sub_label(state, af, sem), evidence(state, af, sem, explain)
