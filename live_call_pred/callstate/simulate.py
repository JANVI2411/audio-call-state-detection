"""
Synthetic call generator: audio plus exact ground-truth labels.

This module is what makes the rest of the package testable. Real call
recordings are expensive to label, cannot be committed to a repo (they are
PHI in this domain), and give you no control over whether a transfer appears
at all. Synthetic calls give exact boundaries to the millisecond, arbitrary
transfer scenarios on demand, and a deterministic corpus for training the
fusion head.

What it is *not*: a substitute for real audio. The synthesis is source-filter
speech — a glottal pulse train through formant resonators — which reproduces
the cues the front-end actually keys on (pitch range, spectral movement,
loop periodicity, tone purity) but not codec artefacts, background noise,
crosstalk, or genuine speaker variability. So numbers measured on synthetic
calls validate that the *mechanism* works; they are not accuracy claims about
production traffic. The eval script prints that caveat with every report.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .audio.codecs import write_wav_pcm16

SR = 8000


@dataclass
class Turn:
    start_s: float
    end_s: float
    state: str            # ivr | human | hold | other
    text: str = ""
    speaker: str = ""     # identity tag for human turns


@dataclass
class SyntheticCall:
    remote: np.ndarray
    agent: np.ndarray
    turns: List[Turn]
    gold_events: List[dict] = field(default_factory=list)
    telephony: List[dict] = field(default_factory=list)
    sample_rate: int = SR

    @property
    def duration_s(self) -> float:
        return len(self.remote) / float(self.sample_rate)

    def script(self) -> List[Tuple[float, float, str]]:
        """(start, end, text) triples for the scripted ASR backend."""
        return [(t.start_s, t.end_s, t.text) for t in self.turns if t.text]

    def gold_frames(self, hop_s: float) -> List[Tuple[float, str]]:
        out = []
        t = hop_s
        while t <= self.duration_s + 1e-9:
            state = "other"
            for turn in self.turns:
                if turn.start_s <= t - hop_s / 2 < turn.end_s:
                    state = turn.state
                    break
            out.append((round(t, 3), state))
            t += hop_s
        return out


def _formant_speech(dur_s: float, f0_base: float, f0_range: float,
                    formants: List[float], syllable_rate: float,
                    rng: np.random.Generator, sr: int = SR) -> np.ndarray:
    """
    Source-filter synthesis: a glottal pulse train shaped by two resonators
    and gated by a syllable envelope.

    `f0_range` is the whole point of this function. Setting it near zero gives
    the flat, even delivery of a recorded IVR prompt; setting it wide gives
    the prosodic movement of spontaneous speech. That single parameter is what
    the `pitch_cv` feature is measuring, so the generator and the front-end
    are testing the same physical property rather than a shared shortcut.
    """
    n = int(dur_s * sr)
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    t = np.arange(n) / sr

    contour = f0_base + f0_range * np.sin(2 * np.pi * 0.35 * t + rng.uniform(0, 6.28))
    contour += f0_range * 0.4 * np.sin(2 * np.pi * 1.1 * t)
    contour += rng.normal(0, f0_range * 0.08, n)
    phase = 2 * np.pi * np.cumsum(np.clip(contour, 60, 400)) / sr
    src = np.zeros(n, dtype=np.float64)
    # band-limited pulse train
    for h in range(1, 12):
        src += (1.0 / h) * np.sin(h * phase)

    from scipy.signal import lfilter

    out = np.zeros(n, dtype=np.float64)
    for fc in formants:
        bw = 90.0
        r = np.exp(-np.pi * bw / sr)
        theta = 2 * np.pi * fc / sr
        a = [1.0, -2 * r * np.cos(theta), r * r]
        y = lfilter([1.0], a, src)
        out += y / (np.max(np.abs(y)) + 1e-9)

    # syllable gating: speech is not continuous energy
    syl = 0.5 + 0.5 * np.sin(2 * np.pi * syllable_rate * t + rng.uniform(0, 6.28))
    env = np.clip(syl, 0.08, 1.0) ** 1.5
    # occasional inter-word pauses
    for _ in range(int(dur_s * 0.7)):
        s = rng.integers(0, max(1, n - int(0.18 * sr)))
        env[s : s + int(rng.uniform(0.06, 0.18) * sr)] *= 0.05

    out = out * env
    out = out / (np.max(np.abs(out)) + 1e-9) * 0.32
    return out.astype(np.float32)


def _hold_music(dur_s: float, rng: np.random.Generator, loop_s: float = 4.0,
                sr: int = SR) -> np.ndarray:
    """A short chord loop repeated — periodic envelope, stable spectrum."""
    n_loop = int(loop_s * sr)
    t = np.arange(n_loop) / sr
    chords = [[262, 330, 392], [294, 349, 440], [220, 277, 330], [247, 311, 392]]
    loop = np.zeros(n_loop)
    seg = n_loop // len(chords)
    for i, chord in enumerate(chords):
        s, e = i * seg, (i + 1) * seg
        tt = t[s:e] - t[s]
        env = np.minimum(1.0, tt / 0.05) * np.exp(-tt * 0.8)
        for f in chord:
            loop[s:e] += np.sin(2 * np.pi * f * tt) * env
    loop = loop / (np.max(np.abs(loop)) + 1e-9) * 0.22
    reps = int(np.ceil(dur_s * sr / n_loop))
    out = np.tile(loop, reps)[: int(dur_s * sr)]
    return (out + rng.normal(0, 0.002, len(out))).astype(np.float32)


def _tone(dur_s: float, freqs: List[float], on_s: float = 2.0, off_s: float = 4.0,
          sr: int = SR) -> np.ndarray:
    n = int(dur_s * sr)
    t = np.arange(n) / sr
    sig = sum(np.sin(2 * np.pi * f * t) for f in freqs) / max(len(freqs), 1)
    cycle = on_s + off_s
    gate = ((t % cycle) < on_s).astype(np.float64)
    return (sig * gate * 0.25).astype(np.float32)


def _silence(dur_s: float, rng: np.random.Generator, sr: int = SR) -> np.ndarray:
    return rng.normal(0, 0.0015, int(dur_s * sr)).astype(np.float32)


IVR_LINES = [
    "thank you for calling the provider services line please listen carefully as our menu options have changed",
    "for eligibility and benefits press or say one for claims status press two",
    "please enter the member identification number followed by the pound key",
    "did you say pharmacy benefits please say yes or no",
    "to repeat this menu press nine to return to the main menu press star",
]
HOLD_LINES = [
    "thanks for holding all of our representatives are currently assisting other callers",
    "your call is important to us please stay on the line and do not hang up",
]
HUMAN_A_LINES = [
    "uh yeah hi this is brenda with provider services can i get the member id",
    "okay let me check that for you one second while i pull up the record",
    "so it looks like the deductible is uh twenty five hundred and about eleven hundred has been met",
]
HUMAN_B_LINES = [
    "hi um this is marcus over in claims sorry about the wait what can i do for you",
    "yeah okay i see that here let me look at the prior auth on this one",
]
TRANSFER_LINES = [
    "okay that one is actually handled by another department let me get you over to claims please hold",
]
FAIL_LINES = [
    "sorry about that they're not available right now so i'll go ahead and help you myself",
]


def make_call(scenario: str = "transfer", seed: int = 7,
              sr: int = SR) -> SyntheticCall:
    """
    Build one labelled synthetic call.

    Scenarios:
      ivr_only     ringback, then an IVR menu throughout — no human ever
      simple       IVR, brief hold, one representative
      transfer     IVR, rep A, announcement, hold, rep B  (the important one:
                   a *successful* transfer with a genuine speaker change)
      failed_transfer  announcement, hold, then rep A returns — the case a
                   naive detector reports as a completed transfer

    Every call draws its own voices, segment durations, line selection, gain
    and noise floor from `seed`. That variation is not decoration: with calls
    that differed only by noise realisation, a trained fusion head scored
    100% on *both* train and a by-call holdout, because a held-out
    `transfer_0` was essentially the same recording as the `transfer_1` it had
    trained on. A holdout only measures generalisation if the calls in it
    genuinely differ, so the generator varies the things that vary in real
    calls — who is speaking, how long they speak, which prompts play, and how
    the line sounds.
    """
    rng = np.random.default_rng(seed)
    turns: List[Turn] = []
    gold_events: List[dict] = []
    telephony: List[dict] = []
    chunks: List[np.ndarray] = []
    agent_chunks: List[np.ndarray] = []
    t = 0.0

    # Per-call voice draw: two distinct speakers, plus an IVR "voice".
    def draw_voice(lo_f0: float, hi_f0: float):
        f0 = float(rng.uniform(lo_f0, hi_f0))
        f1 = float(rng.uniform(400, 720))
        return f0, [f1, f1 + float(rng.uniform(850, 1350))]

    voice_a = draw_voice(170, 230)
    voice_b = draw_voice(95, 145)
    ivr_f0, ivr_formants = draw_voice(140, 190)
    ivr_flatness = float(rng.uniform(2.0, 7.0))     # IVR prosody is narrow but not zero
    human_range = float(rng.uniform(24.0, 45.0))
    line_gain = float(rng.uniform(0.7, 1.25))       # per-call level
    noise_floor = float(rng.uniform(0.0008, 0.004))
    loop_s = float(rng.uniform(3.0, 6.0))           # hold-loop length varies by payer

    def jitter(base: float, frac: float = 0.35) -> float:
        return float(max(2.0, base * rng.uniform(1 - frac, 1 + frac)))

    def pick(lines: List[str]) -> str:
        return lines[int(rng.integers(0, len(lines)))]

    def add(dur: float, audio: np.ndarray, state: str, text: str = "",
            speaker: str = "", agent_says: bool = False) -> None:
        nonlocal t
        audio = audio * line_gain + rng.normal(0, noise_floor, len(audio)).astype(np.float32)
        chunks.append(audio.astype(np.float32))
        agent_chunks.append(
            _formant_speech(dur, 130, 14, [520, 1600], 3.6, rng) * 0.9 if agent_says
            else _silence(dur, rng)
        )
        turns.append(Turn(t, t + dur, state, text, speaker))
        t += dur

    def ivr(dur: float, line: str) -> None:
        # Narrow pitch range: the acoustic signature of a recorded prompt.
        add(dur, _formant_speech(dur, ivr_f0, ivr_flatness, ivr_formants,
                                 float(rng.uniform(3.6, 4.8)), rng), "ivr", line)

    def human(dur: float, line: str, who: str) -> None:
        f0, formants = voice_a if who == "human_a" else voice_b
        add(dur, _formant_speech(dur, f0, human_range, formants,
                                 float(rng.uniform(3.0, 4.0)), rng),
            "human", line, who, agent_says=rng.random() < 0.5)

    def hold(dur: float, line: str = "") -> None:
        audio = _hold_music(dur, rng, loop_s=loop_s)
        if line:
            speech = _formant_speech(min(dur, float(rng.uniform(3.0, 5.0))), ivr_f0,
                                     ivr_flatness, ivr_formants, 4.0, rng)
            audio = audio.copy()
            audio[: len(speech)] += speech * float(rng.uniform(0.6, 1.0))
        add(dur, audio, "hold", line)

    def ringback(base: float) -> None:
        d = jitter(base)
        add(d, _tone(d, [440.0, 480.0], on_s=rng.uniform(1.6, 2.4),
                     off_s=rng.uniform(3.0, 4.5)), "other")

    def gap(base: float, state: str) -> None:
        d = jitter(base, 0.5)
        add(d, _silence(d, rng), state)

    if scenario == "ivr_only":
        ringback(6.0)
        for line in rng.permutation(IVR_LINES)[: int(rng.integers(3, 6))]:
            ivr(jitter(7.0), str(line))
            # The pause after a prompt is labelled IVR, not OTHER. A menu
            # waiting for a keypress is still the IVR state — the caller has
            # not gone anywhere. Labelling these gaps OTHER made gold flip
            # state ten times in a 48 s call and penalised the tracker for
            # correctly staying put; the label was wrong, not the prediction.
            gap(1.5, "ivr")

    elif scenario == "simple":
        ringback(5.0)
        ivr(jitter(8.0), pick(IVR_LINES))
        ivr(jitter(7.0), pick(IVR_LINES))
        hold(jitter(10.0), pick(HOLD_LINES))
        for line in rng.permutation(HUMAN_A_LINES)[: int(rng.integers(2, 4))]:
            human(jitter(7.0), str(line), "human_a")

    elif scenario == "transfer":
        ringback(4.0)
        ivr(jitter(8.0), pick(IVR_LINES))
        if rng.random() < 0.6:
            ivr(jitter(7.0), pick(IVR_LINES))
        hold(jitter(8.0), pick(HOLD_LINES))
        human(jitter(7.0), HUMAN_A_LINES[0], "human_a")
        human(jitter(8.0), HUMAN_A_LINES[1], "human_a")
        announce_t = t
        human(jitter(6.0), pick(TRANSFER_LINES), "human_a")
        gold_events.append({"type": "transfer_start", "t_s": announce_t})
        telephony.append({"t_s": round(t + rng.uniform(0.5, 2.0), 2),
                          "kind": "sip_leg_changed", "detail": "leg-2 created"})
        hold(jitter(12.0), pick(HOLD_LINES))
        complete_t = t
        human(jitter(8.0), HUMAN_B_LINES[0], "human_b")
        human(jitter(7.0), HUMAN_B_LINES[1], "human_b")
        gold_events.append({"type": "transfer_end", "t_s": complete_t, "outcome": "completed"})

    elif scenario == "failed_transfer":
        if rng.random() < 0.5:
            ringback(4.0)
        ivr(jitter(7.0), pick(IVR_LINES))
        human(jitter(7.0), HUMAN_A_LINES[0], "human_a")
        announce_t = t
        human(jitter(6.0), pick(TRANSFER_LINES), "human_a")
        gold_events.append({"type": "transfer_start", "t_s": announce_t})
        hold(jitter(10.0))
        fail_t = t
        human(jitter(7.0), pick(FAIL_LINES), "human_a")
        human(jitter(7.0), HUMAN_A_LINES[2], "human_a")
        gold_events.append({"type": "transfer_end", "t_s": fail_t, "outcome": "failed"})

    else:
        raise ValueError(f"unknown scenario: {scenario}")

    return SyntheticCall(
        remote=np.concatenate(chunks).astype(np.float32),
        agent=np.concatenate(agent_chunks).astype(np.float32),
        turns=turns, gold_events=gold_events, telephony=telephony, sample_rate=sr,
    )


def write_call(call: SyntheticCall, out_dir: str, name: str) -> Dict[str, str]:
    """Write wav + gold labels + telephony log; returns the paths written."""
    os.makedirs(out_dir, exist_ok=True)
    wav_path = os.path.join(out_dir, f"{name}.wav")
    stereo = np.stack([call.agent, call.remote], axis=1)  # ch0 = agent, ch1 = remote
    write_wav_pcm16(wav_path, stereo, call.sample_rate)

    gold_path = os.path.join(out_dir, f"{name}.gold.jsonl")
    with open(gold_path, "w") as fh:
        for turn in call.turns:
            fh.write(json.dumps({
                "start_s": round(turn.start_s, 3), "end_s": round(turn.end_s, 3),
                "state": turn.state, "speaker": turn.speaker, "text": turn.text,
            }) + "\n")

    ev_path = os.path.join(out_dir, f"{name}.gold_events.json")
    with open(ev_path, "w") as fh:
        json.dump(call.gold_events, fh, indent=2)

    tel_path = os.path.join(out_dir, f"{name}.telephony.jsonl")
    with open(tel_path, "w") as fh:
        for ev in call.telephony:
            fh.write(json.dumps(ev) + "\n")

    script_path = os.path.join(out_dir, f"{name}.script.json")
    with open(script_path, "w") as fh:
        json.dump(call.script(), fh, indent=2)

    return {"wav": wav_path, "gold": gold_path, "events": ev_path,
            "telephony": tel_path, "script": script_path}
