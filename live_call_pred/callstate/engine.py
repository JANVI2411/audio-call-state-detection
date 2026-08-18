"""
The engine: one hop of the streaming loop, and the driver that runs a call.

The loop is deliberately boring, because that is the property that matters —
every hop reads the ring buffer (bounded look-back, zero look-ahead), runs the
branches, fuses, filters, and commits. Swapping `WavFileSource` for a live
websocket source changes nothing below the source, which is what makes the
"this is streamable" claim structural rather than aspirational.

Per-hop cost is measured, not asserted: every hop records its own wall-clock
processing time and the per-branch split, written to `<call>.latency.jsonl`.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .audio import features as F
from .audio.source import FrameSource, RingBuffer
from .config import Config
from .encoders.audio_encoder import build_audio_encoder
from .encoders.speaker import SpeakerBranch
from .encoders.text_encoder import build_text_encoder
from .explain import describe
from .fusion.events import TransferDetector
from .fusion.featurizer import Featurizer
from .fusion.hmm import HMMFilter, build_transition_matrix
from .fusion.model import PriorStateModel, build_state_model
from .fusion.state_tracker import StateTracker
from .logging_setup import log_kv
from .semantics.asr import StreamingASR, build_asr_backend
from .semantics.lexicon import score_text
from .telephony import TelephonyBus
from .types import (AudioFeatures, Event, Observation, SemanticObs, Segment,
                    SpeakerObs, State, StateBelief, TimelineRow)

logger = logging.getLogger("callstate.engine")


@dataclass
class CallResult:
    call_id: str
    duration_s: float
    timeline: List[TimelineRow] = field(default_factory=list)
    events: List[Event] = field(default_factory=list)
    segments: List[Segment] = field(default_factory=list)
    latency: List[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)


class CallStateEngine:
    def __init__(self, cfg: Optional[Config] = None, asr_backend=None,
                 audio_encoder: str = "logmel", text_encoder: str = "hashed",
                 speaker_encoder: str = "mfcc",
                 telephony: Optional[TelephonyBus] = None):
        self.cfg = cfg or Config()
        c = self.cfg

        self.audio_encoder = build_audio_encoder(audio_encoder, n_mels=c.n_mels)
        self.text_encoder = build_text_encoder(text_encoder)
        self.speaker = SpeakerBranch(c, encoder_kind=speaker_encoder)
        self.asr = StreamingASR(
            asr_backend or build_asr_backend(c.asr_backend, c.asr_model, c.asr_compute_type),
            window_s=c.window_s,
        )
        self.telephony = telephony or TelephonyBus()

        self.featurizer = Featurizer(c, self.audio_encoder.dim, self.text_encoder.dim)
        self.model = build_state_model(c, self.featurizer.dim)
        self.hmm = HMMFilter(build_transition_matrix(c.hmm_self_prob, c.hmm_transition_bias))
        self.tracker = StateTracker(cfg=c)
        self.transfers = TransferDetector(cfg=c)

        self.buffer = RingBuffer(c.window_samples)
        self.agent_buffer = RingBuffer(int(c.target_sr * 1.0))
        self._last_hop_s = -1e9
        self._last_asr_s = -1e9
        self._hop_index = 0

        # Optional capture of (t_s, feature_vector) for training the fusion
        # head. Off by default — it is the only thing in the hop that grows
        # without bound, so it must never be on in a live call.
        self._last_explanation = (None, "")
        self.collect_features = False
        self.feature_log: List[Tuple[float, np.ndarray]] = []

    # -- one hop -----------------------------------------------------------
    def step(self, t_s: float,
             decision_audio: Optional[np.ndarray] = None
             ) -> Tuple[StateBelief, List[Event], dict]:
        c = self.cfg
        t0 = time.perf_counter()
        window = self.buffer.read()
        window_start_s = max(0.0, t_s - len(window) / float(c.target_sr))

        # 1. acoustic — decided on the recent slice, embedded over the full
        #    window (see Config.decision_window_s for why they differ)
        #
        # `decision_audio` lets a caller supply the slice itself, which is how
        # utterance mode works: the decision covers exactly one utterance
        # rather than a fixed trailing window. The long window is unchanged
        # either way, because the music band genuinely needs several seconds
        # and a 0.6 s utterance cannot supply them.
        if decision_audio is not None and len(decision_audio) >= c.target_sr // 10:
            decision = decision_audio
        else:
            decision = window[-int(c.decision_window_s * c.target_sr) :]
        af = F.extract(decision, c.target_sr, c, x_long=window)
        af.embedding = self.audio_encoder.encode(window, c.target_sr)
        t_acoustic = time.perf_counter()

        # 2. agent leg (privileged: we know when *we* are speaking)
        agent_win = self.agent_buffer.read()
        if len(agent_win) and float(np.sqrt(np.mean(agent_win ** 2))) > 0.01:
            self.telephony.note_agent_speech(t_s)

        # 3. DTMF on the most recent second only — it is a transient, and
        #    rescanning the whole window every hop would re-report old digits.
        recent = window[-c.target_sr :] if len(window) > c.target_sr else window
        if len(recent) > 256:
            for _off, key in F.detect_dtmf(recent, c.target_sr,
                                           min_inband_fraction=c.tone_min_inband_fraction):
                self.telephony.note_dtmf(t_s)
                logger.debug("dtmf digit %s at %.1fs", key, t_s)

        # 4. ASR + language — own cadence, own window (see Config.asr_hop_s)
        #
        # Gated on our own speech evidence. We already know, from the acoustic
        # branch, whether this window plausibly contains speech; handing a
        # music-only or silent window to the recogniser asks it to rediscover
        # that at enormous cost, and on real audio it does not merely waste
        # the time -- it hallucinates. Skipping those windows is both the
        # cheapest and the most accurate option, and it is free evidence we
        # were already computing and throwing away.
        if t_s - self._last_asr_s + 1e-9 >= c.asr_hop_s:
            self._last_asr_s = t_s
            if af.speech_prob >= c.asr_min_speech_prob:
                asr_win = window[-int(c.asr_window_s * c.target_sr) :]
                asr_start = t_s - len(asr_win) / float(c.target_sr)
                self.asr.push(asr_win, c.target_sr, max(0.0, asr_start))
            else:
                logger.debug("asr skipped at %.1fs (speech_prob=%.2f)",
                             t_s, af.speech_prob)
        text = self.asr.recent_text(t_s, lookback_s=c.lexical_lookback_s)
        lex = score_text(text)
        sem = SemanticObs(
            text=text, asr_confidence=self.asr.last_confidence,
            word_rate=self.asr.word_rate(t_s),
            ivr_prompt_prob=lex.ivr_prompt_prob,
            transfer_phrase_prob=lex.transfer_phrase_prob,
            hold_phrase_prob=lex.hold_phrase_prob,
            human_spontaneous_prob=lex.human_spontaneous_prob,
            text_embedding=self.text_encoder.encode(text),
        )
        t_asr = time.perf_counter()

        # 5. speaker — only on human-like audio (see SpeakerBranch.observe)
        human_like = (af.speech_prob > 0.35 and af.music_prob < 0.6
                      and lex.ivr_prompt_prob < 0.6)
        spk_window = window[-int(c.speaker_window_s * c.target_sr) :]
        spk = self.speaker.observe(spk_window, c.target_sr, af.speech_prob, human_like)
        t_speaker = time.perf_counter()

        # 6. fuse -> emission -> HMM -> commit
        self.telephony.drain_until(t_s)
        obs = Observation(
            t_s=t_s, window_start_s=window_start_s, audio=af, speaker=spk, semantic=sem,
            telephony=self.telephony.features(t_s),
            history={"prev_state_is_hold": 1.0 if self.tracker.state == State.HOLD else 0.0,
                     "dwell_s": self.tracker.dwell_s(t_s)},
        )
        x = self.featurizer.transform(obs)
        if self.collect_features:
            self.feature_log.append((t_s, x.copy()))
        raw = self.model.predict_proba(x)
        posterior = self.hmm.step(raw)
        belief = self.tracker.commit(t_s, posterior, raw)

        # Sub-label and human-readable reason, from the same evidence that
        # produced the decision (see callstate/explain.py).
        expl = (self.model.explain(self.featurizer.scalars(obs))
                if isinstance(self.model, PriorStateModel) else None)
        self._last_explanation = describe(belief.state.value, af, sem, expl)

        # 7. events
        events = self.transfers.step(
            t_s=t_s, state=belief.state, speaker_id=spk.speaker_id,
            is_new_speaker=spk.is_new_speaker,
            transfer_phrase_prob=lex.transfer_phrase_prob,
            transfer_fail_prob=lex.transfer_fail_prob,
            confidence=belief.confidence,
            telephony_leg_changed=self.telephony.leg_changed_recently(t_s),
            dwell_s=self.tracker.dwell_s(t_s),
        )
        t_end = time.perf_counter()

        timing = {
            "t_s": round(t_s, 2),
            "total_ms": round((t_end - t0) * 1000, 2),
            "acoustic_ms": round((t_acoustic - t0) * 1000, 2),
            "asr_ms": round((t_asr - t_acoustic) * 1000, 2),
            "speaker_ms": round((t_speaker - t_asr) * 1000, 2),
            "fusion_ms": round((t_end - t_speaker) * 1000, 2),
        }
        self._hop_index += 1

        log_kv(
            logger, logging.DEBUG,
            f"{t_s:7.2f}s {belief.state.value:<6} p={belief.confidence:.2f} "
            f"spk={spk.speaker_id or '-'} text={text[-60:]!r}",
            t_s=round(t_s, 2), state=belief.state.value, probs=belief.probs,
            raw_probs=belief.raw_probs, speaker=spk.speaker_id,
            features=self.featurizer.named_scalars(obs),
            explain=(self.model.explain(self.featurizer.scalars(obs))
                     if isinstance(self.model, PriorStateModel) else None),
            latency_ms=timing["total_ms"],
        )
        for ev in events:
            log_kv(logger, logging.INFO,
                   f"[event] {ev.type.value} @ {ev.t_s:.1f}s — {ev.evidence}",
                   event=ev.to_json())

        return belief, events, timing

    # -- whole call --------------------------------------------------------
    def run(self, source: FrameSource, call_id: str = "call",
            on_hop=None, segmenter=None) -> CallResult:
        """
        `on_hop(t_s, row, timing, events)` is called after every committed
        hop, with the finished TimelineRow and any events it produced.

        It exists so a caller can watch a long run without waiting for it to
        finish, and because the per-hop timing is the number that decides
        whether this can run on a live call at all: a hop that takes longer
        than `cfg.hop_s` to compute means the pipeline is falling behind the
        audio arriving on the wire. Aggregate throughput hides that -- a run
        can be comfortably faster than real time on average while still
        blowing the budget on every hop that triggers speech recognition.
        """
        c = self.cfg
        result = CallResult(call_id=call_id, duration_s=0.0)
        frame_dt = source.frame_samples / float(source.sample_rate)
        t_s = 0.0

        logger.info("start call_id=%s sr=%d window=%.1fs hop=%.2fs model=%s asr=%s",
                    call_id, source.sample_rate, c.window_s, c.hop_s,
                    self.model.name, self.asr.backend.name)

        pending: List[np.ndarray] = []
        for frame in source.frames():
            self.buffer.write(frame.remote)
            self.agent_buffer.write(frame.agent)
            t_s = frame.t_s + frame_dt

            decision_audio = None
            if segmenter is not None:
                # Utterance mode: decide when the audio says a unit ended,
                # not when the clock says so.
                pending.append(frame.remote)
                utt = segmenter.push(frame.remote, t_s)
                if utt is None:
                    continue
                decision_audio = np.concatenate(pending) if pending else None
                pending = []
                self._last_hop_s = t_s
            else:
                if t_s - self._last_hop_s + 1e-9 < c.hop_s:
                    continue
                self._last_hop_s = t_s

            belief, events, timing = self.step(t_s, decision_audio)
            sub, why = self._last_explanation
            result.timeline.append(TimelineRow(
                t_s=round(t_s, 3), state=belief.state.value,
                confidence=round(belief.confidence, 4),
                probs={k: round(v, 4) for k, v in belief.probs.items()},
                speaker_id=self.speaker.registry.active_id,
                text=self.asr.recent_text(t_s, 4.0)[-120:],
                transfer_in_progress=self.transfers.in_progress,
                latency_ms=timing["total_ms"],
                sub_label=sub, evidence=why,
            ))
            result.events.extend(events)
            result.latency.append(timing)
            if on_hop is not None:
                on_hop(t_s, result.timeline[-1], timing, events)

        if segmenter is not None and pending:
            tail = segmenter.finish(t_s)
            if tail is not None:
                belief, events, timing = self.step(t_s, np.concatenate(pending))
                sub, why = self._last_explanation
                result.timeline.append(TimelineRow(
                    t_s=round(t_s, 3), state=belief.state.value,
                    confidence=round(belief.confidence, 4),
                    probs={k: round(v, 4) for k, v in belief.probs.items()},
                    speaker_id=self.speaker.registry.active_id,
                    text=self.asr.recent_text(t_s, 4.0)[-120:],
                    transfer_in_progress=self.transfers.in_progress,
                    latency_ms=timing["total_ms"], sub_label=sub, evidence=why))
                result.events.extend(events)
                result.latency.append(timing)
                if on_hop is not None:
                    on_hop(t_s, result.timeline[-1], timing, events)

        closing = self.transfers.close_open_transfer(t_s)
        if closing:
            result.events.append(closing)
            log_kv(logger, logging.INFO,
                   f"[event] {closing.type.value} @ {closing.t_s:.1f}s — {closing.evidence}",
                   event=closing.to_json())

        result.duration_s = t_s
        result.segments = segment_timeline(result.timeline, c.hop_s)
        result.summary = build_summary(result, self, source)
        logger.info("done call_id=%s duration=%.1fs hops=%d states=%s",
                    call_id, t_s, len(result.timeline),
                    result.summary.get("state_durations_s"))
        return result


def segment_timeline(rows: List[TimelineRow], hop_s: float = 0.0) -> List[Segment]:
    """
    Collapse per-hop rows into contiguous state runs.

    Sub-label and evidence are aggregated rather than sampled: the sub-label
    is the one held for the most hops in the run, and the evidence is taken
    from the most confident hop. Picking either from a single arbitrary hop
    (the first, or the midpoint) misrepresents the run whenever it is not
    uniform -- which is exactly when a reader most needs it to be right.

    `hop_s` extends each run to the end of its last hop. Without it a run's
    end is the *start* of its final decision, which makes a single-hop run
    zero seconds long and leaves a hop-sized hole between every pair of
    neighbours -- so the segments no longer tile the call, and anything that
    sums their durations quietly under-counts by one hop per segment.
    """
    from collections import Counter

    segs: List[Segment] = []
    for r in rows:
        if segs and segs[-1].state == r.state:
            s = segs[-1]
            s.end_s = r.t_s
            s.mean_confidence += r.confidence
            s._n = getattr(s, "_n", 1) + 1            # type: ignore[attr-defined]
            s._subs.append(r.sub_label)               # type: ignore[attr-defined]
            if r.confidence > s._best_conf:           # type: ignore[attr-defined]
                s._best_conf = r.confidence           # type: ignore[attr-defined]
                s.evidence = r.evidence
            if r.speaker_id and not s.speaker_id:
                s.speaker_id = r.speaker_id
        else:
            s = Segment(start_s=r.t_s, end_s=r.t_s, state=r.state,
                        speaker_id=r.speaker_id, mean_confidence=r.confidence,
                        sub_label=r.sub_label, evidence=r.evidence)
            s._n = 1                                  # type: ignore[attr-defined]
            s._subs = [r.sub_label]                   # type: ignore[attr-defined]
            s._best_conf = r.confidence               # type: ignore[attr-defined]
            segs.append(s)

    for s in segs:
        n = getattr(s, "_n", 1)
        s.mean_confidence = round(s.mean_confidence / n, 3)
        subs = [x for x in getattr(s, "_subs", []) if x]
        s.sub_label = Counter(subs).most_common(1)[0][0] if subs else None
        s.end_s = round(s.end_s + hop_s, 3)
    return segs


def build_summary(result: CallResult, engine: CallStateEngine, source) -> dict:
    durations: Dict[str, float] = {s.value: 0.0 for s in State.order()}
    for seg in result.segments:
        durations[seg.state] += seg.end_s - seg.start_s
    lat = [r["total_ms"] for r in result.latency] or [0.0]
    transfer_ends = [e for e in result.events if e.type.value == "transfer_end"]
    return {
        "call_id": result.call_id,
        "duration_s": round(result.duration_s, 2),
        "n_hops": len(result.timeline),
        "state_durations_s": {k: round(v, 1) for k, v in durations.items()},
        "dominant_state": max(durations, key=durations.get) if result.segments else "other",
        "n_state_changes": max(0, len(result.segments) - 1),
        "state_changes_per_hour": round(
            max(0, len(result.segments) - 1) / max(result.duration_s / 3600.0, 1e-9), 1),
        "n_speakers": len(engine.speaker.registry.centroids),
        "speakers": list(engine.speaker.registry.centroids.keys()),
        "n_transfers_started": sum(1 for e in result.events if e.type.value == "transfer_start"),
        "n_transfers_completed": sum(1 for e in transfer_ends if e.meta.get("outcome") == "completed"),
        "n_transfers_failed": sum(1 for e in transfer_ends if e.meta.get("outcome") == "failed"),
        "latency_ms": {
            "mean": round(float(np.mean(lat)), 2),
            "p50": round(float(np.percentile(lat, 50)), 2),
            "p95": round(float(np.percentile(lat, 95)), 2),
            "max": round(float(np.max(lat)), 2),
        },
        "hop_budget_ms": round(engine.cfg.hop_s * 1000, 1),
        "realtime_ok": bool(np.percentile(lat, 95) < engine.cfg.hop_s * 1000),
        "asr_backend": engine.asr.backend.name,
        "state_model": engine.model.name,
        "agent_channel_index": getattr(source, "agent_channel_index", -1),
        "config_fingerprint": engine.cfg.fingerprint(),
    }
