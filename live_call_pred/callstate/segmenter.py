"""
Cut the call at silence instead of on a fixed clock.

A fixed hop decides every N seconds whether or not anything happened, which
means it routinely decides in the middle of a word and splits one IVR prompt
across several decisions. Cutting at silence gives units that line up with
what was actually said: one prompt, one turn, one stretch of hold music.

Two things that improve for free:

  * speech recognition sees whole utterances rather than arbitrary slices, so
    it stops inventing words at the cut points;
  * the acoustic features describe one coherent thing instead of the tail of
    one event and the head of the next.

THE CATCH, and why `max_segment_s` exists
-----------------------------------------
A pure silence trigger cannot decide until the silence arrives. If a
representative talks for forty seconds you learn nothing for forty seconds,
and on a live call that is worse than a slightly ragged boundary. Hold music
makes it worse still: a music bed has no silence at all, so a pure trigger
would run to the end of the call as a single segment.

So the rule is silence OR a hard cap, whichever comes first. Natural
boundaries where the audio offers them, bounded latency always. The cap is
what makes this usable live rather than only on recordings.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional

import numpy as np

from .audio import features as F


@dataclass
class Utterance:
    start_s: float
    end_s: float
    reason: str          # "silence" | "max_length" | "end_of_call"
    speech_fraction: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass
class SilenceSegmenter:
    """
    Streaming segmenter. Feed it frames; it emits an Utterance when a cut
    point is reached.

    Causal by construction: a cut is emitted only once enough trailing silence
    has actually been observed, never by looking ahead for the next one.
    """

    sample_rate: int
    frame_ms: int = 20
    min_silence_ms: int = 400      # trailing quiet needed to call it a boundary
    min_segment_s: float = 0.6     # never emit a sliver
    max_segment_s: float = 8.0     # hard cap, bounds worst-case latency
    vad_rel_db: float = 9.0
    vad_percentile: float = 25.0

    def __post_init__(self) -> None:
        self._buf: List[np.ndarray] = []
        self._start_s = 0.0
        self._n_samples = 0
        self._silence_run_ms = 0
        self._speech_frames = 0
        self._total_frames = 0

    # -- internals ---------------------------------------------------------
    def _duration_s(self) -> float:
        return self._n_samples / float(self.sample_rate)

    def _flush(self, end_s: float, reason: str) -> Optional[Utterance]:
        if self._n_samples == 0:
            return None
        frac = (self._speech_frames / self._total_frames
                if self._total_frames else 0.0)
        u = Utterance(start_s=self._start_s, end_s=end_s, reason=reason,
                      speech_fraction=round(frac, 3))
        self._buf = []
        self._n_samples = 0
        self._silence_run_ms = 0
        self._speech_frames = 0
        self._total_frames = 0
        self._start_s = end_s
        return u

    # -- streaming API -----------------------------------------------------
    def push(self, frame: np.ndarray, t_end_s: float) -> Optional[Utterance]:
        """
        Add one frame. Returns an Utterance when this frame completes one.

        The silence test runs on the frame alone, against a floor taken from
        the segment so far -- a frame cannot be judged quiet or loud without
        something to compare it to, and the segment's own recent history is
        the only reference available causally.
        """
        self._buf.append(frame)
        self._n_samples += len(frame)
        self._total_frames += 1

        # Judge this frame against the segment's own energy, so the decision
        # tracks line noise rather than assuming a fixed level.
        recent = np.concatenate(self._buf[-100:]) if self._buf else frame
        voiced = F.vad(recent, self.sample_rate, self.frame_ms,
                       self.vad_percentile, self.vad_rel_db, min_run=1)
        is_speech = bool(voiced[-1]) if len(voiced) else False
        if is_speech:
            self._speech_frames += 1
            self._silence_run_ms = 0
        else:
            self._silence_run_ms += self.frame_ms

        dur = self._duration_s()
        if dur >= self.max_segment_s:
            return self._flush(t_end_s, "max_length")

        # Cut on silence only if this segment actually contained speech.
        # Without that test a silent stretch flushes, immediately re-triggers
        # on the very next quiet frame, and shatters into one segment per
        # `min_segment_s` -- on a real call that turned 30 seconds of hold
        # into fifty identical fragments. A stretch with no speech in it is
        # one thing (silence, or a music bed) and stays one segment until the
        # cap, or until speech starts it over.
        if (self._speech_frames > 0
                and self._silence_run_ms >= self.min_silence_ms
                and dur >= self.min_segment_s):
            return self._flush(t_end_s, "silence")

        # Speech beginning after a long quiet stretch is also a boundary: the
        # quiet part and the talking part are different things and should not
        # share a segment.
        if (is_speech and self._speech_frames == 1
                and dur - (self.frame_ms / 1000.0) >= self.min_segment_s):
            self._speech_frames = 0
            self._total_frames -= 1
            u = self._flush(t_end_s - self.frame_ms / 1000.0, "speech_onset")
            self._buf = [frame]
            self._n_samples = len(frame)
            self._total_frames = 1
            self._speech_frames = 1
            self._silence_run_ms = 0
            return u
        return None

    def finish(self, t_end_s: float) -> Optional[Utterance]:
        """Close whatever is open at end of call."""
        return self._flush(t_end_s, "end_of_call")


def segment_array(x: np.ndarray, sr: int, **kw) -> List[Utterance]:
    """Offline convenience: segment a whole array. Same rules as streaming."""
    seg = SilenceSegmenter(sample_rate=sr, **kw)
    n = int(sr * seg.frame_ms / 1000.0)
    out: List[Utterance] = []
    for i in range(0, len(x) - n + 1, n):
        u = seg.push(x[i:i + n], (i + n) / float(sr))
        if u:
            out.append(u)
    tail = seg.finish(len(x) / float(sr))
    if tail and tail.duration_s > 0:
        out.append(tail)
    return out
