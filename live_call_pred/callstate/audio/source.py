"""
Frame sources. Everything downstream consumes an iterator of fixed-size PCM
frames, so a recorded file and a live RTP/websocket leg are interchangeable.

Channel routing matters for this problem: on a stereo telephony recording,
one leg is our own agent and the other is the counterparty. Knowing when our
own agent is speaking is *privileged information* — it removes most of the
diarization ambiguity, because we only ever have to answer "what is the
remote side doing", never "who among everyone is talking". `StereoCallSource`
therefore exposes both legs, and the engine classifies only the remote one
while using the agent leg as a feature.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import numpy as np

from .codecs import read_wav, resample_linear


@dataclass
class Frame:
    t_s: float                    # start time of this frame within the call
    remote: np.ndarray            # counterparty PCM
    agent: np.ndarray             # our own agent's PCM (zeros if unavailable)


class FrameSource:
    sample_rate: int
    frame_samples: int

    def frames(self) -> Iterator[Frame]:  # pragma: no cover - interface
        raise NotImplementedError


def channel_activity(samples: np.ndarray, sr: int) -> List[dict]:
    """
    Per-channel activity statistics, measured against a threshold shared by
    all channels.

    The shared threshold is the whole point. An earlier version computed each
    channel's threshold from that channel's own energy percentile, which
    normalises away exactly the difference being measured: on a quiet leg the
    percentile collapses onto the noise floor, so line noise counts as
    "active" and the quiet leg reports a *higher* active fraction than the leg
    playing a continuous IVR menu. On the real call in this repo that inverted
    the result — it labelled the Blue Card IVR leg as our own agent, so the
    pipeline classified our agent's channel all call and read Whisper
    hallucinations ("You", "Yes. Yes.") off near-silence.

    An absolute threshold derived from the loudest channel keeps the
    comparison honest.
    """
    if samples.ndim == 1 or samples.shape[1] < 2:
        return []
    frame = max(1, sr // 50)
    n_ch = samples.shape[1]

    energies = []
    for c in range(n_ch):
        x = samples[:, c]
        n = (len(x) // frame) * frame
        energies.append(np.sqrt(np.mean(x[:n].reshape(-1, frame) ** 2, axis=1))
                        if n else np.zeros(0))

    loudest = max((float(np.percentile(e, 95)) if len(e) else 0.0) for e in energies)
    thr = max(loudest * 0.15, 1e-4)

    stats = []
    for c, energy in enumerate(energies):
        if not len(energy):
            stats.append({"channel": c, "active_fraction": 0.0, "mean_run_s": 0.0, "rms": 0.0})
            continue
        active = energy > thr
        runs, cur = [], 0
        for a in active:
            if a:
                cur += 1
            elif cur:
                runs.append(cur)
                cur = 0
        if cur:
            runs.append(cur)
        stats.append({
            "channel": c,
            "active_fraction": round(float(np.mean(active)), 4),
            "mean_run_s": round(float(np.mean(runs)) * frame / sr, 3) if runs else 0.0,
            "rms": round(float(np.sqrt(np.mean(samples[:, c] ** 2))), 5),
        })
    return stats


def detect_agent_channel(samples: np.ndarray, sr: int) -> int:
    """
    Guess which channel carries our outbound agent, from turn-taking shape.

    Our agent asks a question and waits, so it speaks in short, sparse bursts.
    The counterparty produces long continuous stretches — IVR menus, hold-music
    loops, a representative explaining benefits. So the channel with the lower
    active fraction *and* shorter mean run length is ours.

    Returns the channel index. `channel_activity` exposes the inputs so the
    decision is auditable, and `--agent-channel 0|1` overrides it. Getting this
    backwards silently ruins every downstream stage, so on a real deployment
    prefer taking the routing from the telephony layer, which knows for
    certain, over inferring it here.
    """
    stats = channel_activity(samples, sr)
    if not stats:
        return -1
    scores = [s["active_fraction"] + 0.05 * s["mean_run_s"] for s in stats]
    return int(np.argmin(scores))


class WavFileSource(FrameSource):
    """
    Reads a WAV off disk and hands out frames in call-time order.

    With `realtime=True` it sleeps between frames so the whole pipeline can be
    profiled under genuine wall-clock pressure; with `realtime=False` (default,
    used by tests and batch eval) it runs as fast as the CPU allows. Neither
    mode changes what any downstream stage sees — the contract is identical.
    """

    def __init__(
        self,
        path: str,
        target_sr: int = 8000,
        frame_ms: int = 20,
        agent_channel: str | int = "auto",
        realtime: bool = False,
        max_duration_s: float = 0.0,
    ):
        wav = read_wav(path)
        self.sample_rate = target_sr
        self.frame_samples = int(round(target_sr * frame_ms / 1000.0))
        self.realtime = realtime
        self.source_rate = wav.sample_rate
        self.n_channels = wav.n_channels

        if wav.n_channels >= 2:
            if agent_channel == "auto":
                idx = detect_agent_channel(wav.samples, wav.sample_rate)
            else:
                idx = int(agent_channel)
            self.agent_channel_index = idx
            remote_idx = 1 - idx if idx in (0, 1) else 0
            agent = wav.samples[:, idx] if idx >= 0 else np.zeros(len(wav.samples), np.float32)
            remote = wav.samples[:, remote_idx]
        else:
            # Mono: no free channel split. We treat the whole mix as remote and
            # lose the agent-leg privilege; the speaker branch has to carry more
            # weight. Flagged in the summary rather than silently assumed away.
            self.agent_channel_index = -1
            remote = wav.samples[:, 0]
            agent = np.zeros(len(remote), dtype=np.float32)

        self.remote = resample_linear(remote, wav.sample_rate, target_sr)
        self.agent = resample_linear(agent, wav.sample_rate, target_sr)
        if max_duration_s and max_duration_s > 0:
            # Truncation only, never sub-sampling: the prefix a live call would
            # have delivered by that time, so a partial run is identical to the
            # first N seconds of a full one.
            n = int(max_duration_s * target_sr)
            self.remote, self.agent = self.remote[:n], self.agent[:n]
        self.duration_s = len(self.remote) / float(target_sr)

    def frames(self) -> Iterator[Frame]:
        n = self.frame_samples
        total = len(self.remote) // n
        t0 = time.time()
        for i in range(total):
            s, e = i * n, (i + 1) * n
            t_s = s / float(self.sample_rate)
            if self.realtime:
                target = t0 + t_s
                delay = target - time.time()
                if delay > 0:
                    time.sleep(delay)
            yield Frame(t_s=t_s, remote=self.remote[s:e], agent=self.agent[s:e])


class ArraySource(FrameSource):
    """In-memory source — used by the synthetic generator and by tests."""

    def __init__(self, remote: np.ndarray, agent: Optional[np.ndarray] = None,
                 sample_rate: int = 8000, frame_ms: int = 20):
        self.sample_rate = sample_rate
        self.frame_samples = int(round(sample_rate * frame_ms / 1000.0))
        self.remote = np.asarray(remote, dtype=np.float32)
        self.agent = (np.asarray(agent, dtype=np.float32)
                      if agent is not None else np.zeros_like(self.remote))
        if len(self.agent) < len(self.remote):
            self.agent = np.pad(self.agent, (0, len(self.remote) - len(self.agent)))
        self.duration_s = len(self.remote) / float(sample_rate)
        self.agent_channel_index = -1

    def frames(self) -> Iterator[Frame]:
        n = self.frame_samples
        for i in range(len(self.remote) // n):
            s, e = i * n, (i + 1) * n
            yield Frame(t_s=s / float(self.sample_rate),
                        remote=self.remote[s:e], agent=self.agent[s:e])


class RingBuffer:
    """
    Fixed-capacity causal look-back over the remote channel.

    Capacity is the encoder context window. Writes are O(n) on the appended
    frame, reads return a contiguous copy of the most recent `capacity`
    samples. Nothing can ever read past the write head, which is what makes
    the "no look-ahead" claim structural instead of a promise — see
    tests/test_ringbuffer_causality.py.
    """

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self._buf = np.zeros(self.capacity, dtype=np.float32)
        self._filled = 0
        self.total_written = 0

    def write(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float32)
        n = len(x)
        self.total_written += n
        if n >= self.capacity:
            self._buf[:] = x[-self.capacity :]
            self._filled = self.capacity
            return
        self._buf[:-n] = self._buf[n:]
        self._buf[-n:] = x
        self._filled = min(self.capacity, self._filled + n)

    def read(self, n: Optional[int] = None) -> np.ndarray:
        n = self.capacity if n is None else min(int(n), self.capacity)
        n = min(n, self._filled)
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        return self._buf[-n:].copy()

    @property
    def filled(self) -> int:
        return self._filled
