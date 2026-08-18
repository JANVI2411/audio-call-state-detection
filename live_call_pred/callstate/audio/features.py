"""
Acoustic front-end: VAD, music/hold detection, tone & DTMF detection, pitch
variance, and a log-mel embedding.

Why classical DSP rather than a pretrained encoder by default: these run in
well under a millisecond per window on one CPU core with no model download,
which keeps the whole hop inside the 500 ms budget and keeps call audio
inside our own process. `encoders/audio_encoder.py` documents the swap to
WavLM/BEATs for the embedding branch when a GPU (or a slower budget) is
available — the interface is unchanged.

The important design point for *this* problem: these are multi-label signals,
not a classifier. "Music present" must not directly mean HOLD, because
"your call is important to us" is routinely spoken *over* hold music. The
fusion layer gets the raw evidence and decides.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from ..types import AudioFeatures

DTMF_LOW = [697.0, 770.0, 852.0, 941.0]
DTMF_HIGH = [1209.0, 1336.0, 1477.0, 1633.0]
DTMF_KEYS = [
    ["1", "2", "3", "A"],
    ["4", "5", "6", "B"],
    ["7", "8", "9", "C"],
    ["*", "0", "#", "D"],
]


def frame_signal(x: np.ndarray, sr: int, frame_ms: int) -> np.ndarray:
    n = max(1, int(round(sr * frame_ms / 1000.0)))
    total = (len(x) // n) * n
    if total == 0:
        return np.zeros((0, n), dtype=np.float32)
    return x[:total].reshape(-1, n)


def frame_energy_db(frames: np.ndarray) -> np.ndarray:
    if len(frames) == 0:
        return np.zeros(0, dtype=np.float32)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12)
    return (20.0 * np.log10(rms + 1e-12)).astype(np.float32)


def vad(x: np.ndarray, sr: int, frame_ms: int = 20,
        floor_percentile: float = 25.0, rel_db: float = 9.0,
        min_run: int = 3) -> np.ndarray:
    """
    Adaptive-floor energy VAD.

    The threshold is set relative to a low percentile of the window's own
    energy, so it tracks line noise instead of assuming a fixed level — line
    noise on a mu-law PSTN leg varies by tens of dB between carriers. A short
    run-length filter suppresses single-frame blips.

    Returns a bool array, one entry per frame.
    """
    frames = frame_signal(x, sr, frame_ms)
    if len(frames) == 0:
        return np.zeros(0, dtype=bool)
    db = frame_energy_db(frames)
    floor = np.percentile(db, floor_percentile)
    peak = np.percentile(db, 95.0)
    if peak - floor < 3.0:
        # Flat envelope: either pure silence or an unbroken steady tone/music.
        # Calling this "all speech" was a real bug on synthetic audio, so the
        # flat case is resolved by absolute level, not by the relative rule.
        return np.full(len(db), bool(peak > -45.0), dtype=bool)
    speech = db > (floor + rel_db)
    if min_run > 1:
        out = speech.copy()
        i = 0
        while i < len(speech):
            if speech[i]:
                j = i
                while j < len(speech) and speech[j]:
                    j += 1
                if (j - i) < min_run:
                    out[i:j] = False
                i = j
            else:
                i += 1
        speech = out
    return speech


ENV_RATE = 100  # Hz — amplitude envelope sampling rate for modulation analysis


def envelope(x: np.ndarray, sr: int, env_rate: int = ENV_RATE) -> np.ndarray:
    hop = max(1, sr // env_rate)
    n = (len(x) // hop) * hop
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    return np.sqrt(np.mean(x[:n].reshape(-1, hop).astype(np.float64) ** 2, axis=1))


@dataclass
class Modulation:
    syllable: float   # fraction of envelope-modulation energy at 2.5-8 Hz
    slow: float       # fraction at 0.05-1.2 Hz
    env_cv: float     # envelope coefficient of variation


def modulation_features(x: np.ndarray, sr: int) -> Modulation:
    """
    Modulation spectrum of the amplitude envelope — the single most useful
    speech-vs-music discriminator available at this cost.

    Speech is amplitude-modulated at the syllable rate, roughly 2.5-8 Hz, and
    that peak is remarkably consistent across speakers, languages and codecs.
    Music, hold loops and call-progress tones put their envelope energy far
    lower (bar-rate and slower) or nowhere at all.

    These numbers are measured, not assumed. On the synthetic corpus:

        kind          syllable   slow
        speech (IVR)     0.92     0.02
        speech (human)   0.86     0.06
        hold music       0.14     0.36
        ringback tone    0.02     0.73
        silence          0.07     0.02

    A prompt played *over* hold music lands in between (~0.48/0.20), which is
    correct — it genuinely is both, and the fusion layer should see that
    rather than be handed a forced choice.

    This replaced a long-lag envelope autocorrelation that did not work: at a
    6 s analysis window, a lag of 4 s is computed from 2 s of overlap, so it
    returned noise, and it scored speech (0.71-0.78) *higher* than hold music
    (0.55) — the opposite of the intent. Loop-level periodicity genuinely
    needs several loop periods of context; see `periodicity_score`.
    """
    e = envelope(x, sr)
    if len(e) < 16:
        return Modulation(0.0, 0.0, 0.0)
    mean_e = float(np.mean(e))
    env_cv = float(np.std(e) / (mean_e + 1e-9))
    d = e - mean_e
    if np.allclose(d, 0):
        return Modulation(0.0, 0.0, env_cv)
    spec = np.abs(np.fft.rfft(d * np.hanning(len(d)))) ** 2
    freqs = np.fft.rfftfreq(len(d), 1.0 / ENV_RATE)
    total = float(spec.sum()) + 1e-12
    syl = float(spec[(freqs >= 2.5) & (freqs <= 8.0)].sum() / total)
    slow = float(spec[(freqs > 0.05) & (freqs < 1.2)].sum() / total)
    return Modulation(syl, slow, env_cv)


def periodicity_score(x: np.ndarray, sr: int, min_period_s: float = 0.4,
                      max_period_s: float = 8.0) -> float:
    """
    Unbiased envelope autocorrelation over loop-scale lags.

    Two corrections over the naive version. The lag ceiling is capped at a
    third of the available signal, because a correlation computed from a
    handful of overlapping samples is noise that reliably reports ~0.8. And
    the correlation is normalised by the *overlap count* at each lag rather
    than by lag 0, which removes the systematic bias toward long lags.

    Even corrected, this only detects a loop when the buffer holds at least
    two periods of it — a 4 s loop needs an 8 s+ window. It is therefore used
    as weak corroboration, never as the primary hold signal; `modulation_
    features` carries that load.
    """
    e = envelope(x, sr)
    if len(e) < int(ENV_RATE * min_period_s * 3):
        return 0.0
    e = e - e.mean()
    if np.allclose(e, 0):
        return 0.0
    n = len(e)
    lo = int(min_period_s * ENV_RATE)
    hi = min(n // 3, int(max_period_s * ENV_RATE))
    if hi <= lo:
        return 0.0
    denom = float(np.dot(e, e)) + 1e-12
    best = 0.0
    for lag in range(lo, hi):
        overlap = n - lag
        c = float(np.dot(e[:overlap], e[lag:])) / (denom * overlap / n)
        best = max(best, c)
    return float(np.clip(best, 0.0, 1.0))


def spectral_stability(x: np.ndarray, sr: int, frame_ms: int = 32) -> float:
    """
    1 - mean spectral flux, normalised.

    Music and steady tones hold their spectral shape; conversational speech
    changes it constantly (phoneme to phoneme). High value => stable =>
    music/tone-like.
    """
    frames = frame_signal(x, sr, frame_ms)
    if len(frames) < 3:
        return 0.0
    win = np.hanning(frames.shape[1])
    mags = np.abs(np.fft.rfft(frames * win, axis=1))
    mags = mags / (np.sum(mags, axis=1, keepdims=True) + 1e-12)
    flux = np.sum(np.abs(np.diff(mags, axis=0)), axis=1)
    return float(np.clip(1.0 - np.mean(flux) / 0.6, 0.0, 1.0))


def tonality(x: np.ndarray, sr: int, n_peaks: int = 2) -> float:
    """
    Fraction of spectral energy concentrated in the two strongest narrow
    peaks — the signature of call-progress tones.

    Two peaks, not one, because every North American progress tone is a
    *pair*: ringback is 440+480 Hz, busy is 480+620, reorder is 480+620 at a
    faster cadence. Scoring only the single strongest peak splits the energy
    and badly under-reports them — measured on synthetic ringback, one peak
    gives 0.10 while two gives 0.24, against 0.09 for hold music and 0.03 for
    speech. Two peaks is what makes the threshold separable.
    """
    if len(x) < 256:
        return 0.0
    mag = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    total = float(mag.sum())
    if total <= 0:
        return 0.0
    work = mag.copy()
    acc = 0.0
    for _ in range(n_peaks):
        k = int(np.argmax(work))
        lo, hi = max(0, k - 3), min(len(work), k + 4)
        acc += float(work[lo:hi].sum())
        work[lo:hi] = 0.0
    return float(acc / total)


def _goertzel(x: np.ndarray, sr: int, freq: float) -> float:
    n = len(x)
    k = int(0.5 + (n * freq) / sr)
    w = (2.0 * np.pi * k) / n
    coeff = 2.0 * np.cos(w)
    s_prev = s_prev2 = 0.0
    for sample in x:
        s = sample + coeff * s_prev - s_prev2
        s_prev2, s_prev = s_prev, s
    return float(s_prev2 ** 2 + s_prev ** 2 - coeff * s_prev * s_prev2)


def detect_dtmf(x: np.ndarray, sr: int, block_ms: int = 40,
                min_inband_fraction: float = 0.55) -> List[Tuple[float, str]]:
    """
    Goertzel DTMF detection with a twist check and an in-band energy floor.

    Two guards matter, both added after real false positives on ordinary
    speech: (1) the two winning tones must dominate a fixed *fraction* of the
    block's total energy, not merely be the largest of the eight probes —
    voiced speech harmonics will always produce a "largest" bin; (2) the
    low/high pair must be within a 6:1 power ratio (the standard twist limit),
    which speech formants rarely satisfy.
    """
    block = max(1, int(sr * block_ms / 1000.0))
    hits: List[Tuple[float, str]] = []
    for i in range(0, max(0, len(x) - block + 1), block):
        blk = x[i : i + block]
        total = float(np.sum(blk.astype(np.float64) ** 2)) + 1e-12
        lo_p = [_goertzel(blk, sr, f) for f in DTMF_LOW]
        hi_p = [_goertzel(blk, sr, f) for f in DTMF_HIGH]
        li, hi_i = int(np.argmax(lo_p)), int(np.argmax(hi_p))
        pair = lo_p[li] + hi_p[hi_i]
        if pair / (total * len(blk) / 2.0) < min_inband_fraction:
            continue
        twist = (lo_p[li] + 1e-12) / (hi_p[hi_i] + 1e-12)
        if not (1 / 6.0 <= twist <= 6.0):
            continue
        hits.append((i / float(sr), DTMF_KEYS[li][hi_i]))
    return hits


def pitch_track(x: np.ndarray, sr: int, frame_ms: int = 32,
                fmin: float = 70.0, fmax: float = 350.0) -> np.ndarray:
    """Per-frame F0 by autocorrelation; NaN where the frame is unvoiced."""
    frames = frame_signal(x, sr, frame_ms)
    out = np.full(len(frames), np.nan, dtype=np.float32)
    if len(frames) == 0:
        return out
    lo, hi = int(sr / fmax), int(sr / fmin)
    for i, f in enumerate(frames):
        f = f - f.mean()
        if np.std(f) < 1e-4 or hi >= len(f):
            continue
        ac = np.correlate(f, f, mode="full")[len(f) - 1 :]
        if ac[0] <= 0:
            continue
        ac = ac / ac[0]
        seg = ac[lo:hi]
        if len(seg) == 0:
            continue
        k = int(np.argmax(seg))
        if seg[k] > 0.35:
            out[i] = sr / float(lo + k)
    return out


def pitch_variance(x: np.ndarray, sr: int, frame_ms: int = 32) -> float:
    """
    Coefficient of variation of F0 over voiced frames.

    Human spontaneous speech carries wide prosodic range; concatenated or
    TTS-generated IVR prompts are noticeably flatter. Returns -1.0 when there
    is not enough voiced signal to judge, which the fusion layer treats as
    "no evidence" rather than "flat".
    """
    f0 = pitch_track(x, sr, frame_ms)
    voiced = f0[~np.isnan(f0)]
    if len(voiced) < 4:
        return -1.0
    return float(np.std(voiced) / (np.mean(voiced) + 1e-9))


def _mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel2hz(m):
        return 700.0 * (10 ** (m / 2595.0) - 1.0)

    fmax = sr / 2.0
    pts = mel2hz(np.linspace(hz2mel(50.0), hz2mel(fmax), n_mels + 2))
    bins = np.floor((n_fft + 1) * pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(n_mels):
        l, c, r = bins[m], bins[m + 1], bins[m + 2]
        if c == l:
            c = l + 1
        if r == c:
            r = c + 1
        r = min(r, fb.shape[1] - 1)
        c = min(c, r)
        for k in range(l, c):
            fb[m, k] = (k - l) / max(1, (c - l))
        for k in range(c, r):
            fb[m, k] = (r - k) / max(1, (r - c))
    return fb


_FB_CACHE: dict = {}


def logmel(x: np.ndarray, sr: int, n_mels: int = 24, frame_ms: int = 32) -> np.ndarray:
    """Log-mel spectrogram, shape (n_frames, n_mels)."""
    frames = frame_signal(x, sr, frame_ms)
    if len(frames) == 0:
        return np.zeros((0, n_mels), dtype=np.float32)
    n_fft = frames.shape[1]
    key = (sr, n_fft, n_mels)
    if key not in _FB_CACHE:
        _FB_CACHE[key] = _mel_filterbank(sr, n_fft, n_mels)
    fb = _FB_CACHE[key]
    win = np.hanning(n_fft)
    power = np.abs(np.fft.rfft(frames * win, axis=1)) ** 2
    return np.log(power @ fb.T + 1e-10).astype(np.float32)


def extract(x: np.ndarray, sr: int, cfg, x_long: Optional[np.ndarray] = None) -> AudioFeatures:
    """
    One window in, one multi-label acoustic observation out.

    Two windows, actually, and the reason is a genuine timescale conflict.
    Boundary responsiveness wants a *short* decision window: whatever stale
    audio the decision can still see is a lower bound on how late a state
    change is noticed. But music detection depends on envelope modulation
    down at 0.05-1.2 Hz, and a 2.5 s window has 0.4 Hz frequency resolution,
    so the band that identifies hold music is not measurable in it at all.
    Using one short window for everything drove hold recall to exactly zero —
    every hold frame was misread — while using one long window for everything
    put boundary detection ~1.5 s late.

    So: the slow (music) band is measured over `x_long`, the full encoder
    window, because that property genuinely needs the time. Everything
    responsive — VAD, syllable-rate modulation, tone, energy — is measured
    over the short window `x`. Speech starting is visible immediately;
    "we are in a music bed" is allowed to take longer to establish, which
    matches how the two phenomena actually behave.

    The probabilities are deliberately *soft and overlapping*. A window can be
    0.7 music and 0.4 speech at the same time — that is what a recorded "your
    call is important to us" over hold music actually is. Collapsing it to a
    single label here would discard exactly the evidence the fusion layer
    needs, and it is why this function returns a set of independent scores
    rather than a classification.

    Each score's primary evidence:
      speech  VAD voiced fraction, gated by syllable-rate modulation, so
              sustained non-speech energy (a tone, a music bed) does not
              register as speech merely by being loud
      music   slow envelope modulation with the syllable band absent
      tone    energy concentrated in two narrow spectral peaks
      silence low absolute energy
    """
    if len(x) < sr // 10:
        z = np.zeros(2 * cfg.n_mels, dtype=np.float32)
        return AudioFeatures(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, z, 0.0, 0.0)

    long_x = x if x_long is None or len(x_long) < len(x) else x_long

    v = vad(x, sr, cfg.frame_ms, cfg.vad_energy_percentile,
            cfg.vad_rel_threshold_db, cfg.vad_min_speech_frames)
    voiced_frac = float(np.mean(v)) if len(v) else 0.0
    mod = modulation_features(x, sr)              # syllable band: short window
    mod_long = modulation_features(long_x, sr)    # slow/music band: long window
    per = periodicity_score(long_x, sr)
    stab = spectral_stability(x, sr)
    # Tone over the long window: a call-progress tone is defined by its
    # on/off cadence (ringback is ~2 s on, ~4 s off), so a 2.5 s decision
    # window can land entirely in the silent phase and score zero. The
    # cadence needs more than one cycle to be visible.
    tone = tonality(long_x, sr)
    pv = pitch_variance(x, sr)

    lm = logmel(x, sr, cfg.n_mels)
    emb = (np.concatenate([lm.mean(axis=0), lm.std(axis=0)]).astype(np.float32)
           if len(lm) else np.zeros(2 * cfg.n_mels, dtype=np.float32))

    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    energy_present = float(np.clip(rms / 0.015, 0.0, 1.0))

    # Speech evidence from two independent sources, combined rather than
    # chained. An energy VAD needs the speech to stand out from the floor, so
    # it fails completely on a prompt spoken over a continuous hold-music bed
    # — measured voiced fraction 0.00 while syllable modulation read 0.62,
    # i.e. the speech was plainly there and the VAD could not see it. Because
    # syllable modulation is an energy *fraction*, it is unaffected by a
    # raised floor, so it covers exactly the case the VAD misses. Chaining
    # them (vad * syllable) would let either one veto; averaging lets either
    # one carry the evidence.
    syl_ness = float(np.clip(mod.syllable / 0.45, 0.0, 1.0))
    speech = 0.5 * voiced_frac * (0.25 + 0.75 * syl_ness) + 0.5 * syl_ness * energy_present

    slow_ness = float(np.clip(mod_long.slow / 0.30, 0.0, 1.0))
    music = float(np.clip(0.85 * slow_ness + 0.45 * (1.0 - syl_ness) - 0.30, 0.0, 1.0))
    music *= energy_present
    if tone > cfg.tone_min_inband_fraction:
        # Two pure tones with a slow on/off cadence is call progress
        # (ringback/busy), not a music bed. Suppress rather than zero: some
        # hold loops really are close to pure tones.
        music *= 0.35

    return AudioFeatures(
        speech_prob=float(np.clip(speech, 0.0, 1.0)),
        music_prob=music,
        silence_prob=float(np.clip(1.0 - energy_present, 0.0, 1.0)),
        tone_prob=tone,
        periodicity=per,
        spectral_stability=stab,
        pitch_cv=pv,
        embedding=emb,
        syllable_mod=mod.syllable,
        slow_mod=mod_long.slow,
    )


def modulation_scalars(x: np.ndarray, sr: int) -> Tuple[float, float, float]:
    """Convenience accessor used by the featurizer (syllable, slow, env_cv)."""
    m = modulation_features(x, sr)
    return m.syllable, m.slow, m.env_cv
