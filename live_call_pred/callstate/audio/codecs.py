"""
WAV reading for telephony audio, with no third-party audio dependency.

Python 3.13 removed `audioop`, and the stdlib `wave` module refuses G.711
files outright (`unknown format: 7`), which is exactly the format real
carrier recordings arrive in. So the RIFF parsing and the mu-law/A-law
expansion are done here directly against the spec. Verified against the
real 8 kHz stereo mu-law call in this repo — see tests/test_codecs.py.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Tuple

import numpy as np

WAVE_FORMAT_PCM = 0x0001
WAVE_FORMAT_IEEE_FLOAT = 0x0003
WAVE_FORMAT_ALAW = 0x0006
WAVE_FORMAT_MULAW = 0x0007
WAVE_FORMAT_EXTENSIBLE = 0xFFFE


@dataclass
class WavData:
    samples: np.ndarray  # float32, shape (n_samples, n_channels), range ~[-1, 1]
    sample_rate: int
    n_channels: int

    @property
    def duration_s(self) -> float:
        return len(self.samples) / float(self.sample_rate)


def _mulaw_table() -> np.ndarray:
    """
    G.711 mu-law decode table for all 256 byte values (ITU-T G.711).

    Built once as a lookup table rather than computed per sample: a 10-minute
    8 kHz call is ~4.8M samples per channel and a table lookup keeps decode
    time negligible relative to the rest of the pipeline.
    """
    codes = np.arange(256, dtype=np.int32)
    u = ~codes & 0xFF
    sign = u & 0x80
    exponent = (u >> 4) & 0x07
    mantissa = u & 0x0F
    magnitude = ((mantissa << 3) + 0x84) << exponent
    magnitude -= 0x84
    out = np.where(sign != 0, -magnitude, magnitude).astype(np.int32)
    return out.astype(np.float32) / 32768.0


def _alaw_table() -> np.ndarray:
    codes = np.arange(256, dtype=np.int32)
    a = codes ^ 0x55
    sign = a & 0x80
    exponent = (a >> 4) & 0x07
    mantissa = a & 0x0F
    magnitude = np.where(
        exponent == 0,
        (mantissa << 4) + 8,
        ((mantissa << 4) + 0x108) << (exponent - 1),
    ).astype(np.int32)
    out = np.where(sign != 0, magnitude, -magnitude).astype(np.int32)
    # A-law sign bit convention is inverted relative to mu-law.
    out = -out
    return out.astype(np.float32) / 32768.0


_MULAW = _mulaw_table()
_ALAW = _alaw_table()


def _parse_riff(raw: bytes) -> Tuple[dict, bytes]:
    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE file")
    pos = 12
    fmt = None
    data = None
    while pos + 8 <= len(raw):
        cid = raw[pos : pos + 4]
        (csize,) = struct.unpack("<I", raw[pos + 4 : pos + 8])
        body = raw[pos + 8 : pos + 8 + csize]
        if cid == b"fmt ":
            tag, ch, sr, _byte_rate, _align, bits = struct.unpack("<HHIIHH", body[:16])
            if tag == WAVE_FORMAT_EXTENSIBLE and len(body) >= 40:
                (tag,) = struct.unpack("<H", body[24:26])
            fmt = {"tag": tag, "channels": ch, "rate": sr, "bits": bits}
        elif cid == b"data":
            data = body
        pos += 8 + csize + (csize & 1)  # chunks are word-aligned
    if fmt is None or data is None:
        raise ValueError("WAVE file missing fmt or data chunk")
    return fmt, data


def read_wav(path: str) -> WavData:
    with open(path, "rb") as fh:
        raw = fh.read()
    fmt, data = _parse_riff(raw)
    tag, ch, sr, bits = fmt["tag"], fmt["channels"], fmt["rate"], fmt["bits"]

    if tag == WAVE_FORMAT_MULAW:
        flat = _MULAW[np.frombuffer(data, dtype=np.uint8)]
    elif tag == WAVE_FORMAT_ALAW:
        flat = _ALAW[np.frombuffer(data, dtype=np.uint8)]
    elif tag == WAVE_FORMAT_PCM and bits == 16:
        flat = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
    elif tag == WAVE_FORMAT_PCM and bits == 8:
        flat = (np.frombuffer(data, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif tag == WAVE_FORMAT_PCM and bits == 32:
        flat = np.frombuffer(data, dtype="<i4").astype(np.float32) / 2147483648.0
    elif tag == WAVE_FORMAT_IEEE_FLOAT and bits == 32:
        flat = np.frombuffer(data, dtype="<f4").astype(np.float32)
    else:
        raise ValueError(f"unsupported WAVE format tag={tag} bits={bits}")

    usable = (len(flat) // ch) * ch
    samples = flat[:usable].reshape(-1, ch).copy()
    return WavData(samples=samples, sample_rate=sr, n_channels=ch)


def write_wav_pcm16(path: str, samples: np.ndarray, sample_rate: int) -> None:
    """Write float samples as 16-bit PCM. Used by the synthetic call generator."""
    x = np.asarray(samples, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]
    n_ch = x.shape[1]
    pcm = np.clip(x, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2").tobytes()
    header = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVE"
    header += b"fmt " + struct.pack("<IHHIIHH", 16, WAVE_FORMAT_PCM, n_ch, sample_rate,
                                    sample_rate * n_ch * 2, n_ch * 2, 16)
    header += b"data" + struct.pack("<I", len(pcm))
    with open(path, "wb") as fh:
        fh.write(header + pcm)


def resample_linear(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """
    Linear resampling. Adequate here because we only ever go *down* to 8 kHz
    (the source is already band-limited by the telephony codec) or leave the
    rate alone. If a branch ever needs 16 kHz wideband input for a pretrained
    encoder, replace this with a polyphase filter (`scipy.signal.resample_poly`).
    """
    if sr_in == sr_out or len(x) == 0:
        return np.asarray(x, dtype=np.float32)
    n_out = int(round(len(x) * sr_out / float(sr_in)))
    if n_out <= 1:
        return np.zeros(max(n_out, 0), dtype=np.float32)
    src_idx = np.linspace(0.0, len(x) - 1.0, n_out)
    return np.interp(src_idx, np.arange(len(x)), np.asarray(x, dtype=np.float64)).astype(np.float32)
