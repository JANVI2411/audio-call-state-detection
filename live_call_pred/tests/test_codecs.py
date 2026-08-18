"""WAV decoding: G.711 expansion, RIFF parsing, resampling."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from callstate.audio.codecs import (_ALAW, _MULAW, read_wav, resample_linear,  # noqa: E402
                                    write_wav_pcm16)

REAL_CALL = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "voice_agent", "input",
    "69f3a1e4a7da18e7ee83e734.beeped.wav")


class TestMuLaw(unittest.TestCase):
    def test_table_shape_and_range(self):
        self.assertEqual(_MULAW.shape, (256,))
        self.assertTrue(np.all(np.abs(_MULAW) <= 1.0))

    def test_zero_code_is_near_silence(self):
        # 0xFF is the mu-law code for zero amplitude.
        self.assertAlmostEqual(float(_MULAW[0xFF]), 0.0, places=3)

    def test_monotonic_within_sign(self):
        # Decoded magnitude must increase monotonically as the code walks the
        # positive branch; a broken exponent/mantissa split shows up here.
        pos = _MULAW[0x80:0x100]
        self.assertTrue(np.all(np.diff(pos) <= 1e-9) or np.all(np.diff(pos) >= -1e-9))

    def test_full_scale_codes_saturate(self):
        self.assertGreater(float(np.max(np.abs(_MULAW))), 0.9)

    def test_alaw_table_valid(self):
        self.assertEqual(_ALAW.shape, (256,))
        self.assertTrue(np.all(np.abs(_ALAW) <= 1.0))
        self.assertGreater(float(np.max(np.abs(_ALAW))), 0.9)


class TestRiff(unittest.TestCase):
    def test_pcm16_roundtrip(self):
        sr = 8000
        x = (0.5 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr)).astype(np.float32)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.wav")
            write_wav_pcm16(p, x, sr)
            w = read_wav(p)
        self.assertEqual(w.sample_rate, sr)
        self.assertEqual(w.n_channels, 1)
        self.assertEqual(len(w.samples), sr)
        self.assertLess(float(np.max(np.abs(w.samples[:, 0] - x))), 1e-3)

    def test_stereo_roundtrip_keeps_channels_separate(self):
        sr = 8000
        left = np.full(sr, 0.5, dtype=np.float32)
        right = np.full(sr, -0.25, dtype=np.float32)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "s.wav")
            write_wav_pcm16(p, np.stack([left, right], axis=1), sr)
            w = read_wav(p)
        self.assertEqual(w.n_channels, 2)
        self.assertAlmostEqual(float(w.samples[:, 0].mean()), 0.5, places=3)
        self.assertAlmostEqual(float(w.samples[:, 1].mean()), -0.25, places=3)

    def test_rejects_non_riff(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "bad.wav")
            with open(p, "wb") as fh:
                fh.write(b"NOTAWAVEFILE" + b"\x00" * 64)
            with self.assertRaises(ValueError):
                read_wav(p)


class TestRealCall(unittest.TestCase):
    """
    The reason this decoder exists: Python 3.13+ has no `audioop`, and stdlib
    `wave` raises `unknown format: 7` on exactly this file.
    """

    @unittest.skipUnless(os.path.exists(REAL_CALL), "real call audio not present")
    def test_decodes_real_mulaw_call(self):
        w = read_wav(REAL_CALL)
        self.assertEqual(w.n_channels, 2)
        self.assertEqual(w.sample_rate, 8000)
        self.assertGreater(w.duration_s, 60.0)
        self.assertTrue(np.all(np.abs(w.samples) <= 1.0))
        # Both legs must carry real signal — a decode bug typically yields
        # one silent channel or a DC offset.
        for c in range(2):
            rms = float(np.sqrt(np.mean(w.samples[:, c] ** 2)))
            self.assertGreater(rms, 1e-3, f"channel {c} is silent")
            self.assertLess(abs(float(np.mean(w.samples[:, c]))), 0.05,
                            f"channel {c} has a large DC offset")


class TestResample(unittest.TestCase):
    def test_length_and_content(self):
        sr_in, sr_out = 16000, 8000
        t = np.arange(sr_in) / sr_in
        x = np.sin(2 * np.pi * 200 * t).astype(np.float32)
        y = resample_linear(x, sr_in, sr_out)
        self.assertEqual(len(y), sr_out)
        self.assertLess(abs(float(np.sqrt(np.mean(y ** 2))) - 0.707), 0.05)

    def test_identity_when_rates_match(self):
        x = np.random.default_rng(0).normal(0, 1, 100).astype(np.float32)
        np.testing.assert_allclose(resample_linear(x, 8000, 8000), x)

    def test_empty_input(self):
        self.assertEqual(len(resample_linear(np.zeros(0), 8000, 16000)), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
