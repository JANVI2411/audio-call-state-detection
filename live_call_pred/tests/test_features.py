"""
Acoustic front-end. These tests pin the *separations* the fusion layer
depends on, not exact values — a feature is only useful here if speech and
music land on opposite sides of it.
"""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from callstate.audio import features as F  # noqa: E402
from callstate.config import Config  # noqa: E402
from callstate.simulate import (_formant_speech, _hold_music, _silence,  # noqa: E402
                                _tone)

SR = 8000


def rng(seed=0):
    return np.random.default_rng(seed)


class TestVAD(unittest.TestCase):
    def test_detects_speech_and_rejects_silence(self):
        speech = _formant_speech(4.0, 180, 30, [600, 1700], 3.6, rng())
        quiet = _silence(4.0, rng())
        self.assertGreater(float(np.mean(F.vad(speech, SR))), 0.25)
        self.assertLess(float(np.mean(F.vad(quiet, SR))), 0.05)

    def test_flat_envelope_resolved_by_absolute_level(self):
        """
        Regression: a perfectly flat envelope used to fall through the
        relative-threshold rule and be marked entirely as speech. Flat and
        loud is plausible signal; flat and near-zero is not.
        """
        loud_flat = np.full(SR * 2, 0.3, dtype=np.float32)
        quiet_flat = np.full(SR * 2, 1e-5, dtype=np.float32)
        self.assertTrue(bool(np.all(F.vad(loud_flat, SR))))
        self.assertFalse(bool(np.any(F.vad(quiet_flat, SR))))

    def test_short_blips_filtered(self):
        x = np.zeros(SR, dtype=np.float32)
        x[4000:4160] = 0.5  # 20 ms burst, below min_run
        self.assertLess(int(np.sum(F.vad(x, SR, min_run=5))), 3)

    def test_empty_input(self):
        self.assertEqual(len(F.vad(np.zeros(0), SR)), 0)


class TestModulation(unittest.TestCase):
    """The speech-vs-music discriminator. If this ordering breaks, hold breaks."""

    def setUp(self):
        self.speech = _formant_speech(6.0, 180, 30, [600, 1700], 3.6, rng(1))
        self.ivr = _formant_speech(6.0, 165, 4, [610, 1750], 4.2, rng(2))
        self.music = _hold_music(6.0, rng(3))
        self.tone = _tone(6.0, [440, 480])

    def test_speech_has_high_syllable_modulation(self):
        for name, sig in (("human", self.speech), ("ivr", self.ivr)):
            m = F.modulation_features(sig, SR)
            self.assertGreater(m.syllable, 0.45, f"{name} syllable modulation too low")

    def test_music_and_tone_have_low_syllable_modulation(self):
        for name, sig in (("music", self.music), ("tone", self.tone)):
            m = F.modulation_features(sig, SR)
            self.assertLess(m.syllable, 0.35, f"{name} looks speech-modulated")

    def test_music_dominates_slow_band_over_speech(self):
        self.assertGreater(F.modulation_features(self.music, SR).slow,
                           F.modulation_features(self.speech, SR).slow)

    def test_separation_is_wide(self):
        speech_syl = F.modulation_features(self.speech, SR).syllable
        music_syl = F.modulation_features(self.music, SR).syllable
        self.assertGreater(speech_syl - music_syl, 0.3,
                           "speech/music separation has collapsed")

    def test_degenerate_inputs(self):
        for sig in (np.zeros(SR), np.zeros(4), np.full(SR, 0.2, dtype=np.float32)):
            m = F.modulation_features(np.asarray(sig, dtype=np.float32), SR)
            self.assertTrue(np.isfinite(m.syllable) and np.isfinite(m.slow))


class TestTonality(unittest.TestCase):
    def test_ringback_above_music_above_speech(self):
        tone = F.tonality(_tone(6.0, [440, 480]), SR)
        music = F.tonality(_hold_music(6.0, rng(4)), SR)
        speech = F.tonality(_formant_speech(6.0, 180, 30, [600, 1700], 3.6, rng(5)), SR)
        self.assertGreater(tone, music)
        self.assertGreater(music, speech)

    def test_threshold_separates_tone_from_music(self):
        """
        Checked over many seeds, not one: a threshold justified by a single
        draw is a coincidence. Measured over 6 s windows the bands are
        speech 0.008-0.016, music 0.073-0.101, ringback 0.202-0.321.
        """
        cfg = Config()
        for s in range(8):
            music = F.tonality(_hold_music(6.0, rng(100 + s)), SR)
            tone = F.tonality(_tone(6.0, [440, 480]), SR)
            self.assertLess(music, cfg.tone_min_inband_fraction, f"music seed {s}")
            self.assertGreater(tone, cfg.tone_min_inband_fraction)


class TestPitch(unittest.TestCase):
    def test_flat_ivr_below_expressive_human(self):
        flat = F.pitch_variance(_formant_speech(5.0, 165, 3, [610, 1750], 4.2, rng(7)), SR)
        varied = F.pitch_variance(_formant_speech(5.0, 190, 40, [660, 1900], 3.4, rng(8)), SR)
        self.assertGreaterEqual(flat, 0.0)
        self.assertGreaterEqual(varied, 0.0)
        self.assertLess(flat, varied)

    def test_reports_no_evidence_on_silence(self):
        self.assertEqual(F.pitch_variance(_silence(3.0, rng(9)), SR), -1.0)


class TestDTMF(unittest.TestCase):
    def _digit(self, low, high, dur=0.25):
        t = np.arange(int(dur * SR)) / SR
        return (0.5 * (np.sin(2 * np.pi * low * t) + np.sin(2 * np.pi * high * t))
                ).astype(np.float32)

    def test_detects_digit_one(self):
        hits = F.detect_dtmf(self._digit(697, 1209), SR)
        self.assertTrue(hits)
        self.assertEqual(hits[0][1], "1")

    def test_detects_digit_pound(self):
        hits = F.detect_dtmf(self._digit(941, 1477), SR)
        self.assertTrue(hits)
        self.assertEqual(hits[0][1], "#")

    def test_no_false_positive_on_speech(self):
        """
        Voiced speech always has *some* energy at every probe frequency, so a
        naive "largest of eight bins wins" detector fires constantly on
        conversation. The in-band energy fraction and twist guards exist for
        this case.
        """
        speech = _formant_speech(4.0, 180, 30, [600, 1700], 3.6, rng(10))
        self.assertEqual(len(F.detect_dtmf(speech, SR)), 0)

    def test_no_false_positive_on_music_or_silence(self):
        self.assertEqual(len(F.detect_dtmf(_hold_music(3.0, rng(11)), SR)), 0)
        self.assertEqual(len(F.detect_dtmf(_silence(3.0, rng(12)), SR)), 0)


class TestExtract(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()

    def _extract(self, sig):
        return F.extract(sig[-int(self.cfg.decision_window_s * SR):], SR, self.cfg,
                         x_long=sig)

    def test_speech_scores_speech_not_music(self):
        af = self._extract(_formant_speech(6.0, 185, 32, [640, 1800], 3.5, rng(13)))
        self.assertGreater(af.speech_prob, 0.3)
        self.assertLess(af.music_prob, 0.35)

    def test_music_scores_music_not_speech(self):
        af = self._extract(_hold_music(6.0, rng(14)))
        self.assertGreater(af.music_prob, 0.4)
        self.assertLess(af.speech_prob, 0.3)

    def test_ringback_scores_tone(self):
        # 8 s so the long window spans a full on/off cadence cycle.
        af = self._extract(_tone(8.0, [440, 480], on_s=2.0, off_s=4.0))
        self.assertGreater(af.tone_prob, self.cfg.tone_min_inband_fraction)
        self.assertLess(af.music_prob, 0.6)

    def test_silence_scores_silence(self):
        af = self._extract(_silence(6.0, rng(15)))
        self.assertGreater(af.silence_prob, 0.7)
        self.assertLess(af.speech_prob, 0.1)

    def test_prompt_over_music_is_multi_label(self):
        """
        The case a flat classifier cannot express. A recorded prompt playing
        over a hold bed genuinely is both, and the front-end must report both
        rather than pick one.
        """
        music = _hold_music(6.0, rng(16)).copy()
        speech = _formant_speech(4.0, 160, 5, [600, 1700], 4.0, rng(17))
        # Overlay at the END of the clip: the decision window is the trailing
        # slice, so a prompt that finished seconds ago is correctly no longer
        # "currently speaking". This is what we want tested — a prompt playing
        # *now* over a music bed.
        music[-len(speech):] += speech * 0.9
        af = self._extract(music)
        self.assertGreater(af.speech_prob, 0.05)
        self.assertGreater(af.music_prob, 0.05)

    def test_embedding_shape_and_finiteness(self):
        af = self._extract(_formant_speech(6.0, 180, 30, [600, 1700], 3.6, rng(18)))
        self.assertEqual(af.embedding.shape, (2 * self.cfg.n_mels,))
        self.assertTrue(np.all(np.isfinite(af.embedding)))

    def test_tiny_input_degrades_gracefully(self):
        af = F.extract(np.zeros(10, dtype=np.float32), SR, self.cfg)
        self.assertEqual(af.silence_prob, 1.0)
        self.assertEqual(af.pitch_cv, -1.0)


class TestPeriodicity(unittest.TestCase):
    def test_returns_zero_when_window_too_short_for_a_loop(self):
        """
        The corrected version must refuse to report a loop it cannot see,
        rather than returning inflated correlations from a few overlapping
        samples — the bug that made speech score higher than hold music.
        """
        short = _hold_music(1.0, rng(19))
        self.assertLessEqual(F.periodicity_score(short, SR, min_period_s=0.4), 1.0)
        self.assertEqual(F.periodicity_score(np.zeros(100, dtype=np.float32), SR), 0.0)

    def test_bounded(self):
        for sig in (_hold_music(12.0, rng(20)),
                    _formant_speech(12.0, 180, 30, [600, 1700], 3.6, rng(21))):
            p = F.periodicity_score(sig, SR)
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
