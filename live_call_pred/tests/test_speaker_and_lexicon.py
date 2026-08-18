"""
Speaker branch and lexical branch — the two signals that make a transfer
detectable rather than guessed.
"""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from callstate.config import Config  # noqa: E402
from callstate.encoders.speaker import (SpeakerBranch, SpeakerRegistry,  # noqa: E402
                                        cosine, embed_mfcc)
from callstate.encoders.text_encoder import HashedNGramEncoder  # noqa: E402
from callstate.semantics.lexicon import score_text  # noqa: E402
from callstate.simulate import _formant_speech, _hold_music, _silence  # noqa: E402

SR = 8000


def voice(who: str, dur: float, seed: int) -> np.ndarray:
    r = np.random.default_rng(seed)
    f0, formants = (198, [660, 1900]) if who == "a" else (118, [480, 1420])
    return _formant_speech(dur, f0, 34, formants, 3.4, r)


class TestSpeakerEmbedding(unittest.TestCase):
    def test_same_speaker_similarity_far_above_different(self):
        """
        The threshold in Config is set from these numbers, so the numbers
        themselves are the test. Measured: same >= 0.997, different <= 0.751.
        """
        A = [embed_mfcc(voice("a", 3.0, s), SR) for s in range(5)]
        B = [embed_mfcc(voice("b", 3.0, s), SR) for s in range(5)]
        same = [cosine(A[i], A[j]) for i in range(5) for j in range(i + 1, 5)]
        same += [cosine(B[i], B[j]) for i in range(5) for j in range(i + 1, 5)]
        diff = [cosine(a, b) for a in A for b in B]
        thr = Config().speaker_similarity_threshold
        self.assertGreater(min(same), thr, "same-speaker pairs fall below threshold")
        self.assertLess(max(diff), thr, "different-speaker pairs clear threshold")
        self.assertGreater(min(same) - max(diff), 0.15, "margin has collapsed")

    def test_embedding_is_loudness_invariant(self):
        """
        Regression: keeping MFCC C0 (log frame energy) made the embedding
        mostly a volume measure, so the same person at two levels stopped
        matching. C0 is dropped for exactly this reason.
        """
        x = voice("a", 3.0, 42)
        self.assertGreater(cosine(embed_mfcc(x, SR), embed_mfcc(x * 0.25, SR)), 0.95)

    def test_music_is_far_from_any_speaker(self):
        m = embed_mfcc(_hold_music(3.0, np.random.default_rng(0)), SR)
        self.assertLess(cosine(m, embed_mfcc(voice("a", 3.0, 1), SR)),
                        Config().speaker_similarity_threshold)

    def test_degenerate_input_returns_zeros(self):
        self.assertFalse(np.any(embed_mfcc(np.zeros(100, dtype=np.float32), SR)))


class TestSpeakerRegistry(unittest.TestCase):
    def test_recognises_returning_speaker(self):
        reg = SpeakerRegistry(threshold=0.86, hysteresis=1)
        id1, _, new1 = reg.observe(embed_mfcc(voice("a", 3.0, 1), SR))
        id2, _, new2 = reg.observe(embed_mfcc(voice("a", 3.0, 2), SR))
        self.assertTrue(new1)
        self.assertFalse(new2)
        self.assertEqual(id1, id2)

    def test_distinguishes_second_speaker(self):
        reg = SpeakerRegistry(threshold=0.86, hysteresis=1)
        a, _, _ = reg.observe(embed_mfcc(voice("a", 3.0, 1), SR))
        b, _, new = reg.observe(embed_mfcc(voice("b", 3.0, 1), SR))
        self.assertTrue(new)
        self.assertNotEqual(a, b)
        self.assertEqual(len(reg.centroids), 2)

    def test_hysteresis_rejects_one_off_garbage_embedding(self):
        """
        The phantom-speaker bug: the first window after hold ends carries the
        tail of the music, producing an embedding matching nobody, which used
        to register a spurious third speaker. Requiring two consecutive
        agreeing observations suppresses it.
        """
        reg = SpeakerRegistry(threshold=0.86, hysteresis=2)
        reg.observe(embed_mfcc(voice("a", 3.0, 1), SR))
        self.assertEqual(len(reg.centroids), 1)
        garbage = embed_mfcc(_hold_music(3.0, np.random.default_rng(5)), SR)
        _id, _sim, new = reg.observe(garbage)
        self.assertFalse(new, "single anomalous window created a phantom speaker")
        self.assertEqual(len(reg.centroids), 1)
        # ... but a genuinely persistent new voice is still admitted
        reg.observe(embed_mfcc(voice("b", 3.0, 1), SR))
        reg.observe(embed_mfcc(voice("b", 3.0, 2), SR))
        self.assertEqual(len(reg.centroids), 2)


class TestSpeakerBranch(unittest.TestCase):
    def test_ignores_non_human_audio(self):
        """
        Running the registry on hold music pollutes it with non-speaker
        centroids and destroys the 'did the person change' signal.
        """
        br = SpeakerBranch(Config())
        obs = br.observe(_hold_music(3.0, np.random.default_rng(1)), SR,
                         speech_prob=0.9, is_human_like=False)
        self.assertIsNone(obs.embedding)
        self.assertEqual(len(br.registry.centroids), 0)

    def test_tracks_change_across_speakers(self):
        br = SpeakerBranch(Config())
        br.observe(voice("a", 3.0, 1), SR, 0.9, True)
        br.observe(voice("a", 3.0, 2), SR, 0.9, True)
        before = br.registry.active_id
        br.observe(voice("b", 3.0, 1), SR, 0.9, True)
        after = br.observe(voice("b", 3.0, 2), SR, 0.9, True)
        self.assertNotEqual(before, after.speaker_id)
        self.assertGreater(after.change_prob, 0.0)


class TestLexicon(unittest.TestCase):
    def test_ivr_prompts_score_ivr(self):
        for t in ["for eligibility and benefits press or say one",
                  "please enter the member identification number followed by the pound key",
                  "to repeat this menu press nine",
                  "thank you for calling the provider services line"]:
            self.assertGreater(score_text(t).ivr_prompt_prob, 0.5, t)

    def test_hold_scripts_score_hold_not_ivr(self):
        """
        'your call is important to us' is a hold-queue script, not a menu
        prompt. Having it in the IVR list made every hold segment with a
        spoken overlay classify as IVR.
        """
        s = score_text("thanks for holding your call is important to us please stay on the line")
        self.assertGreater(s.hold_phrase_prob, 0.5)
        self.assertLessEqual(s.ivr_prompt_prob, s.hold_phrase_prob)

    def test_recording_disclosure_still_scores_ivr(self):
        self.assertGreater(score_text("your call may be recorded for quality").ivr_prompt_prob, 0.5)

    def test_transfer_announcements(self):
        for t in ["okay let me get you over to claims please hold",
                  "i'll connect you with a specialist",
                  "transferring you now to billing",
                  "one moment while i transfer you"]:
            self.assertGreater(score_text(t).transfer_phrase_prob, 0.5, t)

    def test_transfer_failure_phrases(self):
        s = score_text("sorry they're not available right now so i'll go ahead and help you myself")
        self.assertGreater(s.transfer_fail_prob, 0.5)

    def test_spontaneous_speech_scores_human(self):
        s = score_text("uh yeah okay let me check that for you one second")
        self.assertGreater(s.human_spontaneous_prob, 0.3)
        self.assertLess(s.ivr_prompt_prob, 0.3)

    def test_ivr_prompt_is_not_spontaneous(self):
        s = score_text("for claims status press two")
        self.assertLess(s.human_spontaneous_prob, s.ivr_prompt_prob)

    def test_empty_text_is_all_zero(self):
        s = score_text("")
        self.assertEqual(s.ivr_prompt_prob, 0.0)
        self.assertEqual(s.transfer_phrase_prob, 0.0)
        self.assertEqual(s.human_spontaneous_prob, 0.0)

    def test_hits_are_reported_for_auditability(self):
        s = score_text("please hold while i transfer you to billing")
        self.assertIn("transfer", s.hits)
        self.assertTrue(s.hits["transfer"])


class TestTextEncoder(unittest.TestCase):
    def test_deterministic_and_normalised(self):
        enc = HashedNGramEncoder(dim=64)
        a, b = enc.encode("press one for claims"), enc.encode("press one for claims")
        np.testing.assert_allclose(a, b)
        self.assertAlmostEqual(float(np.linalg.norm(a)), 1.0, places=5)

    def test_empty_text_is_zero_vector(self):
        self.assertFalse(np.any(HashedNGramEncoder(dim=64).encode("")))

    def test_different_text_differs(self):
        enc = HashedNGramEncoder(dim=64)
        self.assertLess(float(np.dot(enc.encode("press one for claims"),
                                     enc.encode("uh yeah let me check that"))), 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestRealCallLexicon(unittest.TestCase):
    """
    Patterns written against the real payer call in this repo.

    The original IVR list assumed a keypad menu and matched almost none of it:
    that IVR is speech-driven and asks questions in full sentences. These
    lines are verbatim Whisper output from the counterparty leg.
    """

    IVR_LINES = [
        "Thank you for calling Blue Card Eligibility.",
        "If this is a medical emergency, please hang up and call 9-1-1.",
        "Calls may be monitored or recorded.",
        "This line is for health care providers to determine a patient's benefit eligibility",
        "Am I speaking with a health care provider?",
        "Does the member ID start with the letter R followed by numbers?",
        "What is the first three characters of the patient's member ID?",
        "Please say the letters, for example, FDJ.",
        "That was S-Z-M. Is that correct?",
        "Are you calling for pre-certification, benefit and eligibility, or both?",
        "Please hold while your call is being connected to the appropriate plan.",
        "Thank you for calling Highmark.",
    ]
    HUMAN_LINES = [
        "uh yeah hi this is brenda with provider services can i get the member id",
        "okay let me check that for you one second while i pull up the record",
        "yeah so the deductible is twenty five hundred and eleven hundred has been met",
        "sure no problem let me pull that up",
    ]

    def test_covers_the_real_ivr(self):
        missed = [t for t in self.IVR_LINES if score_text(t).ivr_prompt_prob <= 0.5]
        self.assertEqual(missed, [], f"real IVR lines not matched: {missed}")

    def test_does_not_fire_on_real_representatives(self):
        """
        A human rep says "member ID" constantly. A bare
        `(member|group|provider) (id|identification)` pattern fired IVR on
        live representatives, so only the framed variants are kept.
        """
        false_hits = [t for t in self.HUMAN_LINES if score_text(t).ivr_prompt_prob > 0.5]
        self.assertEqual(false_hits, [], f"IVR fired on human speech: {false_hits}")

    def test_punctuation_from_asr_does_not_break_matching(self):
        """
        Windowed decoding inserts punctuation at arbitrary pauses. Whisper
        returned "Thank you. for calling Blue Card Eligibility" — the period
        defeated `\\bthank you for calling\\b` until matching was normalised.
        """
        self.assertGreater(
            score_text("Thank you. for calling BlueCar. Blue Card Eligibility.").ivr_prompt_prob,
            0.5)

    def test_connection_line_is_both_ivr_and_hold(self):
        s = score_text("Please hold while your call is being connected to the appropriate plan.")
        self.assertGreater(s.ivr_prompt_prob, 0.5)
        self.assertGreater(s.hold_phrase_prob, 0.5)
