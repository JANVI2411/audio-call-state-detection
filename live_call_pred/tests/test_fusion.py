"""
Fusion layer: HMM filtering, the state tracker, the state models, and the
featurizer contract.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from callstate.config import Config  # noqa: E402
from callstate.fusion.featurizer import EMB_DIM, SCALAR_NAMES, Featurizer  # noqa: E402
from callstate.fusion.hmm import HMMFilter, build_transition_matrix  # noqa: E402
from callstate.fusion.model import (LogisticStateModel, PriorStateModel,  # noqa: E402
                                    softmax)
from callstate.fusion.state_tracker import StateTracker  # noqa: E402
from callstate.types import (AudioFeatures, Observation, SemanticObs,  # noqa: E402
                             SpeakerObs, State)

IVR, HUMAN, HOLD, OTHER = (State.index(s) for s in ["ivr", "human", "hold", "other"])


def make_obs(t_s=1.0, **over) -> Observation:
    cfg = Config()
    a = AudioFeatures(
        speech_prob=over.get("speech_prob", 0.6), music_prob=over.get("music_prob", 0.0),
        silence_prob=over.get("silence_prob", 0.2), tone_prob=over.get("tone_prob", 0.0),
        periodicity=0.1, spectral_stability=0.3, pitch_cv=over.get("pitch_cv", 0.2),
        embedding=np.zeros(2 * cfg.n_mels, dtype=np.float32),
        syllable_mod=over.get("syllable_mod", 0.8), slow_mod=over.get("slow_mod", 0.05),
    )
    s = SpeakerObs(None, over.get("speaker_change_prob", 0.0), 0.9,
                   over.get("speaker_id", None), False)
    m = SemanticObs(
        text=over.get("text", "hello"), asr_confidence=0.9, word_rate=2.0,
        ivr_prompt_prob=over.get("ivr_prompt_prob", 0.0),
        transfer_phrase_prob=over.get("transfer_phrase_prob", 0.0),
        hold_phrase_prob=over.get("hold_phrase_prob", 0.0),
        human_spontaneous_prob=over.get("human_spontaneous_prob", 0.0),
        text_embedding=np.zeros(64, dtype=np.float32),
    )
    return Observation(t_s=t_s, window_start_s=max(0.0, t_s - 6),
                       audio=a, speaker=s, semantic=m,
                       telephony={"agent_recently_spoke": 0.0, "time_since_agent_spoke": 30.0,
                                  "dtmf_recent": 0.0, "sip_leg_changed": 0.0},
                       history={"prev_state_is_hold": 0.0, "dwell_s": 5.0})


class TestTransitionMatrix(unittest.TestCase):
    def setUp(self):
        c = Config()
        self.A = build_transition_matrix(c.hmm_self_prob, c.hmm_transition_bias)

    def test_rows_are_probability_distributions(self):
        np.testing.assert_allclose(self.A.sum(axis=1), 1.0, atol=1e-9)
        self.assertTrue(np.all(self.A >= 0))

    def test_self_transitions_dominate(self):
        for i in range(len(State.order())):
            self.assertEqual(int(np.argmax(self.A[i])), i)

    def test_encodes_call_structure_not_symmetry(self):
        # Hold -> human is the common exit from hold; human -> ivr is rare.
        self.assertGreater(self.A[HOLD, HUMAN], self.A[HOLD, IVR])
        self.assertGreater(self.A[HUMAN, HOLD], self.A[HUMAN, IVR])


class TestHMMFilter(unittest.TestCase):
    def _filter(self):
        c = Config()
        return HMMFilter(build_transition_matrix(c.hmm_self_prob, c.hmm_transition_bias))

    def test_posterior_normalised(self):
        f = self._filter()
        for _ in range(5):
            p = f.step(np.array([0.4, 0.3, 0.2, 0.1]))
            self.assertAlmostEqual(float(p.sum()), 1.0, places=9)

    def test_suppresses_single_frame_flicker(self):
        """
        The core reason the HMM exists: one contradictory observation inside a
        run of consistent ones must not flip the reported state.
        """
        f = self._filter()
        ivr_ev = np.array([0.85, 0.05, 0.05, 0.05])
        blip = np.array([0.05, 0.85, 0.05, 0.05])
        for _ in range(8):
            f.step(ivr_ev)
        p = f.step(blip)
        self.assertEqual(int(np.argmax(p)), IVR, "one blip flipped the state")

    def test_accepts_a_sustained_change(self):
        f = self._filter()
        for _ in range(8):
            f.step(np.array([0.85, 0.05, 0.05, 0.05]))
        for _ in range(8):
            p = f.step(np.array([0.05, 0.85, 0.05, 0.05]))
        self.assertEqual(int(np.argmax(p)), HUMAN, "sustained evidence never took effect")

    def test_reset_restores_prior(self):
        f = self._filter()
        for _ in range(10):
            f.step(np.array([0.9, 0.04, 0.03, 0.03]))
        f.reset()
        np.testing.assert_allclose(f.posterior, f.prior, atol=1e-9)

    def test_handles_zero_probability_emissions(self):
        f = self._filter()
        p = f.step(np.array([0.0, 0.0, 1.0, 0.0]))
        self.assertTrue(np.all(np.isfinite(p)))
        self.assertAlmostEqual(float(p.sum()), 1.0, places=9)


class TestStateTracker(unittest.TestCase):
    def test_holds_state_when_leader_is_weak(self):
        tr = StateTracker(cfg=Config())
        tr.commit(1.0, np.array([0.9, 0.04, 0.03, 0.03]), np.array([0.9, 0.04, 0.03, 0.03]))
        self.assertEqual(tr.state, State.IVR)
        weak = np.array([0.30, 0.38, 0.17, 0.15])  # leader below commit_min_prob
        b = tr.commit(1.5, weak, weak)
        self.assertEqual(b.state, State.IVR, "committed on a weak, ambiguous leader")

    def test_switches_on_a_confident_leader(self):
        tr = StateTracker(cfg=Config())
        tr.commit(1.0, np.array([0.9, 0.04, 0.03, 0.03]), np.array([0.9, 0.04, 0.03, 0.03]))
        strong = np.array([0.05, 0.85, 0.05, 0.05])
        self.assertEqual(tr.commit(1.5, strong, strong).state, State.HUMAN)

    def test_dwell_tracks_time_in_state(self):
        tr = StateTracker(cfg=Config())
        strong_ivr = np.array([0.9, 0.04, 0.03, 0.03])
        tr.commit(1.0, strong_ivr, strong_ivr)
        tr.commit(5.0, strong_ivr, strong_ivr)
        self.assertAlmostEqual(tr.dwell_s(9.0), 8.0, places=6)

    def test_previous_state_reported(self):
        tr = StateTracker(cfg=Config())
        ivr = np.array([0.9, 0.04, 0.03, 0.03])
        human = np.array([0.04, 0.9, 0.03, 0.03])
        tr.commit(1.0, ivr, ivr)
        tr.commit(2.0, human, human)
        self.assertEqual(tr.previous_state(), State.IVR)


class TestPriorModel(unittest.TestCase):
    def setUp(self):
        self.m = PriorStateModel()
        self.f = Featurizer(Config(), 48, 64)

    def _predict(self, **over):
        obs = make_obs(**over)
        return self.m.predict_proba_scalars(self.f.scalars(obs))

    def test_ivr_prompt_language_selects_ivr(self):
        p = self._predict(ivr_prompt_prob=0.95, pitch_cv=0.03, human_spontaneous_prob=0.0)
        self.assertEqual(int(np.argmax(p)), IVR)

    def test_spontaneous_speech_selects_human(self):
        p = self._predict(human_spontaneous_prob=0.9, pitch_cv=0.35, ivr_prompt_prob=0.0)
        self.assertEqual(int(np.argmax(p)), HUMAN)

    def test_music_selects_hold(self):
        p = self._predict(music_prob=0.9, slow_mod=0.4, syllable_mod=0.1,
                          speech_prob=0.05, text="", hold_phrase_prob=0.6)
        self.assertEqual(int(np.argmax(p)), HOLD)

    def test_tone_selects_other(self):
        p = self._predict(tone_prob=0.4, speech_prob=0.05, syllable_mod=0.02,
                          silence_prob=0.6, text="", music_prob=0.05)
        self.assertEqual(int(np.argmax(p)), OTHER)

    def test_probabilities_are_valid(self):
        p = self._predict()
        self.assertAlmostEqual(float(p.sum()), 1.0, places=6)
        self.assertTrue(np.all(p >= 0))

    def test_explain_attributes_the_decision(self):
        obs = make_obs(ivr_prompt_prob=0.95)
        ex = self.m.explain(self.f.scalars(obs))
        self.assertIn("ivr", ex)
        top = [name for name, _ in ex["ivr"]]
        self.assertIn("ivr_prompt_prob", top)


class TestLogisticModel(unittest.TestCase):
    def _toy(self, n=400, dim=12, seed=0):
        rng = np.random.default_rng(seed)
        y = rng.integers(0, 4, n)
        centres = rng.normal(0, 3, (4, dim))
        X = centres[y] + rng.normal(0, 0.5, (n, dim))
        return X.astype(np.float32), y

    def test_fits_separable_data(self):
        X, y = self._toy()
        m = LogisticStateModel(dim=X.shape[1])
        stats = m.fit(X, y, epochs=400, lr=0.5)
        self.assertGreater(stats["train_accuracy"], 0.9)

    def test_save_load_roundtrip(self):
        X, y = self._toy()
        m = LogisticStateModel(dim=X.shape[1])
        m.fit(X, y, epochs=100)
        m.temperature = 1.7
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.npz")
            m.save(p, {"note": "test"})
            m2 = LogisticStateModel.load(p)
        for row in X[:20]:
            np.testing.assert_allclose(m.predict_proba(row), m2.predict_proba(row), atol=1e-6)

    def test_class_balancing_reaches_rare_class(self):
        """
        A call is mostly IVR and hold; without balancing the fit can simply
        never predict the rare class and still score well on accuracy.
        """
        rng = np.random.default_rng(1)
        dim = 8
        centres = rng.normal(0, 4, (4, dim))
        y = np.array([0] * 300 + [1] * 300 + [2] * 300 + [3] * 12)
        X = (centres[y] + rng.normal(0, 0.4, (len(y), dim))).astype(np.float32)
        m = LogisticStateModel(dim=dim)
        m.fit(X, y, epochs=500, lr=0.5, class_balance=True)
        preds = np.array([np.argmax(m.predict_proba(x)) for x in X[y == 3]])
        self.assertGreater(float(np.mean(preds == 3)), 0.5, "rare class never predicted")

    def test_temperature_softens_confidence(self):
        X, y = self._toy()
        m = LogisticStateModel(dim=X.shape[1])
        m.fit(X, y, epochs=300)
        sharp = float(np.max(m.predict_proba(X[0])))
        m.temperature = 3.0
        self.assertLess(float(np.max(m.predict_proba(X[0]))), sharp)


class TestFeaturizer(unittest.TestCase):
    def setUp(self):
        self.f = Featurizer(Config(), 48, 64)

    def test_dimension_contract(self):
        self.assertEqual(self.f.base_dim, len(SCALAR_NAMES) + 2 * EMB_DIM)
        self.assertEqual(self.f.dim, 3 * self.f.base_dim)
        self.assertEqual(len(self.f.transform(make_obs())), self.f.dim)

    def test_scalars_match_names(self):
        self.assertEqual(len(self.f.scalars(make_obs())), len(SCALAR_NAMES))
        named = self.f.named_scalars(make_obs(ivr_prompt_prob=0.7))
        self.assertAlmostEqual(named["ivr_prompt_prob"], 0.7, places=6)

    def test_context_block_is_causal(self):
        """
        The context block is a running mean over history only. On the very
        first hop it must equal the current vector, and the delta must be zero.
        """
        self.f.reset()
        v = self.f.transform(make_obs(t_s=0.5))
        n = self.f.base_dim
        np.testing.assert_allclose(v[:n], v[n:2 * n], atol=1e-6)
        np.testing.assert_allclose(v[2 * n:], 0.0, atol=1e-6)

    def test_delta_responds_to_change(self):
        self.f.reset()
        for _ in range(5):
            self.f.transform(make_obs(music_prob=0.0, t_s=1.0))
        v = self.f.transform(make_obs(music_prob=0.9, t_s=2.0))
        n = self.f.base_dim
        self.assertGreater(float(np.max(np.abs(v[2 * n:]))), 0.1)

    def test_all_finite(self):
        self.assertTrue(np.all(np.isfinite(self.f.transform(make_obs()))))


class TestSoftmax(unittest.TestCase):
    def test_stable_on_large_logits(self):
        p = softmax(np.array([1000.0, 1000.0, 999.0, -1000.0]))
        self.assertTrue(np.all(np.isfinite(p)))
        self.assertAlmostEqual(float(p.sum()), 1.0, places=9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
