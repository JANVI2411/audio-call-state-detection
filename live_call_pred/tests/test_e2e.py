"""
End-to-end: the whole pipeline over generated calls with known ground truth,
plus the ASR buffering layer and the output sinks.

These are the tests that would catch a regression nobody predicted — they
exercise every stage together on audio, against labels, with the real
streaming loop.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from callstate.audio.source import ArraySource, WavFileSource  # noqa: E402
from callstate.config import Config  # noqa: E402
from callstate.engine import CallStateEngine, segment_timeline  # noqa: E402
from callstate.io_sinks import read_jsonl, write_results  # noqa: E402
from callstate.logging_setup import setup_logging  # noqa: E402
from callstate.metrics import (align_frames, boundary_metrics, calibration,  # noqa: E402
                               state_metrics, transfer_metrics)
from callstate.semantics.asr import (NullBackend, ScriptedBackend,  # noqa: E402
                                     StreamingASR, build_asr_backend)
from callstate.simulate import make_call, write_call  # noqa: E402
from callstate.telephony import TelephonyBus  # noqa: E402
from callstate.types import EventType, State  # noqa: E402

setup_logging("ERROR", quiet_console=True)


def run_call(scenario: str, seed: int = 11, cfg: Config = None):
    cfg = cfg or Config()
    call = make_call(scenario=scenario, seed=seed)
    tel = TelephonyBus(events=[])
    if call.telephony:
        from callstate.types import TelephonyEvent

        tel = TelephonyBus(events=[TelephonyEvent(e["t_s"], e["kind"], e.get("detail", ""))
                                   for e in call.telephony])
    engine = CallStateEngine(cfg, asr_backend=ScriptedBackend(call.script()), telephony=tel)
    src = ArraySource(call.remote, call.agent, sample_rate=call.sample_rate,
                      frame_ms=cfg.frame_ms)
    return engine.run(src, call_id=scenario), call, engine


class TestPipelineRuns(unittest.TestCase):
    def test_produces_a_timeline_covering_the_call(self):
        result, call, _ = run_call("simple")
        self.assertGreater(len(result.timeline), 10)
        self.assertLessEqual(result.duration_s, call.duration_s + 0.1)
        self.assertGreater(result.timeline[-1].t_s, call.duration_s * 0.9)

    def test_hops_are_monotonic_and_evenly_spaced(self):
        result, _, _ = run_call("simple")
        ts = [r.t_s for r in result.timeline]
        self.assertEqual(ts, sorted(ts))
        gaps = np.diff(ts)
        self.assertLess(float(np.max(np.abs(gaps - Config().hop_s))), 0.05)

    def test_every_row_is_a_valid_state_with_a_valid_distribution(self):
        result, _, _ = run_call("transfer")
        valid = {s.value for s in State.order()}
        for r in result.timeline:
            self.assertIn(r.state, valid)
            self.assertAlmostEqual(sum(r.probs.values()), 1.0, places=3)
            self.assertGreaterEqual(r.confidence, 0.0)
            self.assertLessEqual(r.confidence, 1.0)

    def test_runs_without_asr(self):
        """Acoustic-only operation must degrade, not fail."""
        cfg = Config()
        call = make_call("simple", seed=5)
        engine = CallStateEngine(cfg, asr_backend=NullBackend())
        result = engine.run(ArraySource(call.remote, call.agent,
                                        sample_rate=call.sample_rate,
                                        frame_ms=cfg.frame_ms), call_id="noasr")
        self.assertGreater(len(result.timeline), 10)
        self.assertTrue(all(r.text == "" for r in result.timeline))

    def test_silence_only_call_does_not_crash(self):
        cfg = Config()
        engine = CallStateEngine(cfg, asr_backend=NullBackend())
        sig = np.random.default_rng(0).normal(0, 1e-4, cfg.target_sr * 8).astype(np.float32)
        result = engine.run(ArraySource(sig, sample_rate=cfg.target_sr,
                                        frame_ms=cfg.frame_ms), call_id="quiet")
        self.assertGreater(len(result.timeline), 5)

    def test_very_short_call(self):
        cfg = Config()
        engine = CallStateEngine(cfg, asr_backend=NullBackend())
        sig = np.zeros(cfg.target_sr, dtype=np.float32)
        result = engine.run(ArraySource(sig, sample_rate=cfg.target_sr,
                                        frame_ms=cfg.frame_ms), call_id="short")
        self.assertGreaterEqual(len(result.timeline), 1)
        self.assertIn("state_durations_s", result.summary)


class TestRecognisesStates(unittest.TestCase):
    """Each scenario must surface the states it actually contains."""

    def _dominant_over(self, result, lo, hi):
        rows = [r for r in result.timeline if lo <= r.t_s < hi]
        if not rows:
            return None
        counts = {}
        for r in rows:
            counts[r.state] = counts.get(r.state, 0) + 1
        return max(counts, key=counts.get)

    def test_ivr_only_call_is_dominated_by_ivr(self):
        result, _, _ = run_call("ivr_only", seed=21)
        self.assertEqual(result.summary["dominant_state"], "ivr")

    def test_transfer_call_contains_all_three_main_states(self):
        result, _, _ = run_call("transfer", seed=23)
        seen = {r.state for r in result.timeline}
        for want in ("ivr", "human", "hold"):
            self.assertIn(want, seen, f"never detected {want}")

    def test_reaches_reasonable_accuracy_against_gold(self):
        result, call, _ = run_call("transfer", seed=23)
        turns = [{"start_s": t.start_s, "end_s": t.end_s, "state": t.state}
                 for t in call.turns]
        yt, yp = align_frames([r.to_json() for r in result.timeline], turns, offset_s=0.25)
        m = state_metrics(yt, yp)
        # The default engine uses the untrained prior weights, so this is a
        # floor on the no-training-data case, not the trained head's score.
        self.assertGreater(m["accuracy"], 0.55, f"accuracy collapsed: {m['accuracy']}")

    def test_speakers_are_separated_on_a_transfer_call(self):
        _result, _call, engine = run_call("transfer", seed=23)
        self.assertGreaterEqual(len(engine.speaker.registry.centroids), 2,
                                "never distinguished the two representatives")


class TestTransferDetectionE2E(unittest.TestCase):
    def test_successful_transfer_detected_end_to_end(self):
        result, call, _ = run_call("transfer", seed=23)
        gold = call.gold_events
        m = transfer_metrics([e.to_json() for e in result.events], gold, tolerance_s=8.0)
        self.assertEqual(m["transfer_start"]["fn"], 0, "missed the transfer announcement")
        self.assertEqual(m["transfer_end"]["fn"], 0, "missed the transfer completion")
        self.assertEqual(m.get("outcome_accuracy"), 1.0, "wrong completed/failed verdict")

    def test_failed_transfer_reported_as_failed(self):
        result, call, _ = run_call("failed_transfer", seed=31)
        ends = [e for e in result.events if e.type == EventType.TRANSFER_END]
        self.assertTrue(ends, "no transfer resolution at all")
        self.assertIn("failed", [e.meta.get("outcome") for e in ends])

    def test_ivr_only_call_reports_no_transfer(self):
        result, _, _ = run_call("ivr_only", seed=21)
        starts = [e for e in result.events if e.type == EventType.TRANSFER_START]
        self.assertEqual(len(starts), 0, "invented a transfer on an IVR-only call")

    def test_summary_counts_match_the_event_log(self):
        result, _, _ = run_call("transfer", seed=23)
        ends = [e for e in result.events if e.type == EventType.TRANSFER_END]
        self.assertEqual(result.summary["n_transfers_completed"],
                         sum(1 for e in ends if e.meta.get("outcome") == "completed"))
        self.assertEqual(result.summary["n_transfers_failed"],
                         sum(1 for e in ends if e.meta.get("outcome") == "failed"))


class TestLatency(unittest.TestCase):
    def test_every_hop_fits_the_budget(self):
        result, _, _ = run_call("transfer", seed=23)
        budget = Config().hop_s * 1000
        p95 = float(np.percentile([r["total_ms"] for r in result.latency], 95))
        self.assertLess(p95, budget, f"p95 {p95:.0f}ms exceeds the {budget:.0f}ms hop budget")
        self.assertTrue(result.summary["realtime_ok"])

    def test_branch_timings_are_reported(self):
        result, _, _ = run_call("simple")
        for k in ("acoustic_ms", "asr_ms", "speaker_ms", "fusion_ms"):
            self.assertIn(k, result.latency[0])


class TestStreamingASR(unittest.TestCase):
    def test_deduplicates_across_overlapping_windows(self):
        """
        Overlapping windows re-transcribe the same audio each hop. Without
        de-duplication the same words are emitted repeatedly and the
        word-rate feature becomes meaningless.
        """
        script = [(0.0, 2.0, "for claims press two")]
        asr = StreamingASR(ScriptedBackend(script), window_s=6.0)
        sr = 8000
        audio = np.zeros(sr * 6, dtype=np.float32)
        total = 0
        for start in (0.0, 0.5, 1.0, 1.5, 2.0):
            total += len(asr.push(audio[: int(sr * (start + 2))], sr, 0.0))
        self.assertEqual(total, 4, f"emitted {total} words for a 4-word utterance")

    def test_recent_text_respects_lookback(self):
        asr = StreamingASR(ScriptedBackend([(0.0, 1.0, "alpha"), (9.0, 10.0, "omega")]),
                           window_s=12.0, keep_s=30.0)
        asr.push(np.zeros(8000 * 11, dtype=np.float32), 8000, 0.0)
        self.assertIn("omega", asr.recent_text(10.5, lookback_s=4.0))
        self.assertNotIn("alpha", asr.recent_text(10.5, lookback_s=4.0))

    def test_word_rate_is_per_second(self):
        asr = StreamingASR(ScriptedBackend([(0.0, 2.0, "one two three four")]), window_s=6.0)
        asr.push(np.zeros(8000 * 3, dtype=np.float32), 8000, 0.0)
        self.assertGreater(asr.word_rate(2.5, lookback_s=5.0), 0.0)

    def test_old_words_are_evicted(self):
        asr = StreamingASR(ScriptedBackend([(0.0, 1.0, "gone")]), window_s=6.0, keep_s=2.0)
        asr.push(np.zeros(8000 * 20, dtype=np.float32), 8000, 0.0)
        self.assertEqual(len(asr.words), 0)

    def test_backend_falls_back_when_unavailable(self):
        b = build_asr_backend("auto", "definitely-not-a-real-model-name")
        self.assertIn(b.name, ("faster_whisper", "null"))


class TestSegmentation(unittest.TestCase):
    def test_collapses_runs(self):
        from callstate.types import TimelineRow

        rows = [TimelineRow(t, s, 0.9, {}, None, "", False, 1.0) for t, s in
                [(0.5, "ivr"), (1.0, "ivr"), (1.5, "hold"), (2.0, "hold"), (2.5, "human")]]
        segs = segment_timeline(rows)
        self.assertEqual([s.state for s in segs], ["ivr", "hold", "human"])
        self.assertEqual(segs[0].start_s, 0.5)
        self.assertEqual(segs[1].end_s, 2.0)

    def test_empty_timeline(self):
        self.assertEqual(segment_timeline([]), [])


class TestOutputSinks(unittest.TestCase):
    def test_writes_all_streams_as_valid_jsonl(self):
        result, _, _ = run_call("transfer", seed=23)
        with tempfile.TemporaryDirectory() as d:
            paths = write_results(result, d)
            for key in ("timeline", "events", "segments", "latency"):
                rows = read_jsonl(paths[key])
                self.assertIsInstance(rows, list)
            self.assertEqual(len(read_jsonl(paths["timeline"])), len(result.timeline))
            with open(paths["summary"]) as fh:
                summary = json.load(fh)
            for key in ("call_id", "duration_s", "state_durations_s", "latency_ms",
                        "config_fingerprint"):
                self.assertIn(key, summary)

    def test_synthetic_corpus_files_roundtrip(self):
        call = make_call("transfer", seed=77)
        with tempfile.TemporaryDirectory() as d:
            paths = write_call(call, d, "c")
            for p in paths.values():
                self.assertTrue(os.path.exists(p))
            src = WavFileSource(paths["wav"], target_sr=8000, agent_channel=0)
            self.assertAlmostEqual(src.duration_s, call.duration_s, places=1)
            with open(paths["script"]) as fh:
                self.assertTrue(json.load(fh))


class TestMetricsHonesty(unittest.TestCase):
    def test_no_gold_events_is_reported_as_not_evaluated(self):
        """
        Zero gold events and zero predictions is not a perfect score — it is
        an untested capability, and must not read as 1.00.
        """
        m = transfer_metrics([], [])
        self.assertIsNone(m["transfer_start"]["f1"])
        self.assertIn("note", m["transfer_start"])

    def test_missed_event_scores_zero_recall(self):
        m = transfer_metrics([], [{"type": "transfer_start", "t_s": 10.0}])
        self.assertEqual(m["transfer_start"]["recall"], 0.0)

    def test_boundary_metrics_count_gold_and_predicted(self):
        turns = [{"start_s": 0, "end_s": 10, "state": "ivr"},
                 {"start_s": 10, "end_s": 20, "state": "human"}]
        tl = [{"t_s": t, "state": "ivr" if t < 10.2 else "human", "confidence": 0.9}
              for t in np.arange(0.5, 20, 0.5)]
        b = boundary_metrics(tl, turns)
        self.assertEqual(b["n_gold_boundaries"], 1)
        self.assertEqual(b["recall@1.0s"], 1.0)

    def test_calibration_reports_bins_brier_and_ece(self):
        tl = [{"t_s": 1.0, "state": "ivr", "confidence": 0.9},
              {"t_s": 1.5, "state": "ivr", "confidence": 0.5}]
        c = calibration(tl, ["ivr", "human"])
        self.assertIn("brier", c)
        self.assertIn("ece", c)
        self.assertTrue(c["bins"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
