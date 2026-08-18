"""
Causality: the ring buffer must be structurally incapable of look-ahead.

This is the property the whole "it streams" claim rests on, so it is tested
directly rather than argued for in a comment. The trick is to write a strictly
increasing counter as the audio signal — then any sample the buffer returns
whose value exceeds the number written so far is, literally, a sample from
the future.
"""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from callstate.audio.source import ArraySource, RingBuffer  # noqa: E402
from callstate.config import Config  # noqa: E402


class TestRingBuffer(unittest.TestCase):
    def test_never_returns_future_samples(self):
        cap = 100
        rb = RingBuffer(cap)
        written = 0
        for block in range(1, 40):
            chunk = np.arange(written, written + block, dtype=np.float32)
            written += block
            rb.write(chunk)
            got = rb.read()
            self.assertLessEqual(len(got), cap)
            if len(got):
                self.assertLess(float(got.max()), written,
                                "ring buffer returned a sample not yet written")
                # and the newest sample must be the most recent one written
                self.assertAlmostEqual(float(got[-1]), written - 1, places=5)

    def test_holds_only_most_recent_capacity(self):
        rb = RingBuffer(10)
        rb.write(np.arange(25, dtype=np.float32))
        got = rb.read()
        self.assertEqual(len(got), 10)
        np.testing.assert_allclose(got, np.arange(15, 25))

    def test_partial_fill_reports_actual_length(self):
        rb = RingBuffer(50)
        rb.write(np.ones(7, dtype=np.float32))
        self.assertEqual(rb.filled, 7)
        self.assertEqual(len(rb.read()), 7)

    def test_read_n_clamps(self):
        rb = RingBuffer(20)
        rb.write(np.arange(20, dtype=np.float32))
        self.assertEqual(len(rb.read(5)), 5)
        self.assertEqual(len(rb.read(999)), 20)
        np.testing.assert_allclose(rb.read(3), np.array([17, 18, 19]))

    def test_oversized_write_keeps_tail(self):
        rb = RingBuffer(4)
        rb.write(np.arange(10, dtype=np.float32))
        np.testing.assert_allclose(rb.read(), np.array([6, 7, 8, 9]))

    def test_empty_buffer_reads_empty(self):
        self.assertEqual(len(RingBuffer(10).read()), 0)


class TestSourceOrdering(unittest.TestCase):
    def test_frames_are_monotonic_and_contiguous(self):
        cfg = Config()
        n = cfg.target_sr * 3
        src = ArraySource(np.arange(n, dtype=np.float32), sample_rate=cfg.target_sr,
                          frame_ms=cfg.frame_ms)
        prev_t = -1.0
        total = 0
        for f in src.frames():
            self.assertGreater(f.t_s, prev_t)
            prev_t = f.t_s
            self.assertAlmostEqual(f.t_s, total / cfg.target_sr, places=6)
            total += len(f.remote)
        self.assertGreater(total, 0)

    def test_agent_channel_padded_when_absent(self):
        src = ArraySource(np.ones(1600, dtype=np.float32), sample_rate=8000, frame_ms=20)
        for f in src.frames():
            self.assertEqual(len(f.agent), len(f.remote))


class TestEngineCausality(unittest.TestCase):
    def test_step_output_depends_only_on_past(self):
        """
        Run the same prefix twice, with different audio appended *after* it,
        and require identical beliefs over the shared prefix. If any stage
        peeked ahead, the two runs would diverge.
        """
        from callstate.engine import CallStateEngine
        from callstate.semantics.asr import NullBackend

        rng = np.random.default_rng(3)
        cfg = Config()
        prefix = rng.normal(0, 0.05, cfg.target_sr * 5).astype(np.float32)
        tail_a = rng.normal(0, 0.30, cfg.target_sr * 5).astype(np.float32)
        tail_b = np.zeros(cfg.target_sr * 5, dtype=np.float32)

        def run(sig):
            eng = CallStateEngine(cfg, asr_backend=NullBackend())
            res = eng.run(ArraySource(sig, sample_rate=cfg.target_sr,
                                      frame_ms=cfg.frame_ms), call_id="c")
            return res.timeline

        tl_a = run(np.concatenate([prefix, tail_a]))
        tl_b = run(np.concatenate([prefix, tail_b]))

        shared = [r for r in tl_a if r.t_s <= 5.0]
        self.assertGreater(len(shared), 5, "not enough shared hops to be meaningful")
        for ra, rb in zip(shared, tl_b):
            self.assertAlmostEqual(ra.t_s, rb.t_s, places=6)
            self.assertEqual(ra.state, rb.state,
                             f"state at t={ra.t_s}s changed based on future audio")
            self.assertAlmostEqual(ra.confidence, rb.confidence, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
