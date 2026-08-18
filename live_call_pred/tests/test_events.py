"""
Transfer lifecycle. This is the package's central claim under test: states
and transfer events are separate outputs, so a transfer can be in progress
*while* the state is HOLD — something a flat 4-class softmax cannot express.
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from callstate.config import Config  # noqa: E402
from callstate.fusion.events import Phase, TransferDetector  # noqa: E402
from callstate.types import EventType, State  # noqa: E402


class TransferHarness:
    """Feeds a scripted state sequence in and collects the events out."""

    def __init__(self, **cfg_over):
        cfg = Config()
        for k, v in cfg_over.items():
            setattr(cfg, k, v)
        self.det = TransferDetector(cfg=cfg)
        self.t = 0.0
        self.events = []
        self._dwell_start = 0.0
        self._last_state = None

    def feed(self, state, seconds=3.0, speaker=None, is_new=False,
             transfer_phrase=0.0, transfer_fail=0.0, leg_changed=False, hop=0.5):
        n = int(seconds / hop)
        for _ in range(n):
            self.t += hop
            if state != self._last_state:
                self._dwell_start = self.t
                self._last_state = state
            evs = self.det.step(
                t_s=self.t, state=state, speaker_id=speaker, is_new_speaker=is_new,
                transfer_phrase_prob=transfer_phrase, transfer_fail_prob=transfer_fail,
                confidence=0.9, telephony_leg_changed=leg_changed,
                dwell_s=self.t - self._dwell_start,
            )
            self.events += evs
            is_new = False           # "new speaker" fires once, not for a whole run
            transfer_phrase = 0.0    # so does an announcement phrase
            transfer_fail = 0.0
            leg_changed = False
        return self

    def of(self, kind):
        return [e for e in self.events if e.type == kind]

    def outcomes(self):
        return [e.meta.get("outcome") for e in self.of(EventType.TRANSFER_END)]


class TestSuccessfulTransfer(unittest.TestCase):
    def test_human_hold_new_human_is_a_completed_transfer(self):
        h = (TransferHarness()
             .feed(State.HUMAN, 6, speaker="human_1")
             .feed(State.HUMAN, 2, speaker="human_1", transfer_phrase=0.9)
             .feed(State.HOLD, 10)
             .feed(State.HUMAN, 8, speaker="human_2", is_new=True))
        self.assertEqual(len(h.of(EventType.TRANSFER_START)), 1)
        self.assertEqual(h.outcomes(), ["completed"])

    def test_state_stays_hold_while_transfer_in_progress(self):
        """
        The point of separating states from events. During the hold phase the
        state is HOLD *and* a transfer is open — a flat classifier would have
        to discard one of those two true facts.
        """
        h = (TransferHarness()
             .feed(State.HUMAN, 4, speaker="human_1", transfer_phrase=0.9)
             .feed(State.HOLD, 4))
        self.assertTrue(h.det.in_progress)
        self.assertEqual(h.det.phase, Phase.IN_PROGRESS)

    def test_transfer_to_an_ivr_counts(self):
        h = (TransferHarness()
             .feed(State.HUMAN, 5, speaker="human_1", transfer_phrase=0.9)
             .feed(State.HOLD, 6)
             .feed(State.IVR, 6))
        self.assertEqual(h.outcomes(), ["completed"])

    def test_carrier_leg_change_alone_starts_a_transfer(self):
        """A new SIP leg is near-ground-truth and does not need a phrase."""
        h = TransferHarness().feed(State.HUMAN, 3, speaker="human_1", leg_changed=True)
        starts = h.of(EventType.TRANSFER_START)
        self.assertEqual(len(starts), 1)
        self.assertIn("call leg", starts[0].evidence)


class TestFailedTransfer(unittest.TestCase):
    def test_original_speaker_returning_is_a_failure(self):
        """
        The case that separates a real detector from a naive one: after an
        announcement and hold, the *same* representative comes back. Audio-wise
        this is identical to a successful transfer; only speaker identity
        distinguishes them.
        """
        h = (TransferHarness()
             .feed(State.HUMAN, 5, speaker="human_1", transfer_phrase=0.9)
             .feed(State.HOLD, 8)
             .feed(State.HUMAN, 8, speaker="human_1"))
        self.assertEqual(h.outcomes(), ["failed"])

    def test_spoken_failure_closes_immediately(self):
        h = (TransferHarness()
             .feed(State.HUMAN, 4, speaker="human_1", transfer_phrase=0.9)
             .feed(State.HUMAN, 4, speaker="human_1", transfer_fail=0.9))
        self.assertEqual(h.outcomes(), ["failed"])

    def test_announcement_with_nothing_following_times_out(self):
        """An announcement is a prediction, and predictions may be wrong."""
        h = (TransferHarness(transfer_announce_window_s=10.0)
             .feed(State.HUMAN, 3, speaker="human_1", transfer_phrase=0.9)
             .feed(State.HUMAN, 25, speaker="human_1"))
        self.assertEqual(h.outcomes(), ["failed"])
        self.assertIn("nothing followed", h.of(EventType.TRANSFER_END)[0].evidence)

    def test_open_transfer_closed_at_end_of_call(self):
        h = (TransferHarness()
             .feed(State.HUMAN, 3, speaker="human_1", transfer_phrase=0.9)
             .feed(State.HOLD, 6))
        ev = h.det.close_open_transfer(h.t)
        self.assertIsNotNone(ev)
        self.assertEqual(ev.meta["outcome"], "failed")
        self.assertEqual(h.det.phase, Phase.IDLE)

    def test_no_dangling_close_when_nothing_open(self):
        h = TransferHarness().feed(State.HUMAN, 5, speaker="human_1")
        self.assertIsNone(h.det.close_open_transfer(h.t))


class TestSettling(unittest.TestCase):
    def test_one_hop_flicker_does_not_resolve_a_transfer(self):
        """
        Regression: a single flickered IVR hop in the middle of hold used to
        close the transfer as 'completed', and the real completion 12 s later
        was then misattributed. A candidate state must hold before it can
        resolve anything.
        """
        h = (TransferHarness()
             .feed(State.HUMAN, 4, speaker="human_1", transfer_phrase=0.9)
             .feed(State.HOLD, 6)
             .feed(State.IVR, 0.5)          # single-hop blip
             .feed(State.HOLD, 6))
        self.assertEqual(h.outcomes(), [], "a one-hop blip resolved the transfer")
        self.assertTrue(h.det.in_progress)

    def test_transfer_still_resolves_once_the_new_state_settles(self):
        h = (TransferHarness()
             .feed(State.HUMAN, 4, speaker="human_1", transfer_phrase=0.9)
             .feed(State.HOLD, 6)
             .feed(State.IVR, 0.5)
             .feed(State.HOLD, 4)
             .feed(State.HUMAN, 8, speaker="human_2", is_new=True))
        self.assertEqual(h.outcomes(), ["completed"])


class TestBoundaryEvents(unittest.TestCase):
    def test_human_joined_fires_on_entry(self):
        h = TransferHarness().feed(State.HOLD, 4).feed(State.HUMAN, 4, speaker="human_1")
        self.assertEqual(len(h.of(EventType.HUMAN_JOINED)), 1)

    def test_ivr_exit_fires_when_leaving_a_menu(self):
        h = TransferHarness().feed(State.IVR, 4).feed(State.HUMAN, 4, speaker="human_1")
        self.assertEqual(len(h.of(EventType.IVR_EXIT)), 1)

    def test_speaker_changed_fires_once_per_change(self):
        h = (TransferHarness()
             .feed(State.HUMAN, 4, speaker="human_1")
             .feed(State.HUMAN, 4, speaker="human_2")
             .feed(State.HUMAN, 4, speaker="human_2"))
        self.assertEqual(len(h.of(EventType.SPEAKER_CHANGED)), 1)

    def test_no_speaker_change_on_first_speaker(self):
        h = TransferHarness().feed(State.HUMAN, 4, speaker="human_1")
        self.assertEqual(len(h.of(EventType.SPEAKER_CHANGED)), 0)

    def test_events_are_time_ordered(self):
        h = (TransferHarness()
             .feed(State.IVR, 4)
             .feed(State.HUMAN, 4, speaker="human_1", transfer_phrase=0.9)
             .feed(State.HOLD, 6)
             .feed(State.HUMAN, 6, speaker="human_2", is_new=True))
        ts = [e.t_s for e in h.events]
        self.assertEqual(ts, sorted(ts))

    def test_every_event_carries_evidence(self):
        h = (TransferHarness()
             .feed(State.HUMAN, 4, speaker="human_1", transfer_phrase=0.9)
             .feed(State.HOLD, 6)
             .feed(State.HUMAN, 6, speaker="human_2", is_new=True))
        self.assertTrue(h.events)
        for e in h.events:
            self.assertTrue(e.evidence.strip(), f"{e.type} has no evidence string")
            self.assertGreaterEqual(e.confidence, 0.0)
            self.assertLessEqual(e.confidence, 1.0)


class TestNoSpuriousTransfers(unittest.TestCase):
    def test_plain_ivr_navigation_produces_no_transfer(self):
        h = TransferHarness().feed(State.IVR, 30)
        self.assertEqual(len(h.of(EventType.TRANSFER_START)), 0)

    def test_hold_without_an_announcement_is_not_a_transfer(self):
        h = (TransferHarness()
             .feed(State.IVR, 8)
             .feed(State.HOLD, 12)
             .feed(State.HUMAN, 8, speaker="human_1", is_new=True))
        self.assertEqual(len(h.of(EventType.TRANSFER_START)), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
