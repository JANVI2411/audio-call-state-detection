"""
The event head: transfer lifecycle, speaker changes, and phase boundaries.

This is where the package's central claim pays off. A flat softmax over
{IVR, HUMAN, HOLD, TRANSFER} forces a choice that reality does not offer —
during a transfer the audio *is* hold music, so the flat model must either
call it HOLD (and never report the transfer) or call it TRANSFER (and lose the
fact that hold music is playing). Here the two coexist: the state stays HOLD
while `transfer_in_progress` is independently true.

A transfer is scored from converging evidence, none of which is sufficient
alone:

  announcement   "let me get you over to billing"      (language)
  hold entry     HUMAN -> HOLD                          (state transition)
  new party      different speaker embedding, or IVR    (speaker branch)
  carrier signal new SIP leg / participant change       (telephony)

Reaching a new party after an announcement is a confirmed transfer.
Announcement with nothing following it inside `transfer_announce_window_s` is
a *failed* transfer, reported as such — an announcement is a prediction about
the future, and predictions have to be allowed to be wrong.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from ..types import Event, EventType, State


class Phase(Enum):
    IDLE = "idle"
    ANNOUNCED = "announced"
    IN_PROGRESS = "in_progress"


@dataclass
class TransferDetector:
    cfg: object
    phase: Phase = Phase.IDLE
    announced_at_s: Optional[float] = None
    started_at_s: Optional[float] = None
    party_before: Optional[str] = None
    state_before: Optional[State] = None
    hold_entered_at_s: Optional[float] = None
    events: List[Event] = field(default_factory=list)
    _prev_state: Optional[State] = None
    _prev_speaker: Optional[str] = None
    _seen_ivr: bool = False
    settle_s: float = 2.0   # how long a state must hold before it can resolve a transfer

    @property
    def in_progress(self) -> bool:
        return self.phase in (Phase.ANNOUNCED, Phase.IN_PROGRESS)

    def _emit(self, ev: Event) -> Event:
        self.events.append(ev)
        return ev

    def step(self, t_s: float, state: State, speaker_id: Optional[str],
             is_new_speaker: bool, transfer_phrase_prob: float,
             transfer_fail_prob: float, confidence: float,
             telephony_leg_changed: bool = False,
             dwell_s: float = 1e9) -> List[Event]:
        """
        `dwell_s` is how long the tracker has held the current state.

        Resolving a transfer on the first hop of a new state was a real and
        expensive error: a single flickered IVR hop in the middle of hold
        closed the transfer as "completed", and a return-from-hold hop
        arriving one hop before the speaker branch committed the new identity
        closed it as "failed" against the *outgoing* speaker. Both are cases
        where waiting a moment costs nothing and guessing costs the whole
        event. So a transfer only resolves once the candidate state has held
        for `settle_s`, which also gives the speaker registry time to commit.
        """
        out: List[Event] = []
        entered = state != self._prev_state
        settled = dwell_s >= self.settle_s

        # --- boundary events, independent of the transfer lifecycle ---------
        if entered and self._prev_state == State.IVR:
            self._seen_ivr = True
            out.append(self._emit(Event(EventType.IVR_EXIT, t_s, confidence,
                                        f"left IVR into {state.value}")))
        if state == State.IVR:
            self._seen_ivr = True
        if speaker_id and speaker_id != self._prev_speaker and self._prev_speaker is not None:
            out.append(self._emit(Event(
                EventType.SPEAKER_CHANGED, t_s, confidence,
                f"{self._prev_speaker} -> {speaker_id}",
                {"from": self._prev_speaker, "to": speaker_id})))
        if entered and state == State.HUMAN:
            out.append(self._emit(Event(EventType.HUMAN_JOINED, t_s, confidence,
                                        f"human speech after {self._prev_state.value if self._prev_state else 'start'}",
                                        {"speaker_id": speaker_id or ""})))
        if entered and state == State.HOLD:
            self.hold_entered_at_s = t_s

        # --- transfer lifecycle ---------------------------------------------
        if self.phase == Phase.IDLE:
            announced = transfer_phrase_prob >= 0.5 and state in (State.HUMAN, State.IVR)
            if announced or telephony_leg_changed:
                self.phase = Phase.ANNOUNCED
                self.announced_at_s = t_s
                self.party_before = speaker_id
                self.state_before = state
                ev = "announcement phrase" if announced else "carrier reported new call leg"
                out.append(self._emit(Event(
                    EventType.TRANSFER_START, t_s,
                    max(transfer_phrase_prob, 0.8 if telephony_leg_changed else 0.0),
                    ev, {"phase": "announced"})))

        elif self.phase == Phase.ANNOUNCED:
            if transfer_fail_prob >= 0.5:
                self._close(t_s, transfer_fail_prob, "speaker stated the transfer will not happen",
                            success=False, out=out)
            elif state == State.HOLD:
                self.phase = Phase.IN_PROGRESS
                self.started_at_s = t_s
            elif settled and self._reached_new_party(state, speaker_id, is_new_speaker):
                self._close(t_s, confidence, f"new party ({state.value}) without hold",
                            success=True, out=out)
            elif (t_s - (self.announced_at_s or t_s)) > self.cfg.transfer_announce_window_s:
                self._close(t_s, 0.5, "announced but nothing followed within the window",
                            success=False, out=out)

        elif self.phase == Phase.IN_PROGRESS:
            held_long_enough = (self.started_at_s is None or
                                (t_s - self.started_at_s) >= self.cfg.transfer_min_hold_s)
            if state != State.HOLD and held_long_enough and settled:
                if self._reached_new_party(state, speaker_id, is_new_speaker):
                    self._close(t_s, confidence, f"new party on the line ({state.value})",
                                success=True, out=out)
                elif state == State.HUMAN and speaker_id == self.party_before:
                    self._close(t_s, confidence, "original speaker returned after hold",
                                success=False, out=out)
            elif (t_s - (self.announced_at_s or t_s)) > self.cfg.transfer_max_duration_s:
                self._close(t_s, 0.4, "transfer exceeded maximum duration",
                            success=False, out=out)

        self._prev_state = state
        if speaker_id:
            self._prev_speaker = speaker_id
        return out

    def _reached_new_party(self, state: State, speaker_id: Optional[str],
                           is_new_speaker: bool) -> bool:
        if state == State.HUMAN:
            return is_new_speaker or (speaker_id is not None and speaker_id != self.party_before)
        # Landing in an IVR counts as reaching a new party only if we were
        # previously with a human — a menu after a menu is just navigation.
        return state == State.IVR and self.state_before == State.HUMAN

    def _close(self, t_s: float, confidence: float, evidence: str, success: bool,
               out: List[Event]) -> None:
        out.append(self._emit(Event(
            EventType.TRANSFER_END, t_s, confidence, evidence,
            {"outcome": "completed" if success else "failed",
             "duration_s": f"{t_s - (self.announced_at_s or t_s):.1f}"})))
        self.phase = Phase.IDLE
        self.announced_at_s = None
        self.started_at_s = None
        self.state_before = None

    def close_open_transfer(self, t_s: float) -> Optional[Event]:
        """Call at end of call: never leave a transfer dangling in the log."""
        if self.phase == Phase.IDLE:
            return None
        out: List[Event] = []
        self._close(t_s, 0.4, "call ended before the transfer resolved",
                    success=False, out=out)
        return out[0]
