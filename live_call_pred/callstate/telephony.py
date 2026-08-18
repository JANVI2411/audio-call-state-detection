"""
Out-of-band carrier signals.

Everything else in this package infers what happened from the waveform. The
carrier frequently just *tells* you — a new SIP leg, a participant change on
the conference bridge, a DTMF digit, an answering-machine-detection verdict.
When those arrive they are close to ground truth and should outrank acoustic
inference, so they enter the feature vector and the transfer detector
directly.

`TelephonyBus` is a time-ordered queue drained in call-time order, so a
recorded call with a saved event log replays exactly like the live one.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .types import TelephonyEvent


@dataclass
class TelephonyBus:
    events: List[TelephonyEvent] = field(default_factory=list)
    _cursor: int = 0
    last_agent_speech_s: float = -1e9
    last_dtmf_s: float = -1e9
    last_leg_change_s: float = -1e9

    @classmethod
    def from_jsonl(cls, path: str) -> "TelephonyBus":
        evs = []
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                evs.append(TelephonyEvent(float(d["t_s"]), d["kind"], d.get("detail", "")))
        return cls(events=sorted(evs, key=lambda e: e.t_s))

    def drain_until(self, t_s: float) -> List[TelephonyEvent]:
        out = []
        while self._cursor < len(self.events) and self.events[self._cursor].t_s <= t_s:
            ev = self.events[self._cursor]
            self._cursor += 1
            if ev.kind in ("sip_leg_changed", "participant_changed"):
                self.last_leg_change_s = ev.t_s
            elif ev.kind == "dtmf":
                self.last_dtmf_s = ev.t_s
            out.append(ev)
        return out

    def note_agent_speech(self, t_s: float) -> None:
        self.last_agent_speech_s = t_s

    def note_dtmf(self, t_s: float) -> None:
        self.last_dtmf_s = t_s

    def features(self, t_s: float, recent_s: float = 3.0) -> Dict[str, float]:
        since_agent = t_s - self.last_agent_speech_s
        return {
            "agent_recently_spoke": 1.0 if since_agent <= recent_s else 0.0,
            "time_since_agent_spoke": float(min(since_agent, 30.0)) if since_agent >= 0 else 30.0,
            "dtmf_recent": 1.0 if (t_s - self.last_dtmf_s) <= recent_s else 0.0,
            "sip_leg_changed": 1.0 if (t_s - self.last_leg_change_s) <= recent_s else 0.0,
        }

    def leg_changed_recently(self, t_s: float, recent_s: float = 3.0) -> bool:
        return (t_s - self.last_leg_change_s) <= recent_s
