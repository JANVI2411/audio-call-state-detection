"""
Commits a filtered posterior to a single reportable state.

The HMM already suppresses most flicker. This adds the last guard: the state
only *changes* when the new leader clears `commit_min_prob`. Below that the
tracker holds what it had. The asymmetry is intentional — staying put is free,
switching is what makes the agent's policy change behaviour, so switching
should require more evidence than staying.

`dwell_s` (time in the current state) is fed back into the featurizer, which
is why this class is stateful rather than a pure function: how long we have
been on hold is real evidence about what comes next.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..types import State, StateBelief


@dataclass
class StateTracker:
    cfg: object
    state: State = State.OTHER
    entered_at_s: float = 0.0
    history: List[StateBelief] = field(default_factory=list)
    _initialised: bool = False

    def dwell_s(self, t_s: float) -> float:
        return max(0.0, t_s - self.entered_at_s)

    def commit(self, t_s: float, posterior: np.ndarray, raw: np.ndarray) -> StateBelief:
        order = State.order()
        probs = {s.value: float(p) for s, p in zip(order, posterior)}
        raw_probs = {s.value: float(p) for s, p in zip(order, raw)}

        top_i = int(np.argmax(posterior))
        top_state, top_p = order[top_i], float(posterior[top_i])

        if not self._initialised:
            self.state, self.entered_at_s, self._initialised = top_state, t_s, True
        elif top_state != self.state and top_p >= float(self.cfg.commit_min_prob):
            self.state, self.entered_at_s = top_state, t_s

        belief = StateBelief(
            t_s=t_s, probs=probs, state=self.state, raw_probs=raw_probs,
            confidence=float(probs[self.state.value]),
        )
        self.history.append(belief)
        return belief

    def previous_state(self) -> Optional[State]:
        """Last state that differed from the current one (for transfer logic)."""
        for b in reversed(self.history[:-1]):
            if b.state != self.state:
                return b.state
        return None
