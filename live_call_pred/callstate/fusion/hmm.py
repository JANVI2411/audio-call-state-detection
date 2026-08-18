"""
Causal HMM forward filter over the state posterior.

This is the layer that turns a twitchy per-window classifier into something a
voice agent can act on. Taking argmax of a per-window distribution produces
state flicker — IVR, IVR, HUMAN, IVR, HUMAN — because individual windows are
genuinely ambiguous (a two-second gap in an IVR menu looks like hold; the
first word of a representative's greeting looks like a prompt). Filtering
with transition priors means a single odd window cannot move the call state;
several consistent ones can.

Only the forward recursion is used. Forward-backward would be more accurate
but requires the future, which does not exist yet in a live call — that
constraint is the whole reason this is a *filter* and not a smoother.

    alpha_t(j) ∝ b_j(o_t) * Σ_i alpha_{t-1}(i) * a_ij

Computed in log space; a 4x4 transition matrix makes the cost negligible.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..types import State

N = len(State.order())


def build_transition_matrix(self_prob: Dict[str, float],
                            bias: Dict[str, Dict[str, float]]) -> np.ndarray:
    """
    Rows = from-state, columns = to-state. Self-transition mass is set
    directly; the remainder is split across the other states in proportion to
    the configured bias.

    The biases encode real call structure, not symmetry: HOLD→HUMAN is
    weighted well above HOLD→IVR because coming off hold to a representative
    is the common case, and HUMAN→IVR is weighted low because a live person
    rarely hands you straight back to a menu without hold in between.
    """
    order = [s.value for s in State.order()]
    A = np.zeros((N, N), dtype=np.float64)
    for i, src in enumerate(order):
        p_self = float(self_prob.get(src, 0.9))
        A[i, i] = p_self
        row_bias = bias.get(src, {})
        others = [(j, dst) for j, dst in enumerate(order) if dst != src]
        weights = np.array([max(row_bias.get(dst, 1.0), 1e-6) for _, dst in others])
        weights = weights / weights.sum()
        for (j, _), w in zip(others, weights):
            A[i, j] = (1.0 - p_self) * w
    return A / A.sum(axis=1, keepdims=True)


class HMMFilter:
    def __init__(self, A: np.ndarray, prior: Optional[np.ndarray] = None,
                 floor: float = 1e-4):
        self.logA = np.log(np.clip(A, 1e-12, None))
        # Calls start on IVR or ringback far more often than mid-conversation
        # with a human, so the initial prior is not uniform.
        self.prior = prior if prior is not None else np.array([0.45, 0.2, 0.2, 0.15])
        self.floor = floor
        self.log_alpha = np.log(self.prior)

    def reset(self) -> None:
        self.log_alpha = np.log(self.prior)

    def step(self, emission: np.ndarray) -> np.ndarray:
        """One observation in, filtered posterior out."""
        b = np.log(np.clip(np.asarray(emission, dtype=np.float64), self.floor, None))
        # logsumexp over the previous state, per destination
        m = np.max(self.log_alpha[:, None] + self.logA, axis=0)
        s = np.log(np.sum(np.exp(self.log_alpha[:, None] + self.logA - m), axis=0)) + m
        la = s + b
        la -= np.max(la)
        p = np.exp(la)
        p /= p.sum()
        self.log_alpha = np.log(np.clip(p, 1e-12, None))
        return p

    @property
    def posterior(self) -> np.ndarray:
        p = np.exp(self.log_alpha - np.max(self.log_alpha))
        return p / p.sum()
