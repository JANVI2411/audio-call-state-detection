"""
The state head: features in, a distribution over {IVR, HUMAN, HOLD, OTHER} out.

Three implementations behind one interface, because a system like this has to
be useful on day zero and improvable on day ninety:

`PriorStateModel`   Hand-specified evidence weights over the *named* scalars.
                    Needs no labelled data, runs everywhere, and is fully
                    explainable — `explain()` returns the per-feature
                    contributions that produced the score. This is the default
                    and it is what the shipped tests exercise.

`LogisticStateModel` Multinomial logistic regression over the full feature
                    vector, trained with plain gradient descent in numpy.
                    This is the first thing to fit once real labelled calls
                    exist: a few hundred labelled segments is enough for a
                    model of this size, and it is small enough that it will
                    not silently memorise a handful of calls.

`GRUStateModel`     Causal GRU (torch, optional). The right shape for the job
                    once there is real data volume, since it learns temporal
                    structure instead of relying on the featurizer's
                    hand-built context summary. Opt-in, never required.

All three return probabilities that then go through the HMM — none of them is
allowed to be the final answer on its own.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..types import State
from .featurizer import SCALAR_NAMES

N_STATES = len(State.order())


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)


class StateModel:
    name: str = "base"

    def predict_proba(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class PriorStateModel(StateModel):
    """
    Evidence weights, written down rather than fitted.

    Read a row as "this feature is evidence for this state, at this strength".
    They encode the domain facts the architecture is built on:
      - hold is an *acoustic* judgment (music loop + no words) that language
        can confirm but rarely establishes;
      - IVR is primarily a *lexical* judgment ("press or say one"), with flat
        pitch as weak corroboration;
      - human is established by disfluency and prosodic range, never by
        "it wasn't the other two" — a negative definition is what makes flat
        4-class classifiers brittle here.
    """

    name = "prior"

    WEIGHTS: Dict[str, Dict[str, float]] = {
        "ivr": {
            "ivr_prompt_prob": 3.4, "syllable_mod": 1.8, "dtmf_recent": 1.1,
            "speech_prob": 0.9, "has_text": 0.4,
            "pitch_cv": -2.0, "music_prob": -1.2, "slow_mod": -1.0,
            "human_spontaneous_prob": -1.8, "hold_phrase_prob": -0.5,
            "tone_prob": -1.0,
        },
        "human": {
            "human_spontaneous_prob": 3.0, "pitch_cv": 3.0, "syllable_mod": 1.6,
            "speech_prob": 1.2, "has_text": 0.5, "agent_recently_spoke": 0.55,
            "word_rate": 0.3,
            "ivr_prompt_prob": -2.4, "music_prob": -1.8, "slow_mod": -1.0,
            "silence_prob": -1.2, "hold_phrase_prob": -0.9, "tone_prob": -1.0,
        },
        "hold": {
            "music_prob": 3.0, "slow_mod": 1.8, "hold_phrase_prob": 1.6,
            "spectral_stability": 0.6, "periodicity": 0.5, "silence_prob": 0.5,
            "syllable_mod": -2.0, "has_text": -0.8, "word_rate": -0.9,
            "human_spontaneous_prob": -1.4, "speech_prob": -1.0,
            "ivr_prompt_prob": -0.8,
        },
        "other": {
            "tone_prob": 4.0, "silence_prob": 1.6, "slow_mod": 0.8,
            "has_text": -0.8, "speech_prob": -0.8, "music_prob": -0.5,
            "syllable_mod": -1.2,
        },
    }
    BIAS = {"ivr": -0.35, "human": -0.15, "hold": -0.45, "other": -0.55}

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
        idx = {n: i for i, n in enumerate(SCALAR_NAMES)}
        self.W = np.zeros((N_STATES, len(SCALAR_NAMES)), dtype=np.float32)
        self.b = np.zeros(N_STATES, dtype=np.float32)
        for s_i, st in enumerate(State.order()):
            for feat, w in self.WEIGHTS[st.value].items():
                self.W[s_i, idx[feat]] = w
            self.b[s_i] = self.BIAS[st.value]

    def predict_proba_scalars(self, scal: np.ndarray) -> np.ndarray:
        return softmax((self.W @ scal + self.b) / max(self.temperature, 1e-3))

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        # The full vector is [current | context-mean | delta]; the prior model
        # reads only the current block's named scalars.
        return self.predict_proba_scalars(x[: len(SCALAR_NAMES)])

    def explain(self, scal: np.ndarray, top_k: int = 4) -> Dict[str, List[Tuple[str, float]]]:
        out: Dict[str, List[Tuple[str, float]]] = {}
        for s_i, st in enumerate(State.order()):
            contrib = self.W[s_i] * scal
            order = np.argsort(-np.abs(contrib))[:top_k]
            out[st.value] = [(SCALAR_NAMES[i], round(float(contrib[i]), 3))
                             for i in order if abs(contrib[i]) > 1e-6]
        return out


class LogisticStateModel(StateModel):
    name = "logistic"

    def __init__(self, dim: int, temperature: float = 1.0):
        self.dim = dim
        self.temperature = temperature
        self.W = np.zeros((N_STATES, dim), dtype=np.float32)
        self.b = np.zeros(N_STATES, dtype=np.float32)
        self.mu = np.zeros(dim, dtype=np.float32)
        self.sigma = np.ones(dim, dtype=np.float32)

    def _norm(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mu) / self.sigma

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        z = self.W @ self._norm(x) + self.b
        return softmax(z / max(self.temperature, 1e-3))

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 300, lr: float = 0.3,
            l2: float = 1e-3, class_balance: bool = True, seed: int = 0,
            verbose: bool = False) -> Dict[str, float]:
        """
        Full-batch gradient descent. Deliberately not scikit-learn: this keeps
        the package dependency-free, and at these sizes (hundreds of features,
        thousands of rows) the runtime difference is irrelevant.

        Class balancing matters a lot here — a real call is mostly IVR and hold
        with a thin slice of human speech, and an unweighted fit will happily
        never predict HUMAN while scoring well on accuracy.
        """
        rng = np.random.default_rng(seed)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        self.mu = X.mean(axis=0).astype(np.float32)
        self.sigma = (X.std(axis=0) + 1e-3).astype(np.float32)
        Xn = self._norm(X)

        Y = np.zeros((len(y), N_STATES), dtype=np.float32)
        Y[np.arange(len(y)), y] = 1.0
        if class_balance:
            counts = np.bincount(y, minlength=N_STATES).astype(np.float32)
            w = np.where(counts > 0, len(y) / (N_STATES * np.maximum(counts, 1)), 0.0)
            sample_w = w[y].astype(np.float32)
        else:
            sample_w = np.ones(len(y), dtype=np.float32)
        sample_w = sample_w / sample_w.mean()

        self.W = rng.normal(0, 0.01, (N_STATES, Xn.shape[1])).astype(np.float32)
        self.b = np.zeros(N_STATES, dtype=np.float32)

        n = len(Xn)
        loss = float("nan")
        for ep in range(epochs):
            P = softmax(Xn @ self.W.T + self.b, axis=1)
            G = (P - Y) * sample_w[:, None]
            gW = G.T @ Xn / n + l2 * self.W
            gb = G.mean(axis=0)
            self.W -= lr * gW
            self.b -= lr * gb
            if verbose and ep % 50 == 0:
                loss = float(-np.mean(sample_w * np.log(P[np.arange(n), y] + 1e-9)))
                print(f"  epoch {ep:4d}  loss={loss:.4f}")

        P = softmax(Xn @ self.W.T + self.b, axis=1)
        loss = float(-np.mean(sample_w * np.log(P[np.arange(n), y] + 1e-9)))
        acc = float(np.mean(np.argmax(P, axis=1) == y))
        return {"loss": loss, "train_accuracy": acc, "n": float(n)}

    def save(self, path: str, meta: Optional[dict] = None) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez(path, W=self.W, b=self.b, mu=self.mu, sigma=self.sigma,
                 temperature=np.float32(self.temperature),
                 meta=np.array(json.dumps(meta or {})))

    @classmethod
    def load(cls, path: str) -> "LogisticStateModel":
        d = np.load(path, allow_pickle=False)
        m = cls(dim=int(d["W"].shape[1]), temperature=float(d["temperature"]))
        m.W, m.b, m.mu, m.sigma = d["W"], d["b"], d["mu"], d["sigma"]
        return m


class GRUStateModel(StateModel):
    """Causal single-layer GRU + linear head. Requires torch; opt-in."""

    name = "gru"

    def __init__(self, dim: int, hidden: int = 64):
        import torch
        import torch.nn as nn

        self.torch = torch
        self.dim = dim
        self.gru = nn.GRU(dim, hidden, batch_first=True)
        self.head = nn.Linear(hidden, N_STATES)
        self._h = None

    def reset(self) -> None:
        self._h = None

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        torch = self.torch
        with torch.no_grad():
            t = torch.tensor(x, dtype=torch.float32).view(1, 1, -1)
            out, self._h = self.gru(t, self._h)
            logits = self.head(out[:, -1])
            return torch.softmax(logits, dim=-1).numpy().ravel()

    def fit(self, sequences: List[Tuple[np.ndarray, np.ndarray]], epochs: int = 40,
            lr: float = 1e-2) -> Dict[str, float]:
        torch = self.torch
        import torch.nn as nn

        params = list(self.gru.parameters()) + list(self.head.parameters())
        opt = torch.optim.Adam(params, lr=lr)
        lossf = nn.CrossEntropyLoss()
        last = float("nan")
        for _ in range(epochs):
            tot = 0.0
            for X, y in sequences:
                opt.zero_grad()
                xt = torch.tensor(X, dtype=torch.float32).unsqueeze(0)
                yt = torch.tensor(y, dtype=torch.long)
                out, _ = self.gru(xt)
                loss = lossf(self.head(out.squeeze(0)), yt)
                loss.backward()
                opt.step()
                tot += float(loss)
            last = tot / max(len(sequences), 1)
        return {"loss": last}


def build_state_model(cfg, dim: int) -> StateModel:
    if cfg.model_path and os.path.exists(cfg.model_path):
        return LogisticStateModel.load(cfg.model_path)
    return PriorStateModel(temperature=cfg.emission_temperature)
