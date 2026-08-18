#!/usr/bin/env python3
"""
Fit the fusion state head on a labelled corpus and save it as a .npz.

Two methodological points that matter more than the model itself:

**The split is by call, never by frame.** Hops overlap by design — a 6 s
window advancing 0.5 s at a time means consecutive feature vectors share more
than 90% of their audio. A random frame split therefore puts near-identical
rows on both sides and reports a score that has nothing to do with
generalisation. Splitting whole calls is the only honest option.

**Temperature is fitted on held-out data, not on train.** The raw head is
overconfident (measured ECE ~0.19 before scaling). A single temperature
parameter, fitted by minimising held-out negative log-likelihood, corrects
most of that without touching the decision boundary — so accuracy is
unchanged and the confidence number becomes usable by downstream policy.
That matters here because the policy is literally "act when confident a human
is on the line".
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from callstate.audio.source import WavFileSource  # noqa: E402
from callstate.config import Config  # noqa: E402
from callstate.engine import CallStateEngine  # noqa: E402
from callstate.fusion.model import LogisticStateModel, softmax  # noqa: E402
from callstate.logging_setup import setup_logging  # noqa: E402
from callstate.metrics import gold_state_at, load_gold_turns  # noqa: E402
from callstate.semantics.asr import build_asr_backend  # noqa: E402
from callstate.telephony import TelephonyBus  # noqa: E402
from callstate.types import State  # noqa: E402


def extract_call(wav: str, cfg: Config, offset_s: float = 0.25):
    stem = os.path.splitext(wav)[0]
    script = None
    if os.path.exists(f"{stem}.script.json"):
        with open(f"{stem}.script.json") as fh:
            script = [tuple(r) for r in json.load(fh)]
    backend = build_asr_backend("scripted" if script else "null",
                                cfg.asr_model, cfg.asr_compute_type, script)
    tel = (TelephonyBus.from_jsonl(f"{stem}.telephony.jsonl")
           if os.path.exists(f"{stem}.telephony.jsonl") else None)

    source = WavFileSource(wav, target_sr=cfg.target_sr, frame_ms=cfg.frame_ms,
                           agent_channel=0)
    engine = CallStateEngine(cfg, asr_backend=backend, telephony=tel)
    engine.collect_features = True
    engine.run(source, call_id=os.path.basename(stem))

    turns = load_gold_turns(f"{stem}.gold.jsonl")
    X, y = [], []
    for t_s, vec in engine.feature_log:
        label = gold_state_at(turns, max(t_s - offset_s, 0.0))
        X.append(vec)
        y.append(State.index(label))
    return np.stack(X), np.asarray(y, dtype=np.int64)


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Grid search T minimising held-out NLL. One parameter, so a grid is fine."""
    best_t, best_nll = 1.0, float("inf")
    for t in np.arange(0.5, 4.01, 0.05):
        p = softmax(logits / t, axis=1)
        nll = float(-np.mean(np.log(p[np.arange(len(y)), y] + 1e-9)))
        if nll < best_nll:
            best_t, best_nll = float(t), nll
    return best_t


def report(name: str, model: LogisticStateModel, X: np.ndarray, y: np.ndarray) -> dict:
    P = np.stack([model.predict_proba(x) for x in X])
    pred = np.argmax(P, axis=1)
    acc = float(np.mean(pred == y))
    f1s = []
    for i in range(len(State.order())):
        tp = int(np.sum((pred == i) & (y == i)))
        fp = int(np.sum((pred == i) & (y != i)))
        fn = int(np.sum((pred != i) & (y == i)))
        if tp + fn == 0:
            continue
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn)
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    macro = float(np.mean(f1s)) if f1s else 0.0
    conf = P[np.arange(len(y)), pred]
    ece = 0.0
    for lo, hi in zip([0, .4, .6, .8, .9], [.4, .6, .8, .9, 1.01]):
        m = (conf >= lo) & (conf < hi)
        if m.any():
            ece += (m.sum() / len(conf)) * abs(conf[m].mean() - (pred == y)[m].mean())
    print(f"  {name:<12} accuracy={acc:.4f}  macro_F1={macro:.4f}  ECE={ece:.4f}  n={len(y)}")
    return {"accuracy": round(acc, 4), "macro_f1": round(macro, 4), "ece": round(float(ece), 4),
            "n": int(len(y))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="data/synthetic")
    ap.add_argument("--out", default="models/fusion_head.npz")
    ap.add_argument("--holdout", type=float, default=0.34,
                    help="fraction of CALLS held out (never frames)")
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--lr", type=float, default=0.35)
    ap.add_argument("--l2", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    setup_logging("ERROR", quiet_console=True)
    cfg = Config()

    wavs = sorted(w for w in glob.glob(os.path.join(args.corpus, "*.wav"))
                  if os.path.exists(os.path.splitext(w)[0] + ".gold.jsonl"))
    if len(wavs) < 3:
        print(f"need at least 3 labelled calls in {args.corpus}; found {len(wavs)}")
        return 1

    # Stratify by scenario, not uniformly at random. An unstratified draw on a
    # corpus this small produced a holdout containing no transfer call at all,
    # which made every transfer metric read a vacuous 1.00 — zero gold events,
    # zero predictions, nothing actually tested. Holding out at least one call
    # per scenario keeps the rare-but-important cases in the evaluation.
    rng = np.random.default_rng(args.seed)
    by_scenario: dict = {}
    for w in wavs:
        scen = os.path.basename(w).rsplit("_", 1)[0]
        by_scenario.setdefault(scen, []).append(w)

    test_wavs: List[str] = []
    for scen, group in sorted(by_scenario.items()):
        k = max(1, int(round(len(group) * args.holdout))) if len(group) > 1 else 0
        if k:
            picks = rng.permutation(len(group))[:k]
            test_wavs += [group[i] for i in picks]
    train_wavs = [w for w in wavs if w not in set(test_wavs)]
    test_wavs = sorted(test_wavs)

    print(f"corpus: {len(wavs)} calls -> train {len(train_wavs)}, holdout {len(test_wavs)}")
    print("  holdout:", ", ".join(os.path.basename(w) for w in test_wavs))

    print("\nextracting features (runs the full causal pipeline per call)...")
    Xtr, ytr = [], []
    for w in train_wavs:
        X, y = extract_call(w, cfg)
        Xtr.append(X)
        ytr.append(y)
        print(f"  {os.path.basename(w):<26} {len(y):>4} hops")
    Xtr, ytr = np.concatenate(Xtr), np.concatenate(ytr)

    Xte, yte = [], []
    for w in test_wavs:
        X, y = extract_call(w, cfg)
        Xte.append(X)
        yte.append(y)
    Xte, yte = np.concatenate(Xte), np.concatenate(yte)

    counts = np.bincount(ytr, minlength=len(State.order()))
    print("\nclass balance (train):",
          {s.value: int(c) for s, c in zip(State.order(), counts)})

    print(f"\ntraining logistic head: dim={Xtr.shape[1]} rows={len(ytr)}")
    model = LogisticStateModel(dim=Xtr.shape[1])
    stats = model.fit(Xtr, ytr, epochs=args.epochs, lr=args.lr, l2=args.l2,
                      class_balance=True, seed=args.seed, verbose=True)
    print(f"  final train loss={stats['loss']:.4f}")

    print("\nbefore temperature scaling:")
    tr_before = report("train", model, Xtr, ytr)
    te_before = report("holdout", model, Xte, yte)

    logits_te = (model._norm(Xte) @ model.W.T) + model.b
    T = fit_temperature(logits_te, yte)
    model.temperature = T
    print(f"\nfitted temperature T={T:.2f} on holdout")
    tr_after = report("train", model, Xtr, ytr)
    te_after = report("holdout", model, Xte, yte)

    meta = {
        "corpus": args.corpus, "train_calls": [os.path.basename(w) for w in train_wavs],
        "holdout_calls": [os.path.basename(w) for w in test_wavs],
        "config_fingerprint": cfg.fingerprint(), "temperature": T,
        "train": tr_after, "holdout": te_after,
        "holdout_before_temperature": te_before, "train_before_temperature": tr_before,
    }
    model.save(args.out, meta)
    with open(os.path.splitext(args.out)[0] + ".meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"\nsaved {args.out}")
    print("compare against the built-in prior weights with:")
    print(f"  python3 scripts/evaluate.py --model-path {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
