"""
Evaluation. Chunk accuracy alone is close to useless for this problem, so it
is reported but never on its own.

Four families, because four different things can be wrong:

**State quality** — per-state precision/recall/F1 and macro-F1. Macro, not
micro: a call is mostly IVR and hold, so a model that never predicts HUMAN can
still post high overall accuracy while being useless for the one decision the
voice agent actually cares about.

**Boundary quality** — did the transition land within ±0.5 s / ±1 s of truth.
A model can score well per-chunk and still be late on every boundary, which is
precisely the failure that hurts a live agent: it keeps talking into an IVR
menu for two seconds after a human picks up.

**Transfer quality** — precision/recall/F1 over transfer events with a time
tolerance, plus outcome correctness (completed vs. failed). This is scored
separately because transfers are rare, so they contribute almost nothing to
frame-level metrics while being the highest-value thing to get right.

**Stability & latency** — false state changes per hour (flicker), plus
detection latency percentiles, with human-detection latency broken out because
that is the moment the agent's behaviour must change.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .types import State

STATES = [s.value for s in State.order()]


def load_gold_turns(path: str) -> List[dict]:
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def gold_state_at(turns: Sequence[dict], t_s: float) -> str:
    for turn in turns:
        if turn["start_s"] <= t_s < turn["end_s"]:
            return turn["state"]
    return "other"


def align_frames(timeline: List[dict], turns: Sequence[dict],
                 offset_s: float = 0.0) -> Tuple[List[str], List[str]]:
    """
    Sample gold at the *midpoint* of each hop's committed interval.

    `offset_s` exists because a causal system is structurally late: at time t
    it has only seen up to t, so it cannot know about a boundary at t until
    slightly after. Comparing against gold at exactly t therefore penalises
    the system for obeying causality. Sampling gold slightly behind the hop
    (default: half a hop) is the fair comparison, and the boundary metrics
    below measure the remaining lag explicitly rather than hiding it.
    """
    y_true, y_pred = [], []
    for row in timeline:
        t = row["t_s"] - offset_s
        y_true.append(gold_state_at(turns, max(t, 0.0)))
        y_pred.append(row["state"])
    return y_true, y_pred


def confusion(y_true: Sequence[str], y_pred: Sequence[str]) -> np.ndarray:
    idx = {s: i for i, s in enumerate(STATES)}
    m = np.zeros((len(STATES), len(STATES)), dtype=int)
    for a, b in zip(y_true, y_pred):
        if a in idx and b in idx:
            m[idx[a], idx[b]] += 1
    return m


def state_metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> Dict[str, object]:
    m = confusion(y_true, y_pred)
    per: Dict[str, Dict[str, float]] = {}
    f1s = []
    for i, s in enumerate(STATES):
        tp = int(m[i, i])
        fp = int(m[:, i].sum() - tp)
        fn = int(m[i, :].sum() - tp)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per[s] = {"precision": round(prec, 4), "recall": round(rec, 4),
                  "f1": round(f1, 4), "support": int(m[i, :].sum())}
        if m[i, :].sum() > 0:
            f1s.append(f1)
    total = m.sum()
    return {
        "accuracy": round(float(np.trace(m) / total), 4) if total else 0.0,
        "macro_f1": round(float(np.mean(f1s)), 4) if f1s else 0.0,
        "per_state": per,
        "confusion": m.tolist(),
        "labels": STATES,
    }


def _boundaries(pairs: List[Tuple[float, str]]) -> List[Tuple[float, str, str]]:
    out = []
    for (t0, s0), (t1, s1) in zip(pairs, pairs[1:]):
        if s0 != s1:
            out.append((t1, s0, s1))
    return out


def boundary_metrics(timeline: List[dict], turns: Sequence[dict],
                     tolerances: Sequence[float] = (0.5, 1.0, 2.0)) -> Dict[str, object]:
    gold_b = [(t["start_s"], t["state"]) for t in turns]
    gold_changes = [(t, a, b) for t, a, b in _boundaries(gold_b)]
    pred_changes = _boundaries([(r["t_s"], r["state"]) for r in timeline])

    out: Dict[str, object] = {"n_gold_boundaries": len(gold_changes),
                              "n_pred_boundaries": len(pred_changes)}
    lags: List[float] = []
    human_lags: List[float] = []
    for tol in tolerances:
        matched = set()
        hits = 0
        for gt, _ga, gb in gold_changes:
            best, best_j = None, None
            for j, (pt, _pa, pb) in enumerate(pred_changes):
                if j in matched or pb != gb:
                    continue
                d = pt - gt
                if -tol <= d <= tol and (best is None or abs(d) < abs(best)):
                    best, best_j = d, j
            if best is not None:
                hits += 1
                matched.add(best_j)
                if tol == max(tolerances):
                    lags.append(best)
                    if gb == "human":
                        human_lags.append(best)
        out[f"recall@{tol}s"] = round(hits / len(gold_changes), 4) if gold_changes else 0.0
    if lags:
        out["boundary_lag_s"] = {
            "mean": round(float(np.mean(lags)), 3),
            "p50": round(float(np.percentile(lags, 50)), 3),
            "p95": round(float(np.percentile(lags, 95)), 3),
        }
    if human_lags:
        out["human_detection_lag_s"] = {
            "mean": round(float(np.mean(human_lags)), 3),
            "p95": round(float(np.percentile(human_lags, 95)), 3),
            "n": len(human_lags),
        }
    return out


def transfer_metrics(pred_events: List[dict], gold_events: List[dict],
                     tolerance_s: float = 8.0) -> Dict[str, object]:
    def match(kind: str) -> Tuple[int, int, int, List[bool]]:
        g = [e for e in gold_events if e["type"] == kind]
        p = [e for e in pred_events if e["type"] == kind]
        used, tp = set(), 0
        outcome_ok: List[bool] = []
        for ge in g:
            best_j, best_d = None, None
            for j, pe in enumerate(p):
                if j in used:
                    continue
                d = abs(pe["t_s"] - ge["t_s"])
                if d <= tolerance_s and (best_d is None or d < best_d):
                    best_j, best_d = j, d
            if best_j is not None:
                tp += 1
                used.add(best_j)
                if "outcome" in ge:
                    outcome_ok.append(
                        (p[best_j].get("meta", {}) or {}).get("outcome") == ge["outcome"])
        return tp, len(p) - tp, len(g) - tp, outcome_ok

    out: Dict[str, object] = {"tolerance_s": tolerance_s}
    all_outcomes: List[bool] = []
    for kind in ("transfer_start", "transfer_end"):
        tp, fp, fn, oks = match(kind)
        # With no gold events and no predictions there is nothing to score.
        # Reporting 1.00 there is worse than useless — it reads as a perfect
        # result on a corpus that never tested the capability at all.
        if tp + fp + fn == 0:
            out[kind] = {"tp": 0, "fp": 0, "fn": 0, "precision": None,
                         "recall": None, "f1": None, "note": "no gold events; not evaluated"}
            continue
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out[kind] = {"tp": tp, "fp": fp, "fn": fn, "precision": round(prec, 4),
                     "recall": round(rec, 4), "f1": round(f1, 4)}
        all_outcomes.extend(oks)
    if all_outcomes:
        out["outcome_accuracy"] = round(float(np.mean(all_outcomes)), 4)
    return out


def stability_metrics(timeline: List[dict], turns: Optional[Sequence[dict]] = None,
                      duration_s: Optional[float] = None) -> Dict[str, object]:
    states = [r["state"] for r in timeline]
    changes = sum(1 for a, b in zip(states, states[1:]) if a != b)
    dur = duration_s or (timeline[-1]["t_s"] if timeline else 0.0)
    out: Dict[str, object] = {
        "n_state_changes": changes,
        "state_changes_per_hour": round(changes / max(dur / 3600.0, 1e-9), 1),
    }
    if turns is not None:
        gold_changes = sum(1 for a, b in zip(turns, turns[1:]) if a["state"] != b["state"])
        out["gold_state_changes"] = gold_changes
        out["excess_changes_per_hour"] = round(
            max(0, changes - gold_changes) / max(dur / 3600.0, 1e-9), 1)
    return out


def latency_metrics(latency_rows: List[dict], hop_budget_ms: float) -> Dict[str, object]:
    if not latency_rows:
        return {}
    tot = np.array([r["total_ms"] for r in latency_rows])
    branches = {}
    for k in ("acoustic_ms", "asr_ms", "speaker_ms", "fusion_ms"):
        vals = [r[k] for r in latency_rows if k in r]
        if vals:
            branches[k] = {"mean": round(float(np.mean(vals)), 2),
                           "p95": round(float(np.percentile(vals, 95)), 2)}
    return {
        "hop_budget_ms": hop_budget_ms,
        "p50_ms": round(float(np.percentile(tot, 50)), 2),
        "p95_ms": round(float(np.percentile(tot, 95)), 2),
        "max_ms": round(float(np.max(tot)), 2),
        "over_budget_fraction": round(float(np.mean(tot > hop_budget_ms)), 4),
        "realtime_factor": round(float(np.mean(tot) / hop_budget_ms), 4),
        "by_branch": branches,
    }


def calibration(timeline: List[dict], y_true: Sequence[str],
                bins: Sequence[float] = (0.0, 0.4, 0.6, 0.8, 0.9, 1.01)) -> Dict[str, object]:
    """
    Reliability table plus Brier score and ECE.

    Reported because downstream policy uses the confidence number: "act only
    if we are >0.8 sure a human is on the line" is meaningless if 0.8 does not
    correspond to being right 80% of the time.
    """
    conf = np.array([r["confidence"] for r in timeline])
    correct = np.array([p["state"] == t for p, t in zip(timeline, y_true)], dtype=float)
    rows, ece = [], 0.0
    for lo, hi in zip(bins, bins[1:]):
        m = (conf >= lo) & (conf < hi)
        if not m.any():
            continue
        acc, avg_conf, n = float(correct[m].mean()), float(conf[m].mean()), int(m.sum())
        rows.append({"bin": f"[{lo:.1f},{hi:.1f})", "n": n,
                     "mean_confidence": round(avg_conf, 4),
                     "observed_accuracy": round(acc, 4),
                     "gap": round(avg_conf - acc, 4)})
        ece += (n / len(conf)) * abs(avg_conf - acc)
    brier = float(np.mean((conf - correct) ** 2)) if len(conf) else 0.0
    return {"bins": rows, "brier": round(brier, 4), "ece": round(ece, 4)}


def evaluate(timeline: List[dict], events: List[dict], turns: Sequence[dict],
             gold_events: List[dict], latency_rows: List[dict],
             hop_budget_ms: float, offset_s: float = 0.25) -> Dict[str, object]:
    y_true, y_pred = align_frames(timeline, turns, offset_s=offset_s)
    duration = turns[-1]["end_s"] if turns else None
    return {
        "state": state_metrics(y_true, y_pred),
        "boundaries": boundary_metrics(timeline, turns),
        "transfers": transfer_metrics(events, gold_events),
        "stability": stability_metrics(timeline, turns, duration),
        "latency": latency_metrics(latency_rows, hop_budget_ms),
        "calibration": calibration(timeline, y_true),
        "n_frames": len(y_true),
    }


def format_report(m: Dict[str, object]) -> str:
    """Human-readable rendering of `evaluate`'s output."""
    L: List[str] = []
    st = m["state"]  # type: ignore[index]
    L.append(f"frames={m['n_frames']}  accuracy={st['accuracy']:.3f}  macro_F1={st['macro_f1']:.3f}")
    L.append("")
    L.append(f"  {'state':<8}{'prec':>8}{'recall':>8}{'F1':>8}{'support':>9}")
    for s, d in st["per_state"].items():  # type: ignore[index]
        L.append(f"  {s:<8}{d['precision']:>8.3f}{d['recall']:>8.3f}{d['f1']:>8.3f}{d['support']:>9}")
    L.append("")
    L.append("  confusion (rows=gold, cols=pred): " + ", ".join(st["labels"]))  # type: ignore[index]
    for name, row in zip(st["labels"], st["confusion"]):  # type: ignore[index]
        L.append(f"    {name:<8}{row}")

    b = m["boundaries"]  # type: ignore[index]
    L.append("")
    L.append(f"boundaries: gold={b['n_gold_boundaries']} pred={b['n_pred_boundaries']}  "
             + "  ".join(f"{k}={v:.3f}" for k, v in b.items() if str(k).startswith("recall@")))
    if "boundary_lag_s" in b:
        lag = b["boundary_lag_s"]
        L.append(f"  detection lag  mean={lag['mean']:+.2f}s  p50={lag['p50']:+.2f}s  p95={lag['p95']:+.2f}s")
    if "human_detection_lag_s" in b:
        h = b["human_detection_lag_s"]
        L.append(f"  human-detection lag  mean={h['mean']:+.2f}s  p95={h['p95']:+.2f}s  (n={h['n']})")

    t = m["transfers"]  # type: ignore[index]
    L.append("")
    L.append("transfers:")
    for kind in ("transfer_start", "transfer_end"):
        d = t[kind]
        if d.get("f1") is None:
            L.append(f"  {kind:<15} {d['note']}")
        else:
            L.append(f"  {kind:<15} tp={d['tp']} fp={d['fp']} fn={d['fn']}  "
                     f"P={d['precision']:.2f} R={d['recall']:.2f} F1={d['f1']:.2f}")
    if "outcome_accuracy" in t:
        L.append(f"  outcome (completed vs failed) accuracy = {t['outcome_accuracy']:.2f}")

    s = m["stability"]  # type: ignore[index]
    L.append("")
    L.append(f"stability: {s['n_state_changes']} changes "
             f"({s['state_changes_per_hour']:.0f}/hr)"
             + (f", gold {s['gold_state_changes']}, excess {s['excess_changes_per_hour']:.0f}/hr"
                if "gold_state_changes" in s else ""))

    lat = m["latency"]  # type: ignore[index]
    if lat:
        L.append(f"latency:   p50={lat['p50_ms']:.1f}ms p95={lat['p95_ms']:.1f}ms "
                 f"max={lat['max_ms']:.1f}ms  budget={lat['hop_budget_ms']:.0f}ms  "
                 f"over-budget={lat['over_budget_fraction']*100:.1f}%")
        for k, v in lat.get("by_branch", {}).items():
            L.append(f"   {k:<13} mean={v['mean']:>7.2f}ms  p95={v['p95']:>7.2f}ms")

    c = m["calibration"]  # type: ignore[index]
    L.append("")
    L.append(f"calibration: Brier={c['brier']:.4f}  ECE={c['ece']:.4f}")
    L.append(f"  {'bin':<12}{'n':>6}{'mean_conf':>11}{'observed':>10}{'gap':>8}")
    for row in c["bins"]:  # type: ignore[index]
        L.append(f"  {row['bin']:<12}{row['n']:>6}{row['mean_confidence']:>11.3f}"
                 f"{row['observed_accuracy']:>10.3f}{row['gap']:>+8.3f}")
    return "\n".join(L)
