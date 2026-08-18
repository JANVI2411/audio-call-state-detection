"""
Score the pipeline against the real labelled chunks from audio_agent_dataset.

This is deliberately separate from `evaluate.py`. That script scores the
synthetic corpus, where the gold is perfect by construction and every second
carries a label. Real gold is nothing like that, and pretending otherwise
produces a confident number that means nothing:

  * The label sets differ. audio_agent_dataset uses
    {ivr, human, survey, hold, unknown}; this pipeline predicts
    {ivr, human, hold, other}. There is no `survey` state here, so a survey
    frame cannot be scored -- the model has no way to be right.

  * `unknown` is not a state, it is an absence. In the voting step, any
    stretch of time a labeler did not return is filled with `unknown`, so the
    label conflates "the audio is genuinely ambiguous" with "a model returned
    nothing here". Scoring against it would punish the pipeline for a gap in
    the gold.

So frames whose gold label is `survey` or `unknown` are EXCLUDED from
scoring, not remapped, and the excluded fraction is reported alongside every
number. A result covering 30% of a call is a real result about 30% of a call,
and saying so is the difference between a measurement and a press release.

Agreement filtering
-------------------
Each gold segment carries the fraction of labelers that agreed on it. With
three labelers that is 0.33 (all three differed), 0.67 (two agreed) or 1.0
(unanimous). `--min-agreement` restricts scoring to segments at or above a
threshold. Scores on unanimous-only gold are the most trustworthy number
available, and the gap between that and the all-frames score tells you how
much of your apparent error is really gold noise.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np  # noqa: E402

from callstate.audio.source import WavFileSource  # noqa: E402
from callstate.config import Config  # noqa: E402
from callstate.engine import CallStateEngine  # noqa: E402
from callstate.logging_setup import setup_logging  # noqa: E402
from callstate.semantics.asr import build_asr_backend  # noqa: E402

# States this pipeline can actually predict.
SCORED = ["ivr", "human", "hold", "other"]
# Gold labels that cannot be scored, and why.
UNSCORABLE = {
    "survey": "no matching state in this pipeline",
    "unknown": "absence of a label, not a state",
}


def load_real_gold(path: str) -> List[dict]:
    """Read audio_agent_dataset gold, keeping chunk-relative times."""
    out = []
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            out.append({
                "start_s": float(r["start"]),
                "end_s": float(r["end"]),
                "label": r["label"],
                "agreement": float(r.get("agreement", 0.0)),
            })
    return out


def gold_at(turns: Sequence[dict], t_s: float) -> Optional[dict]:
    for turn in turns:
        if turn["start_s"] <= t_s < turn["end_s"]:
            return turn
    return None


def confusion(y_true: Sequence[str], y_pred: Sequence[str]) -> np.ndarray:
    idx = {s: i for i, s in enumerate(SCORED)}
    m = np.zeros((len(SCORED), len(SCORED)), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            m[idx[t], idx[p]] += 1
    return m


def score(y_true: Sequence[str], y_pred: Sequence[str]) -> dict:
    m = confusion(y_true, y_pred)
    per, f1s = {}, []
    for i, s in enumerate(SCORED):
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
    total = int(m.sum())
    return {
        "n": total,
        "accuracy": round(float(np.trace(m) / total), 4) if total else 0.0,
        "macro_f1": round(float(np.mean(f1s)), 4) if f1s else 0.0,
        "per_state": per,
        "confusion": m.tolist(),
    }


def pct(values: Sequence[float], q: float) -> float:
    """Percentile without pulling in scipy; values need not be sorted."""
    if not values:
        return 0.0
    s = sorted(values)
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return float(s[i])


def latency_report(latency: List[dict], hop_s: float) -> dict:
    """
    Summarise per-hop processing cost against the live-call budget.

    The budget is `hop_s`: one hop of audio arrives every `hop_s` seconds, so
    a hop that takes longer than that to process cannot be sustained on a live
    call -- the backlog grows for as long as the condition holds. Medians are
    reassuring and mostly beside the point here; the tail is what breaks a
    deployment, which is why p95/p99/max are reported next to the count of
    hops that actually breached.
    """
    if not latency:
        return {}
    budget_ms = hop_s * 1000.0
    totals = [r["total_ms"] for r in latency]
    over = [r for r in latency if r["total_ms"] > budget_ms]
    worst = max(latency, key=lambda r: r["total_ms"])

    branches = {}
    for key in ("acoustic_ms", "asr_ms", "speaker_ms", "fusion_ms"):
        vals = [r.get(key, 0.0) for r in latency]
        branches[key.replace("_ms", "")] = {
            "median_ms": round(pct(vals, 0.50), 2),
            "p95_ms": round(pct(vals, 0.95), 2),
            "p99_ms": round(pct(vals, 0.99), 2),
            "max_ms": round(max(vals), 2),
            "mean_ms": round(sum(vals) / len(vals), 2),
        }

    return {
        "n_hops": len(latency),
        "hop_budget_ms": round(budget_ms, 1),
        "total": {
            "median_ms": round(pct(totals, 0.50), 2),
            "p95_ms": round(pct(totals, 0.95), 2),
            "p99_ms": round(pct(totals, 0.99), 2),
            "max_ms": round(max(totals), 2),
            "mean_ms": round(sum(totals) / len(totals), 2),
        },
        "branches": branches,
        "n_over_budget": len(over),
        "pct_over_budget": round(100.0 * len(over) / len(latency), 2),
        "worst_hop": {"t_s": worst["t_s"], "total_ms": worst["total_ms"]},
        "sustainable": len(over) == 0,
    }


def print_latency(rep: dict) -> None:
    if not rep:
        print("  no latency data")
        return
    print(f"  latency per hop (ms){'median':>12}{'p95':>9}{'p99':>9}{'max':>9}")
    t = rep["total"]
    print(f"    {'TOTAL':16}{t['median_ms']:12.1f}{t['p95_ms']:9.1f}"
          f"{t['p99_ms']:9.1f}{t['max_ms']:9.1f}")
    for name, b in rep["branches"].items():
        print(f"      {name:14}{b['median_ms']:12.1f}{b['p95_ms']:9.1f}"
              f"{b['p99_ms']:9.1f}{b['max_ms']:9.1f}")
    budget = rep["hop_budget_ms"]
    verdict = ("keeps up in real time" if rep["sustainable"]
               else "FALLS BEHIND on the slowest hops")
    print(f"  budget {budget:.0f}ms/hop: {rep['n_over_budget']} of "
          f"{rep['n_hops']} hops over ({rep['pct_over_budget']:.1f}%) "
          f"-> {verdict}")
    print(f"  worst hop {rep['worst_hop']['total_ms']:.1f}ms "
          f"at t={rep['worst_hop']['t_s']:.1f}s")


def run_chunk(wav: str, cfg: Config, asr_kind: str, agent_channel,
              progress_every_s: float = 0.0) -> Tuple[list, dict, list]:
    call_id = os.path.splitext(os.path.basename(wav))[0]
    backend = build_asr_backend(asr_kind, cfg.asr_model, cfg.asr_compute_type)
    source = WavFileSource(wav, target_sr=cfg.target_sr, frame_ms=cfg.frame_ms,
                           agent_channel=agent_channel)
    engine = CallStateEngine(cfg, asr_backend=backend)

    started = time.perf_counter()
    state = {"next_report_s": progress_every_s, "recent": []}

    def on_hop(t_s, row, timing, events=()):
        if not progress_every_s:
            return
        state["recent"].append(timing["total_ms"])
        if t_s < state["next_report_s"]:
            return
        state["next_report_s"] += progress_every_s
        wall = time.perf_counter() - started
        recent = state["recent"]
        state["recent"] = []
        print(f"    {t_s:6.1f}s audio | wall {wall:6.1f}s "
              f"({wall/max(t_s,1e-9):.2f}x) | state={row.state:6} "
              f"p={row.confidence:.2f} | hop ms med="
              f"{pct(recent, 0.5):6.1f} max={max(recent):6.1f}", flush=True)

    result = engine.run(source, call_id=call_id, on_hop=on_hop)
    info = {"agent_channel_index": getattr(source, "agent_channel_index", None),
            "n_channels": getattr(source, "n_channels", None),
            "wall_s": round(time.perf_counter() - started, 2),
            "audio_s": round(result.duration_s, 2)}
    return result.timeline, info, result.latency


def print_matrix(m: List[List[int]]) -> None:
    print(f"      {'':8}" + "".join(f"{s:>9}" for s in SCORED) + "   <- predicted")
    for i, s in enumerate(SCORED):
        row = "".join(f"{v:>9}" for v in m[i])
        print(f"      {s:8}" + row)
    print("      (rows = gold)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chunk-id", action="append", dest="chunk_ids", required=True,
                    help="e.g. 69f3a1e4_c00. Repeatable.")
    ap.add_argument("--chunks-dir",
                    default="../audio_agent_dataset/chunks")
    ap.add_argument("--gold-dir", default="../audio_agent_dataset/gold")
    ap.add_argument("--asr", default="faster_whisper",
                    choices=["faster_whisper", "null"],
                    help="'null' skips speech-to-text: fast, but the language "
                         "branch goes silent and IVR detection loses its "
                         "strongest cue. Use it to smoke-test, not to measure.")
    ap.add_argument("--asr-model", default="small.en")
    ap.add_argument("--agent-channel", default="auto",
                    help="'auto', or 0/1 to force which channel is our agent.")
    ap.add_argument("--min-agreement", type=float, default=0.0,
                    help="Only score gold segments at or above this labeler "
                         "agreement (0.33 / 0.67 / 1.0 with three labelers).")
    ap.add_argument("--model-path", default="",
                    help="Trained fusion head (.npz). Default: built-in weights.")
    ap.add_argument("--progress-every-s", type=float, default=30.0,
                    help="Print a progress line every N seconds of audio. "
                         "0 disables.")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--log-level", default="WARNING")
    args = ap.parse_args()

    setup_logging(level=args.log_level)
    cfg = Config(model_path=args.model_path, asr_model=args.asr_model)
    agent_channel = (args.agent_channel if args.agent_channel == "auto"
                     else int(args.agent_channel))

    pooled_true: List[str] = []
    pooled_pred: List[str] = []
    pooled_excluded: Counter = Counter()
    per_chunk = []
    all_latency: List[dict] = []

    for cid in args.chunk_ids:
        wav = os.path.join(args.chunks_dir, f"{cid}.wav")
        gold_path = os.path.join(args.gold_dir, f"{cid}.gold.jsonl")
        for p in (wav, gold_path):
            if not os.path.exists(p):
                sys.exit(f"missing: {p}")

        turns = load_real_gold(gold_path)
        print(f"\n=== {cid} " + "=" * 52, flush=True)
        print(f"  processing (progress every {args.progress_every_s:.0f}s "
              f"of audio):", flush=True)
        timeline, info, latency = run_chunk(wav, cfg, args.asr, agent_channel,
                                            args.progress_every_s)
        print(f"  audio channels={info['n_channels']} "
              f"agent_channel={info['agent_channel_index']} "
              f"(counterparty is the other one)")
        print(f"  {info['audio_s']:.0f}s audio processed in "
              f"{info['wall_s']:.0f}s wall "
              f"({info['wall_s']/max(info['audio_s'],1e-9):.2f}x real time)")
        lat = latency_report(latency, cfg.hop_s)
        print_latency(lat)
        print()

        y_true, y_pred = [], []
        excluded: Counter = Counter()
        for row in timeline:
            g = gold_at(turns, row.t_s)
            if g is None:
                excluded["no gold coverage"] += 1
                continue
            if g["label"] in UNSCORABLE:
                excluded[f"gold={g['label']}"] += 1
                continue
            if g["agreement"] < args.min_agreement - 1e-9:
                excluded[f"agreement<{args.min_agreement}"] += 1
                continue
            y_true.append(g["label"])
            y_pred.append(row.state)

        total_hops = len(timeline)
        kept = len(y_true)
        s = score(y_true, y_pred)
        print(f"  hops={total_hops}  scored={kept} "
              f"({100.0*kept/max(total_hops,1):.0f}% of the chunk)")
        if excluded:
            print("  excluded:")
            for reason, n in excluded.most_common():
                why = ""
                for k, v in UNSCORABLE.items():
                    if reason.endswith(k):
                        why = f"  ({v})"
                print(f"    {n:5} hops  {reason}{why}")
        if kept:
            print(f"  accuracy={s['accuracy']:.3f}  macro_f1={s['macro_f1']:.3f}")
            print(f"  {'state':8}{'prec':>8}{'recall':>8}{'f1':>8}{'support':>9}")
            for st in SCORED:
                d = s["per_state"][st]
                if d["support"] == 0 and st not in set(y_pred):
                    continue
                print(f"  {st:8}{d['precision']:8.3f}{d['recall']:8.3f}"
                      f"{d['f1']:8.3f}{d['support']:9}")
            print()
            print_matrix(s["confusion"])
        else:
            print("  nothing scorable in this chunk")

        pooled_true += y_true
        pooled_pred += y_pred
        pooled_excluded += excluded
        per_chunk.append({"chunk_id": cid, "n_hops": total_hops,
                          "n_scored": kept, "excluded": dict(excluded),
                          "metrics": s, "channels": info,
                          "latency": lat})
        all_latency.extend(latency)

    print("\n=== POOLED " + "=" * 52)
    pooled = score(pooled_true, pooled_pred)
    total_hops = sum(c["n_hops"] for c in per_chunk)
    print(f"  chunks={len(per_chunk)}  hops={total_hops}  "
          f"scored={pooled['n']} ({100.0*pooled['n']/max(total_hops,1):.0f}%)")
    if args.min_agreement:
        print(f"  restricted to gold with agreement >= {args.min_agreement}")
    if pooled["n"]:
        print(f"  accuracy={pooled['accuracy']:.3f}  "
              f"macro_f1={pooled['macro_f1']:.3f}")
        print(f"  {'state':8}{'prec':>8}{'recall':>8}{'f1':>8}{'support':>9}")
        for st in SCORED:
            d = pooled["per_state"][st]
            if d["support"] == 0 and st not in set(pooled_pred):
                continue
            print(f"  {st:8}{d['precision']:8.3f}{d['recall']:8.3f}"
                  f"{d['f1']:8.3f}{d['support']:9}")
        print()
        print_matrix(pooled["confusion"])

    pooled_lat = latency_report(all_latency, cfg.hop_s)
    if pooled_lat:
        print("\n--- LATENCY (all chunks pooled) " + "-" * 31)
        print_latency(pooled_lat)
        total_audio = sum(c["channels"]["audio_s"] for c in per_chunk)
        total_wall = sum(c["channels"]["wall_s"] for c in per_chunk)
        print(f"  {total_audio:.0f}s audio in {total_wall:.0f}s wall "
              f"({total_wall/max(total_audio,1e-9):.2f}x real time)")
        print("\n  For a live call the per-hop budget is what matters, not the\n"
              "  real-time factor: audio arrives on a clock and cannot be\n"
              "  batched. A run that averages 0.4x real time still stalls if\n"
              "  individual hops exceed the budget, and the hops that trigger\n"
              "  speech recognition are the ones that do.")

    print("\n  NOTE: measured on real payer calls, with gold produced by a "
          "3-model\n  vote that has NOT been human-reviewed. Frames labelled "
          "`unknown` or\n  `survey` are excluded, not counted as errors. Treat "
          "the excluded\n  fraction as part of the result.")

    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        with open(args.out_json, "w") as fh:
            json.dump({"per_chunk": per_chunk, "pooled": pooled,
                       "pooled_latency": pooled_lat,
                       "hop_s": cfg.hop_s,
                       "asr": args.asr,
                       "min_agreement": args.min_agreement,
                       "excluded_total": dict(pooled_excluded)}, fh, indent=2)
        print(f"\n  written to {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
