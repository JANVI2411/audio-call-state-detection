#!/usr/bin/env python3
"""
Score the pipeline over a labelled corpus and print an aggregate report.

Runs every `*.wav` in the corpus directory that has a matching
`*.gold.jsonl`, pools the frames, and reports state / boundary / transfer /
stability / latency / calibration metrics together. Pooling rather than
averaging per-call scores is deliberate: per-call macro-F1 on a short call
with three human frames is mostly noise, and averaging that noise hides it.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from callstate.audio.source import WavFileSource  # noqa: E402
from callstate.config import Config  # noqa: E402
from callstate.engine import CallStateEngine  # noqa: E402
from callstate.logging_setup import setup_logging  # noqa: E402
from callstate.metrics import (align_frames, boundary_metrics, calibration,  # noqa: E402
                               format_report, latency_metrics, load_gold_turns,
                               stability_metrics, state_metrics, transfer_metrics)
from callstate.semantics.asr import build_asr_backend  # noqa: E402
from callstate.telephony import TelephonyBus  # noqa: E402

CAVEAT = """
NOTE: these numbers are measured on the *synthetic* corpus (callstate/simulate.py).
Synthetic audio reproduces the cues the front-end keys on — pitch range, syllable-rate
modulation, loop periodicity, tone purity — but not codec artefacts, line noise,
crosstalk or real speaker variability. Read them as evidence that the mechanism works
end to end, not as an accuracy claim about production traffic. Re-run this script
against hand-labelled real calls before quoting any number externally.
""".strip()


def run_one(wav: str, cfg: Config, use_script: bool, asr_kind: str):
    stem = os.path.splitext(wav)[0]
    call_id = os.path.basename(stem)

    script = None
    script_path = f"{stem}.script.json"
    if use_script and os.path.exists(script_path):
        with open(script_path) as fh:
            script = [tuple(r) for r in json.load(fh)]
    backend = build_asr_backend("scripted" if script else asr_kind,
                                cfg.asr_model, cfg.asr_compute_type, script)

    tel_path = f"{stem}.telephony.jsonl"
    telephony = TelephonyBus.from_jsonl(tel_path) if os.path.exists(tel_path) else None

    source = WavFileSource(wav, target_sr=cfg.target_sr, frame_ms=cfg.frame_ms,
                           agent_channel=0)
    engine = CallStateEngine(cfg, asr_backend=backend, telephony=telephony)
    result = engine.run(source, call_id=call_id)

    turns = load_gold_turns(f"{stem}.gold.jsonl")
    gold_events = []
    if os.path.exists(f"{stem}.gold_events.json"):
        with open(f"{stem}.gold_events.json") as fh:
            gold_events = json.load(fh)
    return result, turns, gold_events


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="data/synthetic")
    ap.add_argument("--asr", default="scripted",
                    choices=["scripted", "auto", "faster_whisper", "null"])
    ap.add_argument("--model-path", default="")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--quiet", action="store_true", help="suppress per-call output")
    ap.add_argument("--holdout-from", default=None,
                    help="a *.meta.json from train_fusion.py; restricts scoring to that "
                         "run's held-out calls. Use this whenever --model-path is set, "
                         "otherwise the score includes calls the model was fitted on.")
    args = ap.parse_args()

    setup_logging("WARNING", quiet_console=args.quiet)
    cfg = Config(model_path=args.model_path)

    wavs = sorted(w for w in glob.glob(os.path.join(args.corpus, "*.wav"))
                  if os.path.exists(os.path.splitext(w)[0] + ".gold.jsonl"))
    if not wavs:
        print(f"no labelled wavs in {args.corpus} — run scripts/make_synthetic.py first")
        return 1

    if args.holdout_from:
        with open(args.holdout_from) as fh:
            keep = set(json.load(fh)["holdout_calls"])
        wavs = [w for w in wavs if os.path.basename(w) in keep]
        print(f"scoring holdout only ({len(wavs)} calls): "
              + ", ".join(os.path.basename(w) for w in wavs) + "\n")
    elif args.model_path:
        print("WARNING: --model-path without --holdout-from scores calls the model was\n"
              "         trained on. Pass --holdout-from models/<name>.meta.json.\n")

    all_true, all_pred, all_tl, all_lat = [], [], [], []
    per_call = []
    b_recall = {0.5: [0, 0], 1.0: [0, 0], 2.0: [0, 0]}
    tp = {"transfer_start": [0, 0, 0], "transfer_end": [0, 0, 0]}
    outcome_hits, outcome_n = 0, 0
    total_changes, total_gold_changes, total_dur = 0, 0, 0.0

    for wav in wavs:
        result, turns, gold_events = run_one(wav, cfg, args.asr == "scripted", args.asr)
        tl = [r.to_json() for r in result.timeline]
        evs = [e.to_json() for e in result.events]
        yt, yp = align_frames(tl, turns, offset_s=0.25)

        all_true += yt
        all_pred += yp
        all_tl += tl
        all_lat += result.latency

        sm = state_metrics(yt, yp)
        bm = boundary_metrics(tl, turns)
        tm = transfer_metrics(evs, gold_events)
        st = stability_metrics(tl, turns, turns[-1]["end_s"])

        for tol in b_recall:
            key = f"recall@{tol}s"
            if key in bm:
                b_recall[tol][0] += bm[key] * bm["n_gold_boundaries"]
                b_recall[tol][1] += bm["n_gold_boundaries"]
        for kind in tp:
            d = tm[kind]
            tp[kind][0] += d["tp"]
            tp[kind][1] += d["fp"]
            tp[kind][2] += d["fn"]
        if "outcome_accuracy" in tm:
            n_g = len([e for e in gold_events if "outcome" in e])
            outcome_hits += tm["outcome_accuracy"] * n_g
            outcome_n += n_g
        total_changes += st["n_state_changes"]
        total_gold_changes += st.get("gold_state_changes", 0)
        total_dur += turns[-1]["end_s"]

        per_call.append({
            "call": os.path.basename(wav), "accuracy": sm["accuracy"],
            "macro_f1": sm["macro_f1"], "changes": st["n_state_changes"],
            "gold_changes": st.get("gold_state_changes", 0),
            "transfer_start_f1": tm["transfer_start"]["f1"],
            "transfer_end_f1": tm["transfer_end"]["f1"],
            "has_transfers": bool(gold_events),
        })
        if not args.quiet:
            p = per_call[-1]
            tf = (f"tstart_F1={p['transfer_start_f1']:.2f} tend_F1={p['transfer_end_f1']:.2f}"
                  if p["has_transfers"] else "no transfers in gold")
            print(f"{p['call']:<26} acc={p['accuracy']:.3f} macroF1={p['macro_f1']:.3f} "
                  f"changes={p['changes']:>2}/{p['gold_changes']:<2} {tf}")

    agg = {
        "n_calls": len(wavs),
        "state": state_metrics(all_true, all_pred),
        "boundaries": {f"recall@{t}s": round(v[0] / v[1], 4) if v[1] else 0.0
                       for t, v in b_recall.items()},
        "transfers": {},
        "stability": {
            "total_state_changes": total_changes,
            "gold_state_changes": total_gold_changes,
            "changes_per_hour": round(total_changes / max(total_dur / 3600, 1e-9), 1),
            "excess_per_hour": round(max(0, total_changes - total_gold_changes)
                                     / max(total_dur / 3600, 1e-9), 1),
        },
        "latency": latency_metrics(all_lat, cfg.hop_s * 1000),
        "calibration": calibration(all_tl, all_true),
        "per_call": per_call,
    }
    for kind, (t, f, n) in tp.items():
        if t + f + n == 0:
            agg["transfers"][kind] = {"tp": 0, "fp": 0, "fn": 0, "precision": None,
                                      "recall": None, "f1": None,
                                      "note": "no gold events in this corpus; not evaluated"}
            continue
        prec = t / (t + f) if (t + f) else 0.0
        rec = t / (t + n) if (t + n) else 0.0
        agg["transfers"][kind] = {
            "tp": t, "fp": f, "fn": n, "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(2 * prec * rec / (prec + rec), 4) if (prec + rec) else 0.0,
        }
    if outcome_n:
        agg["transfers"]["outcome_accuracy"] = round(outcome_hits / outcome_n, 4)

    st = agg["state"]
    print("\n" + "=" * 68)
    print(f"AGGREGATE over {agg['n_calls']} calls, {len(all_true)} frames")
    print("=" * 68)
    print(f"accuracy={st['accuracy']:.3f}  macro_F1={st['macro_f1']:.3f}")
    print(f"\n  {'state':<8}{'prec':>8}{'recall':>8}{'F1':>8}{'support':>9}")
    for s, d in st["per_state"].items():
        print(f"  {s:<8}{d['precision']:>8.3f}{d['recall']:>8.3f}{d['f1']:>8.3f}{d['support']:>9}")
    print("\n  confusion (rows=gold, cols=pred): " + ", ".join(st["labels"]))
    for name, row in zip(st["labels"], st["confusion"]):
        print(f"    {name:<8}{row}")
    print("\nboundaries: " + "  ".join(f"{k}={v:.3f}" for k, v in agg["boundaries"].items()))
    print("transfers:")
    for kind in ("transfer_start", "transfer_end"):
        d = agg["transfers"][kind]
        if d.get("f1") is None:
            print(f"  {kind:<15} {d['note']}")
        else:
            print(f"  {kind:<15} tp={d['tp']} fp={d['fp']} fn={d['fn']}  "
                  f"P={d['precision']:.2f} R={d['recall']:.2f} F1={d['f1']:.2f}")
    if "outcome_accuracy" in agg["transfers"]:
        print(f"  outcome accuracy = {agg['transfers']['outcome_accuracy']:.2f}")
    s = agg["stability"]
    print(f"stability: {s['total_state_changes']} changes vs {s['gold_state_changes']} gold "
          f"({s['changes_per_hour']:.0f}/hr, excess {s['excess_per_hour']:.0f}/hr)")
    lat = agg["latency"]
    print(f"latency:   p50={lat['p50_ms']:.1f}ms p95={lat['p95_ms']:.1f}ms "
          f"max={lat['max_ms']:.1f}ms budget={lat['hop_budget_ms']:.0f}ms "
          f"rt_factor={lat['realtime_factor']:.3f}")
    c = agg["calibration"]
    print(f"calibration: Brier={c['brier']:.4f} ECE={c['ece']:.4f}")
    for row in c["bins"]:
        print(f"  {row['bin']:<12}n={row['n']:<5} mean_conf={row['mean_confidence']:.3f} "
              f"observed={row['observed_accuracy']:.3f} gap={row['gap']:+.3f}")
    print("\n" + CAVEAT)

    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        with open(args.out_json, "w") as fh:
            json.dump(agg, fh, indent=2)
        print(f"\nwrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
