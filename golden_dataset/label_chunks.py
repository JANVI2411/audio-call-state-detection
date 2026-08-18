"""
Label chunks produced by make_chunks.py.

Run make_chunks.py first. This reads chunks/manifest.json, picks which pieces
to label, and runs the three-labeler pipeline on each one.

Every output file is named after the chunk it came from, so nothing collides:

    gold/model_outputs/69f3a1e4_c00.gemini.json
    gold/model_outputs/69f3a1e4_c00.gpt_audio.json
    gold/model_outputs/69f3a1e4_c00.asr_llm.json
    gold/69f3a1e4_c00.gold.jsonl
    gold/review_queue.69f3a1e4_c00.jsonl

Default selection is the first chunk of each recording (--first-only), which
is the cheap way to sanity-check the labels before paying for the rest.

Costs money: each chunk is sent to two audio models and one text model.
Already-finished labelers are cached on disk, so re-running after a failure
only pays for the ones that did not complete. Pass --force-labelers to
override that.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import run_pipeline


def load_manifest(path):
    if not os.path.exists(path):
        sys.exit(f"No manifest at {path}. Run make_chunks.py first.")
    with open(path) as f:
        return json.load(f)


def select(chunks, first_only, only_ids):
    if only_ids:
        wanted = set(only_ids)
        picked = [c for c in chunks if c["chunk_id"] in wanted]
        missing = wanted - {c["chunk_id"] for c in picked}
        if missing:
            sys.exit(f"Unknown chunk id(s): {', '.join(sorted(missing))}")
        return picked
    if first_only:
        # One chunk per source recording: the lowest index of each.
        by_call = {}
        for c in sorted(chunks, key=lambda c: (c["source_call"], c["index"])):
            by_call.setdefault(c["source_call"], c)
        return list(by_call.values())
    return chunks


def main():
    ap = argparse.ArgumentParser(description="Label call chunks.")
    ap.add_argument("--manifest", default="./chunks/manifest.json")
    ap.add_argument("--out-dir", default="./gold")
    ap.add_argument("--first-only", action="store_true",
                    help="Label only the first chunk of each recording.")
    ap.add_argument("--chunk-id", action="append", dest="chunk_ids",
                    help="Label a specific chunk. Repeatable.")
    ap.add_argument("--force-labelers", action="store_true",
                    help="Re-run labelers even if cached output exists.")
    ap.add_argument("--grid-vote", action="store_true",
                    help="Use the old fixed 250ms voting grid.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would be labeled, call no APIs.")
    args = ap.parse_args()

    chunks = load_manifest(args.manifest)
    picked = select(chunks, args.first_only, args.chunk_ids)

    total_min = sum(c["duration_s"] for c in picked) / 60.0
    print(f"Selected {len(picked)} chunk(s), {total_min:.1f} minutes of audio:")
    for c in picked:
        end = c["offset_s"] + c["duration_s"]
        print(f"  {c['chunk_id']:16} from {c['source_call'][:8]}  "
              f"{c['offset_s']:6.0f}s-{end:6.0f}s  ({c['duration_s']:.0f}s)")

    if args.dry_run:
        print("\nDry run: nothing sent, no cost incurred.")
        return

    print(f"\nEach chunk goes to 3 labelers. Starting.\n")
    started = time.perf_counter()
    done, failed = [], []

    for i, c in enumerate(picked, 1):
        print(f"--- [{i}/{len(picked)}] {c['chunk_id']} " + "-" * 40)
        if not os.path.exists(c["path"]):
            print(f"  missing audio: {c['path']}  (skipped)")
            failed.append((c["chunk_id"], "audio file missing"))
            continue
        try:
            summary = run_pipeline.run(
                audio_path=c["path"],
                call_id=c["chunk_id"],
                out_dir=args.out_dir,
                force_labelers=args.force_labelers,
                offset_seconds=c["offset_s"],
                grid_vote=args.grid_vote,
            )
            done.append(summary)
        except Exception as exc:
            # One chunk failing must not lose the chunks already paid for.
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            failed.append((c["chunk_id"], f"{type(exc).__name__}: {exc}"))
        print()

    wall = time.perf_counter() - started
    print("=" * 72)
    print(f"Finished in {wall:.0f}s: {len(done)} labeled, {len(failed)} failed\n")

    if done:
        print(f"  {'chunk':16} {'segs':>5} {'review':>7} {'audio':>7} "
              f"{'wall':>7} {'speed':>8} {'cost':>10}")
        for s in done:
            rtf = s["wall_s"] / s["duration_s"] if s["duration_s"] else 0
            cost = (f"${s['cost_usd']:.4f}" if s["cost_complete"]
                    else f"${s['cost_usd']:.4f}+?")
            print(f"  {s['call_id']:16} {s['n_segments']:5} {s['n_review']:7} "
                  f"{s['duration_s']:6.0f}s {s['wall_s']:6.0f}s "
                  f"{rtf:7.2f}x {cost:>10}")

        # Per-labeler breakdown, so a slow or expensive one is obvious.
        print(f"\n  {'labeler':12} {'calls':>6} {'total s':>9} {'avg s':>8} "
              f"{'cost':>10}")
        by_labeler = {}
        for s in done:
            for r in s["labelers"]:
                b = by_labeler.setdefault(r["labeler"],
                                          {"n": 0, "t": 0.0, "c": 0.0,
                                           "unknown": False})
                b["n"] += 1
                b["t"] += r.get("latency_s") or 0.0
                if r.get("cost_usd") is None:
                    if r.get("status") == "ok":
                        b["unknown"] = True
                else:
                    b["c"] += r["cost_usd"]
        for name, b in by_labeler.items():
            cost = f"${b['c']:.4f}" + ("+?" if b["unknown"] else "")
            print(f"  {name:12} {b['n']:6} {b['t']:8.1f}s "
                  f"{b['t']/max(b['n'],1):7.1f}s {cost:>10}")

        total_cost = sum(s["cost_usd"] for s in done)
        total_audio = sum(s["duration_s"] for s in done)
        any_unknown = any(not s["cost_complete"] for s in done)
        print(f"\n  audio labeled : {total_audio/60:.1f} min")
        print(f"  wall time     : {wall/60:.1f} min "
              f"({wall/total_audio:.2f}x real time)")
        print(f"  cost          : ${total_cost:.4f}" +
              ("  (partial -- some labelers reported no usage)"
               if any_unknown else ""))
        if total_audio and total_cost:
            per_min = total_cost / (total_audio / 60.0)
            print(f"  cost per min  : ${per_min:.4f}"
                  f"{'+' if any_unknown else ''}")
            print(f"  projected 1h  : ${per_min * 60:.2f}"
                  f"{'+' if any_unknown else ''} of audio")
        print(f"\n  per-chunk detail: {args.out_dir}/metrics/<chunk>.metrics.json")

    for cid, why in failed:
        print(f"  {cid:16} FAILED: {why}")
    if failed:
        print("\nRe-run the same command to retry only the failed parts "
              "(finished labelers are cached, so you won't pay twice).")


if __name__ == "__main__":
    main()
