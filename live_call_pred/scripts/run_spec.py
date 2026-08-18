"""
Run one call and emit the section 4.5 deliverables.

    python3 -u scripts/run_spec.py --wav <file.wav> --hop-s 2.0

Produces, in --out:
    <call>.timeline.jsonl   segments: start, end, label, sub_label,
                            human_id, confidence, evidence
    <call>.events.jsonl     transfer lifecycle transitions
    <call>.summary.json     humans, transfers attempted vs completed,
                            dominant counterparty type
    <call>.hops.jsonl       per-decision detail, for debugging a bad segment
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from callstate.audio.source import WavFileSource  # noqa: E402
from callstate.config import Config  # noqa: E402
from callstate.engine import CallStateEngine  # noqa: E402
from callstate.export_spec import (format_timeline, summary_row,  # noqa: E402
                                   write_spec_outputs)
from callstate.live_sink import LiveSink, load_gold  # noqa: E402
from callstate.logging_setup import setup_logging  # noqa: E402
from callstate.segmenter import SilenceSegmenter  # noqa: E402
from callstate.semantics.asr import build_asr_backend  # noqa: E402
from callstate.telephony import TelephonyBus  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wav", required=True)
    ap.add_argument("--call-id", default=None)
    ap.add_argument("--out", default="out/spec")
    ap.add_argument("--hop-s", type=float, default=2.0,
                    help="How often to decide, in seconds. Also the latency "
                         "budget: audio arrives on a clock, so a hop that "
                         "takes longer than this cannot be sustained live.")
    ap.add_argument("--window-s", type=float, default=None,
                    help="How far back each decision looks. Defaults to the "
                         "configured 6s.")
    ap.add_argument("--asr", default="faster_whisper",
                    choices=["faster_whisper", "null", "auto"])
    ap.add_argument("--asr-model", default="small.en")
    ap.add_argument("--agent-channel", default="auto")
    ap.add_argument("--max-duration-s", type=float, default=0.0)
    ap.add_argument("--telephony", default=None)
    ap.add_argument("--model-path", default="")
    ap.add_argument("--progress-every-s", type=float, default=30.0,
                    help="Heartbeat gap for quiet stretches. State changes "
                         "and events always print immediately.")
    ap.add_argument("--every-hop", action="store_true",
                    help="Print every single decision, not just changes.")
    ap.add_argument("--segment", default="fixed", choices=["fixed", "silence"],
                    help="'fixed' decides every --hop-s seconds. 'silence' "
                         "decides at silence gaps instead, capped by "
                         "--max-segment-s so latency stays bounded.")
    ap.add_argument("--min-silence-ms", type=int, default=400)
    ap.add_argument("--max-segment-s", type=float, default=8.0)
    ap.add_argument("--gold", default=None,
                    help="Path to a golden_dataset .gold.jsonl. Shows the "
                         "answer key next to each prediction and a running "
                         "accuracy. Pass 'auto' to look it up by call id.")
    ap.add_argument("--gold-dir", default="../golden_dataset/gold",
                    help="Where --gold auto looks.")
    ap.add_argument("--no-live", action="store_true",
                    help="Old behaviour: stay silent, write only at the end.")
    ap.add_argument("--log-level", default="WARNING")
    args = ap.parse_args()

    call_id = args.call_id or os.path.splitext(os.path.basename(args.wav))[0]
    setup_logging(args.log_level)

    cfg = Config(asr_backend=args.asr, asr_model=args.asr_model,
                 model_path=args.model_path)
    cfg.hop_s = args.hop_s
    if args.window_s:
        cfg.window_s = args.window_s

    backend = build_asr_backend(args.asr, args.asr_model, cfg.asr_compute_type)
    telephony = TelephonyBus.from_jsonl(args.telephony) if args.telephony else None
    source = WavFileSource(args.wav, target_sr=cfg.target_sr,
                           frame_ms=cfg.frame_ms, agent_channel=args.agent_channel,
                           max_duration_s=args.max_duration_s)
    engine = CallStateEngine(cfg, asr_backend=backend, telephony=telephony)

    print(f"call_id={call_id}")
    print(f"  audio {source.duration_s:.1f}s, {source.n_channels} channels, "
          f"agent on channel {source.agent_channel_index}")
    mode = (f"deciding every {cfg.hop_s:.1f}s" if args.segment == "fixed"
            else "deciding at silence boundaries")
    print(f"  {mode}, looking back {cfg.window_s:.1f}s, asr={backend.name}")
    print()

    gold = None
    if args.gold:
        gold_path = (os.path.join(args.gold_dir, f"{call_id}.gold.jsonl")
                     if args.gold == "auto" else args.gold)
        if os.path.exists(gold_path):
            gold = load_gold(gold_path)
            print(f"  scoring live against {gold_path} ({len(gold)} segments)")
        else:
            print(f"  no gold at {gold_path} -- running without scoring")

    started = time.perf_counter()
    sink = None
    if not args.no_live:
        sink = LiveSink(args.out, call_id, cfg.hop_s,
                        print_every_hop=args.every_hop,
                        min_print_gap_s=args.progress_every_s,
                        gold=gold)
        legend = ("  * = state change   OK = matches gold   <-- = disagrees"
                  if gold else "  * marks a state change")
        print(legend + "; every decision hits disk as it happens\n")

    def on_hop(t_s, row, timing, events=()):
        if sink is None:
            return
        sink.hop(t_s, row, timing)
        for ev in events:
            sink.event(ev)

    segmenter = None
    if args.segment == "silence":
        segmenter = SilenceSegmenter(sample_rate=cfg.target_sr,
                                     frame_ms=cfg.frame_ms,
                                     min_silence_ms=args.min_silence_ms,
                                     max_segment_s=args.max_segment_s)
        print(f"  segmenting on silence (gap >= {args.min_silence_ms}ms, "
              f"cap {args.max_segment_s:.0f}s)")
    result = engine.run(source, call_id=call_id, on_hop=on_hop,
                        segmenter=segmenter)
    wall = time.perf_counter() - started
    live = sink.close() if sink else None

    print("\n=== 1. TIMELINE ===")
    print(format_timeline(result.segments))

    print("\n=== 2. EVENTS ===")
    if result.events:
        for e in result.events:
            tag = "lifecycle" if e.type.value.startswith("transfer") else "boundary"
            extra = f"  {e.meta}" if e.meta else ""
            print(f"  {e.t_s:8.2f}s  {e.type.value:<16} [{tag}] "
                  f"conf={e.confidence:.2f}  {e.evidence}{extra}")
    else:
        print("  (none)")

    print("\n=== 3. SUMMARY ===")
    print(json.dumps(summary_row(result, engine, source), indent=2))

    # The batch write still runs: it produces the segment view, which only
    # exists once the whole call is known. The per-decision record it also
    # writes is identical to what the live sink already flushed.
    paths = write_spec_outputs(result, engine, args.out, source)
    if live:
        print(f"\n  live: {live['n_hops']} decisions, "
              f"{live['n_over_budget']} over budget, streamed to "
              f"{live['hops_path']}")
        if "accuracy" in live:
            print(f"\n=== 4. ACCURACY vs GOLD ===")
            print(f"  scored {live['n_scored']} of {live['n_hops']} decisions "
                  f"({live['n_skipped']} skipped: gold was survey/unknown)")
            print(f"  accuracy = {live['accuracy']:.3f}")
            print("  where it went wrong (gold -> predicted):")
            for k, n in list(live["confusion"].items())[:8]:
                t, p = k.split("->")
                flag = "   OK" if t == p else "  <--"
                print(f"    {k:22} {n:4}{flag}")
    print(f"\nprocessed {result.duration_s:.0f}s of audio in {wall:.0f}s "
          f"({wall/max(result.duration_s,1e-9):.2f}x real time)")
    print("written:")
    for k, v in paths.items():
        print(f"  {k:<9} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
