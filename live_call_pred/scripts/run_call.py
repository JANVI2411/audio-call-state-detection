#!/usr/bin/env python3
"""
Run the call-state pipeline over one WAV and write timeline / events /
latency / summary logs.

Examples
--------
# Synthetic call with its scripted transcript (no network, no models needed)
python3 scripts/run_call.py --wav data/synthetic/transfer_0.wav \\
    --call-id transfer_0 --script data/synthetic/transfer_0.script.json \\
    --telephony data/synthetic/transfer_0.telephony.jsonl --evaluate

# Real call, real ASR
python3 scripts/run_call.py --wav ../voice_agent/input/call.wav \\
    --call-id real_call --asr faster_whisper --text-encoder minilm
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from callstate.audio.source import WavFileSource  # noqa: E402
from callstate.config import Config  # noqa: E402
from callstate.engine import CallStateEngine  # noqa: E402
from callstate.io_sinks import format_segments, write_results  # noqa: E402
from callstate.logging_setup import setup_logging  # noqa: E402
from callstate.semantics.asr import build_asr_backend  # noqa: E402
from callstate.telephony import TelephonyBus  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wav", required=True)
    ap.add_argument("--call-id", default=None)
    ap.add_argument("--out", default="out")
    ap.add_argument("--log-dir", default="logs")

    ap.add_argument("--asr", default="auto",
                    choices=["auto", "faster_whisper", "scripted", "null"])
    ap.add_argument("--asr-model", default="small.en")
    ap.add_argument("--script", default=None,
                    help="JSON [[start,end,text],...] for the scripted ASR backend")
    ap.add_argument("--audio-encoder", default="logmel", choices=["logmel", "wavlm"])
    ap.add_argument("--text-encoder", default="hashed", choices=["hashed", "minilm"])
    ap.add_argument("--speaker-encoder", default="mfcc", choices=["mfcc", "ecapa"])
    ap.add_argument("--model-path", default="", help="trained fusion head (.npz)")

    ap.add_argument("--telephony", default=None, help="JSONL of carrier events")
    ap.add_argument("--agent-channel", default="auto")
    ap.add_argument("--max-duration-s", type=float, default=0.0,
                    help="process only the first N seconds (prefix, not a sample)")
    ap.add_argument("--hop-s", type=float, default=None)
    ap.add_argument("--window-s", type=float, default=None)
    ap.add_argument("--realtime", action="store_true",
                    help="pace the file at wall-clock speed, as a live leg would arrive")
    ap.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    ap.add_argument("--evaluate", action="store_true",
                    help="score against <wav-stem>.gold.jsonl if it exists")
    args = ap.parse_args()

    call_id = args.call_id or os.path.splitext(os.path.basename(args.wav))[0]
    cfg = Config(asr_backend=args.asr, asr_model=args.asr_model,
                 model_path=args.model_path, realtime=args.realtime,
                 log_level=args.log_level)
    if args.hop_s:
        cfg.hop_s = args.hop_s
    if args.window_s:
        cfg.window_s = args.window_s

    log_path = os.path.join(args.log_dir, f"{call_id}.log.jsonl")
    if os.path.exists(log_path):
        os.remove(log_path)
    log = setup_logging(cfg.log_level, jsonl_path=log_path)
    log.info("config fingerprint=%s hop=%.2fs window=%.1fs", cfg.fingerprint(),
             cfg.hop_s, cfg.window_s)

    script = None
    if args.script:
        with open(args.script) as fh:
            script = [tuple(r) for r in json.load(fh)]
    asr_backend = build_asr_backend(args.asr, args.asr_model, cfg.asr_compute_type, script)

    telephony = TelephonyBus.from_jsonl(args.telephony) if args.telephony else None

    source = WavFileSource(args.wav, target_sr=cfg.target_sr, frame_ms=cfg.frame_ms,
                           agent_channel=args.agent_channel, realtime=args.realtime,
                           max_duration_s=args.max_duration_s)
    log.info("source channels=%d source_sr=%d duration=%.1fs agent_channel=%d",
             source.n_channels, source.source_rate, source.duration_s,
             source.agent_channel_index)

    engine = CallStateEngine(cfg, asr_backend=asr_backend,
                             audio_encoder=args.audio_encoder,
                             text_encoder=args.text_encoder,
                             speaker_encoder=args.speaker_encoder,
                             telephony=telephony)
    result = engine.run(source, call_id=call_id)
    paths = write_results(result, args.out)

    print("\n=== segments ===")
    print(format_segments(result))
    print("\n=== events ===")
    if result.events:
        for e in result.events:
            extra = f" {e.meta}" if e.meta else ""
            print(f"  {e.t_s:8.1f}s  {e.type.value:<16} conf={e.confidence:.2f}  {e.evidence}{extra}")
    else:
        print("  (none)")
    print("\n=== summary ===")
    print(json.dumps(result.summary, indent=2))
    print("\nlogs:", log_path)
    for k, v in paths.items():
        print(f"  {k:<9} {v}")

    if args.evaluate:
        gold = os.path.splitext(args.wav)[0] + ".gold.jsonl"
        gold_ev = os.path.splitext(args.wav)[0] + ".gold_events.json"
        if os.path.exists(gold):
            from callstate.metrics import evaluate, format_report, load_gold_turns

            turns = load_gold_turns(gold)
            gevents = json.load(open(gold_ev)) if os.path.exists(gold_ev) else []
            m = evaluate([r.to_json() for r in result.timeline],
                         [e.to_json() for e in result.events],
                         turns, gevents, result.latency, cfg.hop_s * 1000)
            print("\n=== evaluation ===")
            print(format_report(m))
        else:
            print(f"\n(--evaluate requested but no gold file at {gold})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
