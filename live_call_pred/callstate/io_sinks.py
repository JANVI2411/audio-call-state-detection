"""
Writes a call's results to disk in a shape that is easy to diff, grep and
replay: one JSONL per stream (timeline, events, latency) plus one JSON
summary. JSONL because these are append-only time series — a live deployment
writes them as the call happens, and a crashed process still leaves a valid,
readable partial file, which a single big JSON document would not.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

from .engine import CallResult


def _write_jsonl(path: str, rows: List[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=str) + "\n")


def write_results(result: CallResult, out_dir: str) -> Dict[str, str]:
    base = os.path.join(out_dir, result.call_id)
    paths = {
        "timeline": f"{base}.timeline.jsonl",
        "events": f"{base}.events.jsonl",
        "segments": f"{base}.segments.jsonl",
        "latency": f"{base}.latency.jsonl",
        "summary": f"{base}.summary.json",
    }
    _write_jsonl(paths["timeline"], [r.to_json() for r in result.timeline])
    _write_jsonl(paths["events"], [e.to_json() for e in result.events])
    _write_jsonl(paths["segments"], [s.to_json() for s in result.segments])
    _write_jsonl(paths["latency"], result.latency)
    os.makedirs(out_dir, exist_ok=True)
    with open(paths["summary"], "w", encoding="utf-8") as fh:
        json.dump(result.summary, fh, indent=2, default=str)
    return paths


def read_jsonl(path: str) -> List[dict]:
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def format_segments(result: CallResult) -> str:
    """The at-a-glance view of a call — what a person reads first."""
    lines = [f"{'start':>8} {'end':>8}  {'state':<7} {'speaker':<9} conf"]
    for s in result.segments:
        lines.append(f"{s.start_s:>8.1f} {s.end_s:>8.1f}  {s.state:<7} "
                     f"{(s.speaker_id or '-'):<9} {s.mean_confidence:.2f}")
    return "\n".join(lines)
