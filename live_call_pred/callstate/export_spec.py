"""
Write a call's results in the deliverable format from section 4.5.

Three files per call:

  <call>.timeline.jsonl   one line per segment:
                          {start, end, label, sub_label, human_id,
                           confidence, evidence}
  <call>.events.jsonl     one line per transfer-lifecycle transition
  <call>.summary.json     counts of humans, transfers attempted vs completed,
                          dominant counterparty type

This is a *view*, not a second source of truth -- every field is read off the
engine's own output. It exists because the internal names (`state_s`,
`speaker_id`, `mean_confidence`) are the pipeline's vocabulary, while the
spec's names are the vocabulary of whoever consumes the dataset, and quietly
renaming one to look like the other inside the engine would be worse than
keeping an explicit translation in one place.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

from .types import Event, Segment

# Transfer lifecycle, in the spec's sense: the events that describe a call
# moving from one party to another. Speaker changes and IVR exits are useful
# but are boundary markers, not lifecycle transitions, so they are tagged
# separately rather than mixed in.
LIFECYCLE = {"transfer_start", "transfer_end"}


def segment_rows(segments: List[Segment]) -> List[dict]:
    return [{
        "start": round(s.start_s, 2),
        "end": round(s.end_s, 2),
        "label": s.state,
        "sub_label": s.sub_label,
        "human_id": s.speaker_id,
        "confidence": round(s.mean_confidence, 3),
        "evidence": s.evidence,
    } for s in segments]


def event_rows(events: List[Event]) -> List[dict]:
    out = []
    for e in events:
        row = {
            "t": round(e.t_s, 2),
            "type": e.type.value,
            "lifecycle": e.type.value in LIFECYCLE,
            "confidence": round(e.confidence, 3),
            "evidence": e.evidence,
        }
        if e.meta:
            row.update({k: v for k, v in e.meta.items()})
        out.append(row)
    return out


def summary_row(result, engine, source=None) -> dict:
    """
    The spec's three questions: how many humans, how many transfers were
    attempted versus completed, and what was this call mostly talking to.

    `n_transfers_attempted` counts announcements, not successes, because an
    announcement is a prediction the call makes about itself and predictions
    are allowed to be wrong. The gap between attempted and completed is the
    interesting number -- a system that only reported completions would look
    flawless while silently dropping every failed handoff.
    """
    durations: Dict[str, float] = {}
    for s in result.segments:
        durations[s.state] = durations.get(s.state, 0.0) + (s.end_s - s.start_s)

    ends = [e for e in result.events if e.type.value == "transfer_end"]
    completed = sum(1 for e in ends if e.meta.get("outcome") == "completed")
    failed = sum(1 for e in ends if e.meta.get("outcome") == "failed")
    attempted = sum(1 for e in result.events if e.type.value == "transfer_start")

    humans = sorted(engine.speaker.registry.centroids.keys())
    lat = [r["total_ms"] for r in result.latency] or [0.0]
    budget_ms = engine.cfg.hop_s * 1000.0
    over = [x for x in lat if x > budget_ms]

    return {
        "call_id": result.call_id,
        "duration_s": round(result.duration_s, 2),

        "n_humans": len(humans),
        "human_ids": humans,

        "n_transfers_attempted": attempted,
        "n_transfers_completed": completed,
        "n_transfers_failed": failed,

        "dominant_label": (max(durations, key=durations.get)
                           if durations else "unknown"),
        "label_durations_s": {k: round(v, 1) for k, v in
                              sorted(durations.items(), key=lambda kv: -kv[1])},
        "label_fractions": {k: round(v / max(result.duration_s, 1e-9), 3)
                            for k, v in sorted(durations.items(),
                                               key=lambda kv: -kv[1])},
        "n_segments": len(result.segments),

        "hop_s": engine.cfg.hop_s,
        "n_hops": len(result.timeline),
        "latency_ms": {
            "median": round(sorted(lat)[len(lat) // 2], 2),
            "max": round(max(lat), 2),
            "over_budget": len(over),
            "pct_over_budget": round(100.0 * len(over) / max(len(lat), 1), 2),
        },
        "realtime_ok": not over,
        "asr_backend": engine.asr.backend.name,
    }


def write_spec_outputs(result, engine, out_dir: str, source=None) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, result.call_id)
    paths = {}

    paths["timeline"] = f"{stem}.timeline.jsonl"
    with open(paths["timeline"], "w") as fh:
        for row in segment_rows(result.segments):
            fh.write(json.dumps(row) + "\n")

    paths["events"] = f"{stem}.events.jsonl"
    with open(paths["events"], "w") as fh:
        for row in event_rows(result.events):
            fh.write(json.dumps(row) + "\n")

    paths["summary"] = f"{stem}.summary.json"
    with open(paths["summary"], "w") as fh:
        json.dump(summary_row(result, engine, source), fh, indent=2)

    # Per-hop detail is kept alongside: the segment view is what the spec
    # asks for, but it cannot answer "what did it think at 91.5 seconds",
    # which is the first question anyone debugging a wrong segment asks.
    paths["hops"] = f"{stem}.hops.jsonl"
    with open(paths["hops"], "w") as fh:
        for r in result.timeline:
            fh.write(json.dumps(r.to_json()) + "\n")

    return paths


def format_timeline(segments: List[Segment]) -> str:
    lines = [f"  {'start':>8} {'end':>8} {'label':<7} {'sub_label':<22} "
             f"{'human':<8} {'conf':>5}  evidence"]
    for r in segment_rows(segments):
        lines.append(
            f"  {r['start']:8.2f} {r['end']:8.2f} {r['label']:<7} "
            f"{(r['sub_label'] or '-'):<22} {(r['human_id'] or '-'):<8} "
            f"{r['confidence']:5.2f}  {r['evidence'][:58]}")
    return "\n".join(lines)
