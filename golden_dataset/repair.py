"""
Repair labeler timestamps before voting.

Audio models are asked for times in seconds and mostly comply, but two
failure modes showed up on real 5-minute chunks and both silently poison the
vote rather than announcing themselves:

1. Wrong unit. On one chunk a model described all 300 seconds of audio but
   numbered it 0.00 to 4.59 -- minutes, not seconds. Its segment ordering and
   descriptions were correct; only the clock was wrong. Left alone, every one
   of its votes landed inside the first five seconds and the remaining 295
   seconds looked like an abstention.

2. Over-run. On another chunk a model's final segment ended at 500.0s on a
   300s file. Times past the end of the audio cannot be checked against
   anything, so they are dropped rather than trusted.

Both repairs are conservative and reported. `detect_unit_scale` only fires
when rescaling actually makes the numbers fit the audio, so a model that
genuinely labelled the first few seconds and stopped is left alone.
"""
from __future__ import annotations

from dataclasses import replace
from typing import List, Tuple

from schema import Segment

# A model that used minutes will have a span roughly duration/60. Require the
# observed span to be under this fraction of the audio before considering a
# rescale, so a short-but-honest labeling is never touched.
MAX_PLAUSIBLE_SHORT_FRACTION = 0.25
# After rescaling, the span must still fit inside the audio (plus slack for a
# model that rounds the final boundary up past the true end).
POST_SCALE_TOLERANCE = 1.15


def detect_unit_scale(segments: List[Segment], duration_s: float) -> float:
    """
    Return the factor to multiply timestamps by: 60.0 if the labeler appears
    to have answered in minutes, otherwise 1.0.
    """
    if len(segments) < 3 or duration_s <= 0:
        return 1.0
    span = max(s.end for s in segments)
    if span <= 0:
        return 1.0
    if span >= duration_s * MAX_PLAUSIBLE_SHORT_FRACTION:
        return 1.0                      # span is plausible as seconds
    if span * 60.0 > duration_s * POST_SCALE_TOLERANCE:
        return 1.0                      # rescaling would overshoot the audio
    return 60.0


def clip_to_duration(segments: List[Segment], duration_s: float
                     ) -> Tuple[List[Segment], int, int]:
    """Drop segments starting past the end; truncate ones that run over."""
    kept, dropped, truncated = [], 0, 0
    for s in segments:
        if s.start >= duration_s:
            dropped += 1
            continue
        if s.end > duration_s:
            s = replace(s, end=duration_s)
            truncated += 1
        if s.end > s.start:
            kept.append(s)
    return kept, dropped, truncated


def repair_segments(segments: List[Segment], duration_s: float,
                    labeler: str = "") -> Tuple[List[Segment], dict]:
    """
    Apply both repairs. Returns (segments, report) where report records
    exactly what was changed so it can be written into the metrics and
    reviewed later -- a silent correction is worse than the original bug.
    """
    report = {"labeler": labeler, "unit_scale": 1.0, "dropped": 0,
              "truncated": 0, "span_before_s": 0.0, "span_after_s": 0.0,
              "coverage_before_s": 0.0, "coverage_after_s": 0.0}
    if not segments:
        return segments, report

    report["span_before_s"] = round(max(s.end for s in segments), 2)
    report["coverage_before_s"] = round(sum(s.end - s.start for s in segments), 2)

    scale = detect_unit_scale(segments, duration_s)
    if scale != 1.0:
        segments = [replace(s, start=s.start * scale, end=s.end * scale)
                    for s in segments]
        report["unit_scale"] = scale

    segments, dropped, truncated = clip_to_duration(segments, duration_s)
    report["dropped"] = dropped
    report["truncated"] = truncated
    report["span_after_s"] = round(max((s.end for s in segments), default=0.0), 2)
    report["coverage_after_s"] = round(sum(s.end - s.start for s in segments), 2)
    report["coverage_fraction"] = (
        round(report["coverage_after_s"] / duration_s, 3) if duration_s else 0.0)
    return segments, report


def describe(report: dict) -> str:
    """One-line human summary, or empty string if nothing was changed."""
    bits = []
    if report["unit_scale"] != 1.0:
        bits.append(f"rescaled x{report['unit_scale']:.0f} "
                    f"(looked like minutes: span {report['span_before_s']}s "
                    f"-> {report['span_after_s']}s)")
    if report["dropped"]:
        bits.append(f"dropped {report['dropped']} segment(s) past end of audio")
    if report["truncated"]:
        bits.append(f"truncated {report['truncated']} segment(s) to audio end")
    return "; ".join(bits)
