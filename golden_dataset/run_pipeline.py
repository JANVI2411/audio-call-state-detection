"""
End-to-end orchestrator for one call. Run per-call over your corpus.

python run_pipeline.py --audio calls/CALL123.beeped.wav --call-id CALL123
"""
import argparse
from dataclasses import asdict
import json
import os
import tempfile
import time
import traceback

import soundfile as sf

from vote import (vote_frames, frames_to_segments, vote_intervals,
                  merge_intervals, needs_human_review)
from repair import repair_segments, describe

PRICES_PER_MILLION = {
    "gemini": {
        "input": float(os.getenv("GEMINI_INPUT_PRICE_PER_MILLION", "1.50")),
        "output": float(os.getenv("GEMINI_OUTPUT_PRICE_PER_MILLION", "7.50")),
    },
    "gpt_audio": {
        "text_input": float(os.getenv("OPENAI_AUDIO_TEXT_INPUT_PRICE_PER_MILLION", "2.50")),
        "text_output": float(os.getenv("OPENAI_AUDIO_TEXT_OUTPUT_PRICE_PER_MILLION", "10.00")),
        "audio_input": float(os.getenv("OPENAI_AUDIO_INPUT_PRICE_PER_MILLION", "32.00")),
        "audio_output": float(os.getenv("OPENAI_AUDIO_OUTPUT_PRICE_PER_MILLION", "64.00")),
    },
    "asr_llm": {
        "input": float(os.getenv("OPENAI_CLASSIFY_INPUT_PRICE_PER_MILLION", "0.25")),
        "output": float(os.getenv("OPENAI_CLASSIFY_OUTPUT_PRICE_PER_MILLION", "2.00")),
    },
}


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_env(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(__file__), ".env")

    if not os.path.exists(path):
        return

    with open(path) as env_f:
        for raw_line in env_f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"'")
            os.environ.setdefault(key, value)


def get_duration(path):
    info = sf.info(path)
    return info.frames / info.samplerate


def trim_audio(path, limit_seconds):
    if limit_seconds is None:
        return path, None

    if limit_seconds <= 0:
        raise ValueError("--limit-seconds must be greater than 0")

    info = sf.info(path)
    max_frames = int(limit_seconds * info.samplerate)
    data, sr = sf.read(path, frames=max_frames)

    fd, tmp_path = tempfile.mkstemp(suffix=".wav", prefix="audio_agent_")
    os.close(fd)
    sf.write(tmp_path, data, sr)
    return tmp_path, min(info.frames / info.samplerate, limit_seconds)


def save_labeler_output(out_dir, call_id, labeler, result=None, error=None):
    model_dir = os.path.join(out_dir, "model_outputs")
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, f"{call_id}.{labeler}.json")

    if result is None:
        payload = {
            "labeler": labeler,
            "call_id": call_id,
            "status": "error",
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
    else:
        payload = asdict(result)
        payload["status"] = "ok"
        payload["segment_count"] = len(result.segments)

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def load_labeler_output(out_dir, call_id, labeler):
    from schema import LabelerResult, Segment

    path = os.path.join(out_dir, "model_outputs", f"{call_id}.{labeler}.json")
    if not os.path.exists(path):
        return None

    with open(path) as f:
        payload = json.load(f)
    if payload.get("status") != "ok":
        log(f"{labeler}: cached output exists but status={payload.get('status')}; rerunning")
        return None

    segments = [Segment(**seg) for seg in payload.get("segments", [])]
    result = LabelerResult(
        labeler=payload["labeler"],
        call_id=payload["call_id"],
        segments=segments,
        raw_response=payload.get("raw_response", ""),
        model=payload.get("model", ""),
        usage=payload.get("usage", {}),
    )
    log(f"{labeler}: using cached output from {path}; no new API cost")
    return result


def _usage_value(usage, *keys):
    current = usage
    for key in keys:
        if not current:
            return 0
        current = current.get(key, 0)
    return current or 0


def estimate_cost(labeler, usage):
    if not usage:
        return None

    if labeler == "gemini":
        input_tokens = (
            _usage_value(usage, "input_token_count")
            or _usage_value(usage, "prompt_token_count")
            or _usage_value(usage, "inputTokens")
            or _usage_value(usage, "promptTokenCount")
        )
        output_tokens = (
            _usage_value(usage, "output_token_count")
            or _usage_value(usage, "candidates_token_count")
            or _usage_value(usage, "outputTokens")
            or _usage_value(usage, "candidatesTokenCount")
        )
        prices = PRICES_PER_MILLION["gemini"]
        return ((input_tokens * prices["input"]) +
                (output_tokens * prices["output"])) / 1_000_000

    if labeler == "gpt_audio":
        prompt_tokens = _usage_value(usage, "prompt_tokens")
        completion_tokens = _usage_value(usage, "completion_tokens")
        prompt_audio = _usage_value(usage, "prompt_tokens_details", "audio_tokens")
        completion_audio = _usage_value(usage, "completion_tokens_details", "audio_tokens")
        prompt_text = max(0, prompt_tokens - prompt_audio)
        completion_text = max(0, completion_tokens - completion_audio)
        prices = PRICES_PER_MILLION["gpt_audio"]
        return (
            prompt_text * prices["text_input"] +
            completion_text * prices["text_output"] +
            prompt_audio * prices["audio_input"] +
            completion_audio * prices["audio_output"]
        ) / 1_000_000

    if labeler == "asr_llm":
        prompt_tokens = _usage_value(usage, "prompt_tokens")
        completion_tokens = _usage_value(usage, "completion_tokens")
        prices = PRICES_PER_MILLION["asr_llm"]
        return ((prompt_tokens * prices["input"]) +
                (completion_tokens * prices["output"])) / 1_000_000

    return None


def token_totals(usage):
    """Pull input/output token counts out of whichever shape the SDK returned."""
    if not usage:
        return None, None
    inp = (_usage_value(usage, "prompt_tokens")
           or _usage_value(usage, "input_token_count")
           or _usage_value(usage, "prompt_token_count")
           or _usage_value(usage, "inputTokens")
           or _usage_value(usage, "promptTokenCount"))
    out = (_usage_value(usage, "completion_tokens")
           or _usage_value(usage, "output_token_count")
           or _usage_value(usage, "candidates_token_count")
           or _usage_value(usage, "outputTokens")
           or _usage_value(usage, "candidatesTokenCount"))
    return (inp or None), (out or None)


def log_cost(result):
    cost = estimate_cost(result.labeler, result.usage)
    if cost is None:
        log(f"{result.labeler}: cost estimate unavailable; no usage returned")
        return
    log(f"{result.labeler}: estimated API cost ${cost:.6f}")


def run_labeler(labeler, label_fn, audio_path, call_id, out_dir, force=False,
                metrics=None):
    """
    `metrics` is a list this appends one record to per labeler: how long the
    call took, what it cost, and whether it was billed at all. Cost used to be
    printed and then thrown away, which made it impossible to answer "what did
    this corpus cost to label" after the fact. Cached results are recorded too,
    with cost 0 and cached=True, so the totals stay honest on a re-run.
    """
    def record(**kw):
        if metrics is not None:
            metrics.append({"labeler": labeler, **kw})

    if not force:
        cached = load_labeler_output(out_dir, call_id, labeler)
        if cached is not None:
            inp, out = token_totals(cached.usage)
            record(status="cached", latency_s=0.0, cost_usd=0.0, cached=True,
                   input_tokens=inp, output_tokens=out,
                   n_segments=len(cached.segments), model=cached.model or None)
            return cached

    start = time.perf_counter()
    log(f"{labeler}: started")
    try:
        result = label_fn(audio_path, call_id)
    except Exception as exc:
        elapsed = time.perf_counter() - start
        path = save_labeler_output(out_dir, call_id, labeler, error=exc)
        record(status="error", latency_s=round(elapsed, 2), cost_usd=None,
               cached=False, error=f"{type(exc).__name__}: {exc}")
        log(f"{labeler}: failed after {elapsed:.1f}s; saved error to {path}")
        raise

    elapsed = time.perf_counter() - start
    # Save under the same name the cache lookup above reads. Saving under
    # `result.labeler` instead would silently never hit the cache if a labeler
    # ever reported a different name than it was registered with, and a cache
    # miss here means paying for the same API call again.
    path = save_labeler_output(out_dir, call_id, labeler, result=result)
    cost = estimate_cost(result.labeler, result.usage)
    inp, out = token_totals(result.usage)
    record(status="ok", latency_s=round(elapsed, 2),
           cost_usd=(round(cost, 6) if cost is not None else None),
           cached=False, input_tokens=inp, output_tokens=out,
           n_segments=len(result.segments), model=result.model or None)
    log(f"{labeler}: done in {elapsed:.1f}s; "
        f"{len(result.segments)} segments saved to {path}")
    log_cost(result)
    return result


def default_call_id(audio_path):
    """
    Name outputs after the audio file they came from.

    Every file this pipeline writes is prefixed with the call id, so deriving
    it from the file name means a chunk called `69f3a1e4_c00.wav` always
    produces `69f3a1e4_c00.gemini.json`, `69f3a1e4_c00.gold.jsonl` and so on.
    Typing the id by hand invites a mismatch that silently overwrites another
    chunk's results.
    """
    return os.path.basename(audio_path).split(".")[0]


def run(audio_path, call_id=None, out_dir="./gold", limit_seconds=None,
        force_labelers=False, offset_seconds=0.0, grid_vote=False):
    """
    `offset_seconds` is where this audio starts inside the original full-length
    recording. Labelers number their segments from zero, so a chunk taken from
    5:00 onward reports its first second as 0.0. Recording the offset lets the
    gold rows also carry `abs_start`/`abs_end` on the original call's clock,
    which is what any later merge across chunks needs.
    """
    total_start = time.perf_counter()
    load_env()
    import gemini_labeler
    import gpt_labeler
    import asr_llm_labeler

    call_id = call_id or default_call_id(audio_path)
    os.makedirs(out_dir, exist_ok=True)
    log(f"call_id={call_id}")
    log(f"audio={audio_path}")
    if offset_seconds:
        log(f"chunk offset={offset_seconds:.1f}s into the original recording")

    label_audio_path, limited_duration = trim_audio(audio_path, limit_seconds)
    duration = limited_duration if limited_duration is not None else get_duration(audio_path)
    if limit_seconds is None:
        log(f"processing full audio ({duration:.1f}s)")
    else:
        log(f"processing first {duration:.1f}s only")

    metrics = []
    try:
        results = []
        results.append(run_labeler("gemini", gemini_labeler.label_call,
                                   label_audio_path, call_id, out_dir,
                                   force=force_labelers, metrics=metrics))
        results.append(run_labeler("gpt_audio", gpt_labeler.label_call,
                                   label_audio_path, call_id, out_dir,
                                   force=force_labelers, metrics=metrics))
        results.append(run_labeler("asr_llm", asr_llm_labeler.label_call,
                                   label_audio_path, call_id, out_dir,
                                   force=force_labelers, metrics=metrics))
    except Exception:
        # Save whatever we measured before the failure -- the labelers that
        # did finish were still billed, and that spend should be visible.
        write_metrics(out_dir, call_id, audio_path, duration, offset_seconds,
                      metrics, time.perf_counter() - total_start)
        raise
    finally:
        if label_audio_path != audio_path:
            os.remove(label_audio_path)

    # Repair obviously-broken timestamps before they reach the vote. A model
    # that answered in minutes, or ran past the end of the audio, would
    # otherwise cast its votes at the wrong times and look like an abstention
    # everywhere else -- indistinguishable, in the gold, from real ambiguity.
    repairs = []
    for r in results:
        fixed, report = repair_segments(r.segments, duration, r.labeler)
        note = describe(report)
        if note:
            log(f"{r.labeler}: REPAIRED -- {note}")
        else:
            log(f"{r.labeler}: covers {report['coverage_after_s']:.0f}s of "
                f"{duration:.0f}s ({100*report['coverage_fraction']:.0f}%)")
        r.segments = fixed
        repairs.append(report)

    # Vote on the union of the labelers' own boundaries rather than a fixed
    # grid. Measured on this corpus the grid put every transition a mean 56 ms
    # late (max 200 ms) and reproduced only 9 of 17 boundaries exactly, because
    # snapping each segment outward to cell edges makes neighbours overlap and
    # the later one wins. The union reproduces all 17 exactly. Same labels,
    # honest timings. `--grid-vote` restores the old behaviour for comparison.
    if grid_vote:
        log("voting model outputs (250ms grid)")
        voted = vote_frames(results, duration)
        segments = frames_to_segments(voted["majority"], voted["agreement"],
                                      voted["frame_s"], voted.get("n_voters"))
    else:
        log("voting model outputs (union of labeler boundaries)")
        intervals = vote_intervals(results, duration)
        segments = merge_intervals(intervals)
        log(f"{len(intervals)} intervals -> {len(segments)} segments")

    gold_path = os.path.join(out_dir, f"{call_id}.gold.jsonl")
    # One queue per call id, not one shared file. The shared file was opened
    # in append mode, so re-running a chunk silently stacked a second copy of
    # its segments on top of the first and the review app would show every
    # clip twice. Per-call files are rewritten cleanly on each run.
    queue_path = os.path.join(out_dir, f"review_queue.{call_id}.jsonl")

    review_count = 0
    with open(gold_path, "w") as gold_f, open(queue_path, "w") as queue_f:
        for seg in segments:
            seg["call_id"] = call_id
            seg["source_audio"] = os.path.basename(audio_path)
            seg["chunk_offset_s"] = round(offset_seconds, 3)
            seg["abs_start"] = round(seg["start"] + offset_seconds, 2)
            seg["abs_end"] = round(seg["end"] + offset_seconds, 2)
            flag, reason = needs_human_review(seg, n_labelers=len(results))
            seg["reason"] = reason
            seg["model_labels"] = {
                r.labeler: _label_at(r, (seg["start"] + seg["end"]) / 2)
                for r in results
            }
            gold_f.write(json.dumps(seg) + "\n")
            if flag:
                review_count += 1
                queue_f.write(json.dumps(seg) + "\n")

    wall_s = time.perf_counter() - total_start
    m = write_metrics(out_dir, call_id, audio_path, duration, offset_seconds,
                      metrics, wall_s, repairs)

    log(f"gold saved to {gold_path}")
    log(f"review queue saved to {queue_path}")
    log(f"{call_id}: {len(segments)} segments, {review_count} flagged for review")
    log(f"metrics saved to {m['path']}")
    log(f"cost: {m['cost_display']}   wall time: {wall_s:.1f}s "
        f"({m['realtime_factor_display']})")
    log(f"finished in {wall_s:.1f}s")
    return {"call_id": call_id, "gold_path": gold_path,
            "queue_path": queue_path, "n_segments": len(segments),
            "n_review": review_count, "duration_s": duration,
            "wall_s": round(wall_s, 2),
            "cost_usd": m["total_cost_usd"],
            "cost_complete": m["cost_complete"],
            "labelers": metrics}


def write_metrics(out_dir, call_id, audio_path, duration_s, offset_s,
                  metrics, wall_s, repairs=None):
    """
    Persist what this chunk cost and how long it took.

    Cost can be genuinely unknown: an SDK that returns no usage block leaves
    us with no token counts, and inventing a number there would be worse than
    admitting the gap. So `total_cost_usd` sums only the labelers that
    actually reported usage, and `cost_complete` says whether that sum covers
    all of them. Read a total with cost_complete=false as a floor, not a bill.
    """
    metrics_dir = os.path.join(out_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    path = os.path.join(metrics_dir, f"{call_id}.metrics.json")

    known = [r for r in metrics if r.get("cost_usd") is not None]
    unknown = [r for r in metrics if r.get("cost_usd") is None
               and r.get("status") == "ok"]
    total_cost = round(sum(r["cost_usd"] for r in known), 6) if known else 0.0
    cost_complete = not unknown

    if cost_complete:
        cost_display = f"${total_cost:.4f}"
    elif known:
        cost_display = (f"${total_cost:.4f} + unknown "
                        f"({', '.join(r['labeler'] for r in unknown)} "
                        f"reported no usage)")
    else:
        cost_display = ("unknown (no labeler reported usage)")

    # Real-time factor: wall seconds spent per second of audio. Below 1.0
    # means the pipeline keeps up with the call; above means it does not.
    rtf = (wall_s / duration_s) if duration_s else None
    rtf_display = (f"{rtf:.2f}x real time" if rtf else "n/a")

    payload = {
        "call_id": call_id,
        "source_audio": os.path.basename(audio_path),
        "chunk_offset_s": round(offset_s, 3),
        "audio_duration_s": round(duration_s, 2),
        "wall_time_s": round(wall_s, 2),
        "realtime_factor": (round(rtf, 3) if rtf else None),
        "total_cost_usd": total_cost,
        "cost_complete": cost_complete,
        "cost_per_audio_minute_usd": (
            round(total_cost / (duration_s / 60.0), 6)
            if duration_s and total_cost else 0.0),
        "labelers": metrics,
        "timestamp_repairs": repairs or [],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    payload["path"] = path
    payload["cost_display"] = cost_display
    payload["realtime_factor_display"] = rtf_display
    return payload


def _label_at(result, t):
    for seg in result.segments:
        if seg.start <= t < seg.end:
            return seg.label
    return "unknown"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--call-id",
                    help="Defaults to the audio file name, so outputs are "
                         "always named after the file they came from.")
    ap.add_argument("--out-dir", default="./gold")
    ap.add_argument("--limit-seconds", type=float,
                    help="Only process the first N seconds of audio.")
    ap.add_argument("--offset-seconds", type=float, default=0.0,
                    help="Where this audio starts in the original recording, "
                         "used to write whole-call timestamps into the gold.")
    ap.add_argument("--force-labelers", action="store_true",
                    help="Rerun labelers even when cached model outputs exist.")
    ap.add_argument("--grid-vote", action="store_true",
                    help="Use the old fixed 250ms voting grid instead of the "
                         "union of labeler boundaries.")
    args = ap.parse_args()
    run(args.audio, args.call_id, args.out_dir, args.limit_seconds,
        args.force_labelers, args.offset_seconds, args.grid_vote)
