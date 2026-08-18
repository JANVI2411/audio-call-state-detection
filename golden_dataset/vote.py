"""
Segments from different labelers never share boundaries, so "agreement"
is undefined until you put everyone on the same grid. This resamples
each labeler's segments onto fixed-width frames, majority-votes per
frame, and collapses the result back into segments for the gold JSONL.
"""
import math
from collections import Counter

from schema import Segment, LabelerResult, FRAME_MS


def _segments_to_frames(segments: list[Segment], n_frames: int,
                         frame_s: float) -> list:
    """
    Resample one labeler's segments onto the frame grid.

    Frames the labeler said nothing about stay `None`, meaning "abstained",
    NOT "unknown". Those are different claims and conflating them was a real
    bug: filling gaps with `unknown` let a labeler that returned nothing at
    all cast a full-strength vote for `unknown` across the whole call. On a
    5-minute chunk where two of three labelers stopped early, that produced
    gold that was 70% `unknown` -- a coverage hole recorded as a finding.
    """
    frames = [None] * n_frames
    for seg in segments:
        start_f = max(0, math.floor(seg.start / frame_s))
        end_f = min(n_frames, math.ceil(seg.end / frame_s))
        for i in range(start_f, end_f):
            frames[i] = seg.label
    return frames


def vote_frames(labeler_results: list[LabelerResult],
                 call_duration_s: float) -> dict:
    """
    Returns per-frame majority label, agreement, how many labelers actually
    voted, and the raw per-labeler frame arrays (kept for provenance and the
    review UI overlay).

    Agreement is the fraction of *voting* labelers that agreed, so it is no
    longer diluted by labelers that had nothing to say. `n_voters` is reported
    alongside it because the two together are what make a frame trustworthy:
    3 of 3 agreeing is strong, 1 of 1 agreeing is one model's opinion, and
    both score 1.0 on agreement alone.
    """
    frame_s = FRAME_MS / 1000
    n_frames = math.ceil(call_duration_s / frame_s)

    per_labeler_frames = {
        r.labeler: _segments_to_frames(r.segments, n_frames, frame_s)
        for r in labeler_results
    }

    majority, agreement, n_voters = [], [], []
    for i in range(n_frames):
        votes = Counter(frames[i] for frames in per_labeler_frames.values()
                        if frames[i] is not None)
        total = sum(votes.values())
        if total == 0:
            # Nobody covered this frame. That is genuinely unknown, and it is
            # the only case that should produce `unknown` from absence.
            majority.append("unknown")
            agreement.append(0.0)
            n_voters.append(0)
            continue
        label, count = votes.most_common(1)[0]
        majority.append(label)
        agreement.append(count / total)
        n_voters.append(total)

    return {
        "frame_s": frame_s,
        "majority": majority,
        "agreement": agreement,       # fraction of *voting* labelers agreeing
        "n_voters": n_voters,         # how many labelers covered this frame
        "per_labeler": per_labeler_frames,
    }


def frames_to_segments(majority: list[str], agreement: list[float],
                        frame_s: float, n_voters: list[int] | None = None) -> list[dict]:
    """
    Collapse consecutive same-label frames into segments for output.

    `n_voters` is optional so older callers keep working, but pass it: a
    segment's minimum voter count is what tells a reviewer whether they are
    looking at a consensus or a single model talking to itself.
    """
    if not majority:
        return []
    if n_voters is None:
        n_voters = [0] * len(majority)

    out = []
    cur_label = majority[0]
    cur_start = 0
    agree_sum = agreement[0]
    voters_min = n_voters[0]

    def flush(end_i):
        n = end_i - cur_start
        out.append({
            "start": round(cur_start * frame_s, 2),
            "end": round(end_i * frame_s, 2),
            "label": cur_label,
            "agreement": round(agree_sum / n, 2),
            "min_voters": int(voters_min),
        })

    for i in range(1, len(majority)):
        if majority[i] != cur_label:
            flush(i)
            cur_label, cur_start, agree_sum = majority[i], i, 0.0
            voters_min = n_voters[i]
        agree_sum += agreement[i]
        voters_min = min(voters_min, n_voters[i])
    flush(len(majority))
    return out


# Order used only to break an exact tie, most trustworthy first.
#
# Measured, not assumed. On this corpus asr_llm covered 98-100% of every chunk
# and its timestamps come from real speech-recognition word timings, so they
# are anchored to the audio. gpt_audio covered only 42-61%. gemini covered well
# but produced two separate timestamp faults (one chunk answered in minutes,
# another running 200s past the end of the audio).
#
# Without this, ties were resolved by `Counter.most_common`, which returns
# whichever label was inserted first -- i.e. whichever labeler happened to be
# first in the list. That is not a decision, it is an accident, and it silently
# handed every tie to the least reliable labeler.
TIEBREAK_ORDER = ["asr_llm", "gpt_audio", "gemini"]


def _seg_at(segments: list[Segment], t: float):
    """A labeler's whole segment at an instant, or None if it said nothing."""
    for seg in segments:
        if seg.start <= t < seg.end:
            return seg
    return None


def _label_at(segments: list[Segment], t: float):
    """A labeler's label at an instant, or None if it said nothing there."""
    seg = _seg_at(segments, t)
    return seg.label if seg else None


def _resolve(counts, votes) -> tuple[str, bool]:
    """
    Pick the winning label. Returns (label, was_a_tie).

    A genuine tie is not resolved by picking harder -- it is resolved by
    admitting it. The label still has to be *something* so the timeline stays
    continuous, so we take the most trustworthy labeler that voted, but the
    tie flag travels with the segment and forces it to human review rather
    than sitting in the gold looking like consensus.
    """
    top = max(counts.values())
    winners = [lbl for lbl, n in counts.items() if n == top]
    if len(winners) == 1:
        return winners[0], False
    for labeler in TIEBREAK_ORDER:
        v = votes.get(labeler)
        if v in winners:
            return v, True
    return sorted(winners)[0], True


def vote_intervals(labeler_results: list[LabelerResult], call_duration_s: float,
                    min_interval_s: float = 0.05) -> list[dict]:
    """
    Vote on the *common refinement* of the labelers' own boundaries, instead
    of on a fixed grid.

    Every boundary any labeler proposed becomes a cut point. Between two
    consecutive cut points, by construction, no labeler changes its mind --
    so each interval has exactly one label per labeler and the vote is exact.

    Why this beats the 250 ms grid it replaces:

      * No quantisation. The grid version snapped each segment outward with
        floor(start) and ceil(end), which grew every segment by up to 250 ms
        at each edge. Neighbouring segments therefore overlapped, and since
        frames were written in list order the later segment silently
        overwrote the earlier one -- boundaries drifted late, and any segment
        shorter than one cell could erase its neighbour outright.

      * Boundaries survive. A switch that a labeler put at 76.8 s stays at
        76.8 s rather than moving to 76.75 s. That matters when the gold is
        used to score boundary timing, which is most of what it is for.

      * Fewer, more meaningful units. Three labelers with ~110 segments
        between them produce ~110 intervals, not 1200 cells, and each one is
        a real stretch of the call rather than an arbitrary quarter second.

    `min_interval_s` merges cut points that are nearly coincident. Two models
    placing a boundary 8 ms apart is not a disagreement worth representing;
    without this, such pairs create slivers that are pure noise and inflate
    the segment count.

    Returns a list of {start, end, label, agreement, n_voters, votes}.
    """
    bounds = {0.0, float(call_duration_s)}
    for r in labeler_results:
        for seg in r.segments:
            if 0.0 <= seg.start <= call_duration_s:
                bounds.add(float(seg.start))
            if 0.0 <= seg.end <= call_duration_s:
                bounds.add(float(seg.end))

    ordered = sorted(bounds)
    cuts = [ordered[0]]
    for b in ordered[1:]:
        if b - cuts[-1] >= min_interval_s:
            cuts.append(b)
    if cuts[-1] < call_duration_s:
        cuts[-1] = float(call_duration_s)

    out = []
    for start, end in zip(cuts, cuts[1:]):
        mid = (start + end) / 2.0
        segs = {r.labeler: _seg_at(r.segments, mid) for r in labeler_results}
        votes = {k: (s.label if s else None) for k, s in segs.items()}
        # Speaker ids, kept per labeler. They are NOT comparable across
        # labelers -- gemini's "human_1" and gpt_audio's "human_1" are not
        # claimed to be the same person -- so they are never voted on
        # directly. See `assign_speakers` for what is comparable.
        speakers = {k: (s.human_id if s and s.human_id else None)
                    for k, s in segs.items()}
        confs = {k: (round(s.confidence, 2) if s else None)
                 for k, s in segs.items()}

        cast = [v for v in votes.values() if v is not None]
        if not cast:
            out.append({"start": start, "end": end, "label": "unknown",
                        "agreement": 0.0, "n_voters": 0, "tie": False,
                        "votes": votes, "speakers": speakers,
                        "confidences": confs})
            continue
        counts = Counter(cast)
        label, tie = _resolve(counts, votes)
        out.append({"start": start, "end": end, "label": label,
                    "agreement": counts[label] / len(cast),
                    "n_voters": len(cast), "tie": tie,
                    "votes": votes, "speakers": speakers,
                    "confidences": confs})
    return out


def informative_speaker_labelers(intervals: list[dict]) -> set:
    """
    Which labelers' speaker ids actually distinguish anyone.

    A labeler that returns the same id everywhere is not identifying people,
    it is naming the channel. On this corpus `asr_llm` returned `far_end` for
    all 64 of its segments. Counting that as a vote for "same speaker" is
    worse than ignoring it: it dilutes the labelers that genuinely track turn
    taking, and on stretches only it covers it produces a confident
    "speaker did not change" backed by no evidence at all.
    """
    seen: dict[str, set] = {}
    for iv in intervals:
        for lab, sid in (iv.get("speakers") or {}).items():
            if sid:
                seen.setdefault(lab, set()).add(sid)
    return {lab for lab, ids in seen.items() if len(ids) > 1}


def assign_speakers(intervals: list[dict]) -> list[dict]:
    """
    Give every `human` interval a consensus speaker id.

    The labelers each number speakers independently, so their ids cannot be
    compared or voted on directly -- one model's "human_1" carries no promise
    of being another model's "human_1". What *is* comparable is whether a
    labeler thinks the speaker **changed** between two stretches of human
    speech, and that is the only thing voted on here.

    This matters because it is the point of the dataset. A transfer is
    detected by noticing a new person arrived, and the merge used to throw
    every speaker id away -- gemini reported thirteen alternations between two
    people on chunk 2, and none of it reached the gold.
    """
    useful = informative_speaker_labelers(intervals)

    # Map each labeler's own name for a person onto a shared name, e.g.
    # ("gemini", "human_2") -> "human_2". Tracking identity, not just change,
    # is what makes alternation work: a two-person conversation goes
    # A, B, A, B, and counting changes alone would invent a new person on
    # every turn -- which is exactly what the first version of this did,
    # producing fourteen speakers for a conversation between two people.
    mapping: dict[tuple, str] = {}
    n_speakers = 0
    prev: dict | None = None

    for iv in intervals:
        if iv["label"] != "human":
            continue
        cur = {k: v for k, v in (iv.get("speakers") or {}).items()
               if v and k in useful}

        known = [mapping[(k, v)] for k, v in cur.items() if (k, v) in mapping]
        if known:
            consensus = Counter(known).most_common(1)[0][0]
        elif cur:
            n_speakers += 1
            consensus = f"human_{n_speakers}"
        else:
            # Nobody informative named a speaker here. Carrying the previous
            # identity is the honest default: absence of a name is not
            # evidence that somebody new arrived.
            consensus = (prev or {}).get("_id") or (
                f"human_{n_speakers or 1}" if not n_speakers else f"human_{n_speakers}")
            if not n_speakers:
                n_speakers = 1

        for k, v in cur.items():
            mapping.setdefault((k, v), consensus)

        if prev is not None and prev.get("_id") != consensus:
            shared = [k for k in cur if k in prev]
            if shared:
                changed = sum(1 for k in shared if cur[k] != prev[k])
                iv["speaker_change_votes"] = f"{changed}/{len(shared)}"
        iv["human_id"] = consensus
        prev = dict(cur, _id=consensus)
    return intervals


def merge_intervals(intervals: list[dict]) -> list[dict]:
    """
    Collapse neighbouring intervals that agreed on the same label.

    Agreement is averaged by *duration*, not by interval count. Intervals
    here are deliberately unequal, so a plain mean would let a 40 ms sliver
    of disagreement count as much as 40 seconds of consensus.
    """
    if not intervals:
        return []

    # Resolve speakers on the intervals, before merging, so a change of
    # speaker can act as a segment boundary. Merging first destroys exactly
    # the information the dataset exists to capture: on chunk 2 the whole
    # 143s-290s conversation collapsed into one `human` segment, hiding
    # thirteen turns between two different people.
    intervals = assign_speakers(intervals)

    out = []
    for iv in intervals:
        same_speaker = (out and out[-1].get("human_id") == iv.get("human_id"))
        if out and out[-1]["label"] == iv["label"] and same_speaker:
            cur = out[-1]
            d_cur = cur["end"] - cur["start"]
            d_new = iv["end"] - iv["start"]
            total = d_cur + d_new
            cur["agreement"] = ((cur["agreement"] * d_cur +
                                 iv["agreement"] * d_new) / total) if total else 0.0
            cur["min_voters"] = min(cur["min_voters"], iv["n_voters"])
            cur["has_tie"] = cur["has_tie"] or iv.get("tie", False)
            cur["end"] = iv["end"]
            # Keep the speaker naming from the longest-running piece of the
            # run: a one-second sliver should not rename a forty-second turn.
            if d_new > cur.get("_widest", 0.0):
                cur["_widest"] = d_new
                cur["speakers"] = iv.get("speakers", {})
        else:
            out.append({"start": iv["start"], "end": iv["end"],
                        "label": iv["label"], "agreement": iv["agreement"],
                        "min_voters": iv["n_voters"],
                        "has_tie": iv.get("tie", False),
                        "human_id": iv.get("human_id"),
                        "speaker_change_votes": iv.get("speaker_change_votes"),
                        "speakers": iv.get("speakers", {}),
                        "_widest": iv["end"] - iv["start"]})

    for s in out:
        s.pop("_widest", None)
        s["start"] = round(s["start"], 2)
        s["end"] = round(s["end"], 2)
        s["agreement"] = round(s["agreement"], 3)
        # Per-labeler speaker names are kept for audit but are not the answer;
        # `human_id` is. Drop the empties so the gold stays readable.
        s["speakers"] = {k: v for k, v in (s.get("speakers") or {}).items() if v}
    return out


def needs_human_review(segment: dict, n_labelers: int,
                        spot_check_rate: float = 0.12) -> tuple[bool, str]:
    """
    Routing rule. Returns (needs_review, reason).
    - full agreement -> spot-check sample only
    - 2/n agreement -> higher spot-check rate
    - <2/n (a true split) -> mandatory review
    """
    import random
    # An exact tie was resolved by labeler priority, not by consensus. It is
    # a coin toss wearing a label, so it always goes to a person.
    if segment.get("has_tie"):
        return (True, "mandatory_tie")
    agreement = segment["agreement"]
    if agreement >= 0.999:
        return (random.random() < spot_check_rate, "spot_check_full_agreement")
    if agreement >= (2.0 / n_labelers) - 1e-6:
        return (random.random() < spot_check_rate * 2.5, "spot_check_majority")
    return (True, "mandatory_split")
