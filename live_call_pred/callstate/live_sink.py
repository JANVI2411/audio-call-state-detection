"""
Print and persist each decision the moment it is made.

The batch writer holds everything in memory and writes once the call ends.
That is fine for replaying a recording and wrong for a live call, where the
two things you most need are to see what is happening now, and to still have
the record if the process dies. A crash under the batch writer loses the
whole call.

So this does two jobs per decision, both immediately:

  * prints one line, only when something changed worth showing
  * appends to `<call>.hops.jsonl` and flushes, so the file on disk is
    always current to the last decision

Printing every decision is not useful -- at a 2 second hop a ten minute call
is 300 lines of mostly "still on hold". `min_print_gap_s` keeps a heartbeat
for the quiet stretches while every state change and every event prints the
moment it happens.
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Optional, TextIO

RESET = "\033[0m"
DIM = "\033[2m"
BOLD = "\033[1m"
COLOR = {
    "ivr": "\033[38;5;33m",     # blue
    "human": "\033[38;5;35m",   # green
    "hold": "\033[38;5;172m",   # amber
    "other": "\033[38;5;245m",  # grey
}


# Gold labels this pipeline cannot predict, so they cannot be scored. They are
# skipped, never counted as errors -- `survey` has no matching state here, and
# `unknown` is the absence of a label rather than a label.
UNSCORABLE = {"survey", "unknown"}


def load_gold(path: str) -> list:
    """Read golden_dataset gold, keeping only what scoring needs."""
    turns = []
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            turns.append((float(r["start"]), float(r["end"]), r["label"]))
    return turns


def gold_at(turns, t_s: float) -> Optional[str]:
    for start, end, label in turns:
        if start <= t_s < end:
            return label
    return None


class LiveSink:
    def __init__(self, out_dir: str, call_id: str, hop_s: float,
                 print_every_hop: bool = False,
                 min_print_gap_s: float = 10.0,
                 gold: Optional[list] = None,
                 color: Optional[bool] = None,
                 stream: TextIO = sys.stdout):
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(out_dir, f"{call_id}.hops.jsonl")
        self.events_path = os.path.join(out_dir, f"{call_id}.events.jsonl")
        # Truncate: a live run starts a new record rather than appending to
        # whatever the last run left behind.
        self._fh = open(self.path, "w")
        self._ev = open(self.events_path, "w")
        self.hop_s = hop_s
        self.print_every_hop = print_every_hop
        self.min_print_gap_s = min_print_gap_s
        self.stream = stream
        self.color = stream.isatty() if color is None else color
        self.started = time.perf_counter()

        self._last_state: Optional[str] = None
        self._last_printed_s = -1e9
        # Also heartbeat on wall-clock, not just audio time. When a hop is
        # slow -- and the recogniser makes them very slow -- 30 seconds of
        # audio can be over a minute of silence on screen, which is
        # indistinguishable from a hang. Something must show up regularly
        # even when the call state is not changing.
        self._last_printed_wall = time.perf_counter()
        self.max_quiet_wall_s = 8.0
        self._n_over = 0
        self._n_hops = 0

        # Live scoring against the answer key, when one is supplied.
        self.gold = gold
        self._n_scored = 0
        self._n_right = 0
        self._n_skipped = 0
        self._confusion: dict = {}

    # -- formatting --------------------------------------------------------
    def _c(self, text: str, code: str) -> str:
        return f"{code}{text}{RESET}" if self.color else text

    def _score(self, t_s: float, predicted: str):
        """Returns (gold_label, verdict) where verdict is ok / wrong / skip."""
        if self.gold is None:
            return None, None
        truth = gold_at(self.gold, t_s)
        if truth is None:
            return None, "skip"
        if truth in UNSCORABLE:
            self._n_skipped += 1
            return truth, "skip"
        self._n_scored += 1
        right = (truth == predicted)
        self._n_right += int(right)
        key = (truth, predicted)
        self._confusion[key] = self._confusion.get(key, 0) + 1
        return truth, ("ok" if right else "wrong")

    def _line(self, t_s: float, row, timing: dict, changed: bool,
              truth=None, verdict=None) -> str:
        budget = self.hop_s * 1000.0
        ms = timing["total_ms"]
        state = self._c(f"{row.state:<6}", COLOR.get(row.state, ""))
        if changed:
            state = self._c(state, BOLD)
        mark = "*" if changed else " "
        late = self._c(f"  LATE {ms:.0f}ms", "\033[38;5;196m") if ms > budget else ""
        spk = f" {row.speaker_id}" if row.speaker_id else ""
        sub = f" {row.sub_label}" if row.sub_label else ""
        text = (row.text or "").strip()
        tail = self._c(f'  "{text[-40:]}"', DIM) if text else ""

        gold_col = ""
        if self.gold is not None:
            if verdict == "ok":
                gold_col = self._c(f"  = {truth:<6} OK ", "\033[38;5;35m")
            elif verdict == "wrong":
                gold_col = self._c(f"  ! {truth:<6} <-- ", "\033[38;5;196m")
            else:
                gold_col = self._c(f"  ~ {(truth or 'none'):<6} skip", DIM)
            if self._n_scored:
                acc = 100.0 * self._n_right / self._n_scored
                gold_col += self._c(f" {acc:5.1f}%", DIM)

        return (f"{mark} {t_s:7.1f}s  {state} {row.confidence:.2f}"
                f"{gold_col}{self._c(sub, DIM)}{spk}{late}{tail}")

    # -- called once per decision -----------------------------------------
    def hop(self, t_s: float, row, timing: dict) -> None:
        self._n_hops += 1
        if timing["total_ms"] > self.hop_s * 1000.0:
            self._n_over += 1

        # Persist first, print second. If the process dies mid-line the
        # record is already safe on disk.
        self._fh.write(json.dumps(row.to_json()) + "\n")
        self._fh.flush()

        truth, verdict = self._score(t_s, row.state)

        changed = row.state != self._last_state
        now = time.perf_counter()
        due = ((t_s - self._last_printed_s) >= self.min_print_gap_s
               or (now - self._last_printed_wall) >= self.max_quiet_wall_s)
        # A disagreement with the answer key is always worth showing, even in
        # the middle of a quiet stretch -- that is the whole point of running
        # with gold attached.
        if changed or due or self.print_every_hop or verdict == "wrong":
            print(self._line(t_s, row, timing, changed, truth, verdict),
                  file=self.stream, flush=True)
            self._last_printed_s = t_s
            self._last_printed_wall = now
        self._last_state = row.state

    def event(self, ev) -> None:
        self._ev.write(json.dumps(ev.to_json()) + "\n")
        self._ev.flush()
        tag = self._c(f"[{ev.type.value}]", BOLD)
        print(f"  {' ':7}   {tag} at {ev.t_s:.1f}s  {ev.evidence}"
              + (f"  {ev.meta}" if ev.meta else ""),
              file=self.stream, flush=True)

    def close(self) -> dict:
        self._fh.close()
        self._ev.close()
        wall = time.perf_counter() - self.started
        out = {"n_hops": self._n_hops, "n_over_budget": self._n_over,
               "wall_s": round(wall, 2), "hops_path": self.path,
               "events_path": self.events_path}
        if self.gold is not None:
            out.update({
                "n_scored": self._n_scored,
                "n_skipped": self._n_skipped,
                "accuracy": (round(self._n_right / self._n_scored, 4)
                             if self._n_scored else 0.0),
                "confusion": {f"{t}->{p}": n
                              for (t, p), n in sorted(self._confusion.items(),
                                                      key=lambda kv: -kv[1])},
            })
        return out
