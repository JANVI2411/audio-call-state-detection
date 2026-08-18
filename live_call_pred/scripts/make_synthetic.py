#!/usr/bin/env python3
"""Generate the labelled synthetic call corpus under data/."""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zlib import crc32  # noqa: E402

from callstate.simulate import make_call, write_call  # noqa: E402

SCENARIOS = ["ivr_only", "simple", "transfer", "failed_transfer"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--per-scenario", type=int, default=2,
                    help="calls per scenario (different seeds)")
    ap.add_argument("--scenarios", nargs="*", default=SCENARIOS)
    args = ap.parse_args()

    total = 0
    for scen in args.scenarios:
        for k in range(args.per_scenario):
            # crc32, not hash(): builtin hash() of a str is salted per process, so
            # the "reproducible corpus" would silently differ between runs.
            seed = 1000 + 977 * k + (crc32(scen.encode()) % 5000)
            call = make_call(scenario=scen, seed=seed)
            name = f"{scen}_{k}"
            paths = write_call(call, args.out, name)
            print(f"{name:<20} {call.duration_s:6.1f}s  turns={len(call.turns):<3} "
                  f"gold_events={len(call.gold_events)}  -> {paths['wav']}")
            total += 1
    print(f"\nwrote {total} synthetic calls to {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
