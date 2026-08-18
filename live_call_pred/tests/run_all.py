#!/usr/bin/env python3
"""
Run the whole suite with stdlib unittest — no pytest, no network, no model
downloads. `python3 tests/run_all.py` from the package root.

`--fast` skips the end-to-end module, which runs the real streaming loop over
generated audio and takes ~30 s; everything else finishes in a few seconds.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MODULES = [
    "tests.test_codecs",
    "tests.test_causality",
    "tests.test_features",
    "tests.test_speaker_and_lexicon",
    "tests.test_fusion",
    "tests.test_events",
    "tests.test_e2e",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fast", action="store_true", help="skip the slow end-to-end module")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    modules = [m for m in MODULES if not (args.fast and m.endswith("test_e2e"))]

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    total = 0
    for m in modules:
        s = loader.loadTestsFromName(m)
        total += s.countTestCases()
        suite.addTest(s)

    print(f"running {total} tests across {len(modules)} modules\n")
    t0 = time.time()
    result = unittest.TextTestRunner(verbosity=2 if args.verbose else 1).run(suite)
    elapsed = time.time() - t0

    print(f"\n{'=' * 60}")
    print(f"{result.testsRun} tests in {elapsed:.1f}s  "
          f"failures={len(result.failures)}  errors={len(result.errors)}  "
          f"skipped={len(result.skipped)}")
    if result.skipped:
        for case, reason in result.skipped:
            print(f"  SKIPPED {case}: {reason}")
    print("PASS" if result.wasSuccessful() else "FAIL")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
