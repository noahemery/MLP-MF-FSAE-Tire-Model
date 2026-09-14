"""Acceptance gate for the fit_pipeline refactor.

Compares the vectors fit_pipeline produced against the reference captured from
magic.py before it was touched. The bar is BITWISE equality, not allclose:
these fits are deterministic -- no seeds, no shuffling, no parallelism inside a
fit -- so any difference at all means a literal drifted.

  python verify_acceptance.py

Exit 0 means the refactor reproduces magic.py exactly and the baseline run may
proceed. Exit 1 means it does not, and the run must not proceed.

*** THIS GATE'S REFERENCE IS STALE AS OF 2026-09-09. READ THIS. ***

The reference in outputs/reference/ was captured from the ORIGINAL magic.py, in
August, before:

  - William's September magic.py, which appends tire diameter to the pure-slip
    vectors. The reference lateral vector is 23 long; the current one is 24.
    That is a structural mismatch, not drift.
  - the 14 authorised defect fixes, several of which change fitted values
    (first_pass_GY reading IA instead of SA, the live G-correction penalty,
    the tm_lat epsilon guard).
  - the SL == 0 case-selection filter, which changes every straight-run fit.

So the two targets below CANNOT match, and their failure says nothing about
whether the refactor is faithful. Recapturing the reference needs the two live
blocks in magic.py un-commented (see comment_out_live_blocks.py --revert) and
about 25 minutes.

Until then the live guard is audit_literals.py, which parses magic.py's own
source text and checks every literal in all 18 blocks against the config. It
does not depend on a stored artifact, so it cannot go stale, and it passes.
"""

import os
import sys

import numpy as np

REFERENCE_DIR = os.path.join("outputs", "reference")
PIPELINE_DIR = os.path.join("outputs", "specs")

# The two blocks that were live in magic.py as committed, and so are the only
# two with a reference. They are a dependent pair: the G_y block consumes the
# lateral vector, so matching G_y also proves the lateral vector fed into it
# was bit-identical.
TARGETS = ("lat_180X60_R20_70", "GY_180X60_R20_70")

try:
    from fit_pipeline import PARAM_NAMES, ALL_SPECS
except Exception:                                    # pragma: no cover
    PARAM_NAMES, ALL_SPECS = {}, {}


def names_for(spec_id):
    spec = ALL_SPECS.get(spec_id)
    if spec is None:
        return None
    return PARAM_NAMES.get(spec.family)


def compare(spec_id):
    ref_path = os.path.join(REFERENCE_DIR, spec_id + ".npy")
    got_path = os.path.join(PIPELINE_DIR, spec_id + ".npy")

    if not os.path.exists(ref_path):
        print("  MISSING reference: " + ref_path)
        return False
    if not os.path.exists(got_path):
        print("  MISSING pipeline output: " + got_path)
        return False

    ref = np.load(ref_path).ravel()
    got = np.load(got_path).ravel()

    if ref.shape != got.shape:
        print("  SHAPE MISMATCH: reference " + str(ref.shape) +
              " vs pipeline " + str(got.shape))
        return False

    identical = np.array_equal(ref, got)
    names = names_for(spec_id) or [str(i) for i in range(ref.size)]

    # Print the full elementwise diff either way -- a pass should be readable
    # as evidence, not just an assertion.
    print("  %-8s %24s %24s %14s" % ("param", "reference", "pipeline", "delta"))
    for name, a, b in zip(names, ref, got):
        delta = b - a
        flag = "" if a == b else "   <-- DIFFERS"
        print("  %-8s %24r %24r %14.6e%s" % (name, a, b, delta, flag))

    if identical:
        print("\n  BITWISE IDENTICAL (" + str(ref.size) + " values)")
    else:
        n_diff = int(np.sum(ref != got))
        print("\n  NOT IDENTICAL: " + str(n_diff) + " of " + str(ref.size) +
              " values differ; max |delta| = " +
              repr(float(np.max(np.abs(got - ref)))))
    return identical


def main():
    print("=" * 78)
    print("ACCEPTANCE GATE: fit_pipeline vs unmodified magic.py")
    print("=" * 78)

    results = {}
    for spec_id in TARGETS:
        print("\n" + spec_id)
        print("-" * 78)
        results[spec_id] = compare(spec_id)

    print("\n" + "=" * 78)
    passed = all(results.values())
    for spec_id, ok in results.items():
        print("  %-22s %s" % (spec_id, "PASS" if ok else "FAIL"))
    if passed:
        print("\nGATE PASSED -- the refactor reproduces magic.py bitwise.")
    else:
        print("\nGATE FAILED.")
        print("\nBefore reading this as drift: the reference is STALE as of")
        print("2026-09-09 and cannot match the current magic.py. See this")
        print("file's docstring. A 23-vs-24 shape mismatch on the lateral")
        print("vector is the signature of exactly that, and is expected.")
        print("The live guard is audit_literals.py, which needs no stored")
        print("artifact and checks all 18 blocks against magic.py's source.")
    print("=" * 78)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
