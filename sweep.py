"""Overnight driver: train every family across folds, seeds and configs.

One command, unattended-safe:

    python sweep.py                 # everything
    python sweep.py --quick         # small grid, ~10 min, use to smoke-test
    python sweep.py --summary-only  # rebuild the summary from existing results

Properties that matter for running this while asleep:

  incremental   every run is appended to outputs/sweep/results.jsonl the moment
                it finishes, so a crash at 3am keeps everything before it
  resumable     re-running skips configs already present in that file
  isolated      one config failing is recorded and the sweep continues
  awake         calls progress.keep_hot(), the same power-throttling opt-out
                that stopped a run being starved to 1.6% CPU overnight
  self-healing  builds the segment cache, and regenerates the NLLS baseline,
                if either is missing

Why a sweep rather than one run: training on 6 tires takes seconds, so a
single number is indistinguishable from luck. Repeating across seeds and folds
is what turns it into a result with a spread attached.
"""

import argparse
import itertools
import json
import os
import time
import traceback

import numpy as np

OUT_DIR = os.path.join("outputs", "sweep")
RESULTS = os.path.join(OUT_DIR, "results.jsonl")


def shard_results(shard):
    """Each worker writes its own file; nothing is shared, so no locking."""
    if shard is None:
        return RESULTS
    return os.path.join(OUT_DIR, "results_shard%d.jsonl" % shard)


def all_result_files():
    if not os.path.isdir(OUT_DIR):
        return []
    return [os.path.join(OUT_DIR, f) for f in sorted(os.listdir(OUT_DIR))
            if f.startswith("results") and f.endswith(".jsonl")]

FAMILIES = ("lateral", "longitudinal", "gx", "gy")
PROTOCOLS = ("segment", "spec")

GRID = {
    "lr": (1e-4, 1e-3, 1e-2),
    "hidden": ((32, 32), (64, 64)),
    "normalize": (False, True),
}
SEEDS = (0, 1, 2)
N_FOLDS = 5
STEPS = 2000
MAX_ROWS = 6000

# Deep mode. Trades wall clock for convergence and tighter error bars, which
# is where extra compute actually helps here. The grid deliberately does NOT
# widen the network much: with 6 training specs a larger net memorises rather
# than generalises, so steps and seeds are the axes worth paying for.
DEEP = {
    "grid": {"lr": (1e-4, 3e-4, 1e-3, 3e-3, 1e-2),
             "hidden": ((32, 32), (64, 64)),
             "normalize": (False, True)},
    "seeds": (0, 1, 2, 3, 4), "n_folds": 5, "steps": 10000, "max_rows": 12000,
    "protocols": ("segment", "spec"),
}

QUICK = {
    "grid": {"lr": (1e-3,), "hidden": ((32, 32),), "normalize": (False,)},
    "seeds": (0,), "n_folds": 3, "steps": 300, "max_rows": 3000,
    "protocols": ("segment",),
}


def run_key(r):
    """Identity of a run, for resume.

    steps is part of the identity: a deeper re-run of the same config is a
    different run, not a duplicate, and must not be skipped.
    """
    return (r["family"], r["protocol"], r["fold"], r["seed"], r["lr"],
            tuple(r["hidden"]), bool(r["normalize"]), r.get("steps"))


def load_done():
    """Union across every shard file, so resume works however it was sharded."""
    done = set()
    for path in all_result_files():
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(run_key(json.loads(line)))
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def append(record, path):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(record, default=float) + "\n")


def ensure_prereqs(verbose=True):
    """Baseline parquet and segment cache must exist before training."""
    import dataset

    baseline = os.path.join("outputs", "baseline_params.parquet")
    if not os.path.exists(baseline):
        if verbose:
            print("NLLS baseline missing -- running it first "
                  "(~30 min). This is a one-time cost.", flush=True)
        import subprocess, sys
        rc = subprocess.call([sys.executable, "run_baseline.py"])
        if rc != 0 or not os.path.exists(baseline):
            raise SystemExit("baseline run failed; cannot continue")

    missing = [s for s in _all_spec_ids()
               if not os.path.exists(dataset.cache_path(s))]
    if missing:
        if verbose:
            print("building segment cache for %d specs (~5 min, one time)"
                  % len(missing), flush=True)
        dataset.build_all_caches(verbose=verbose)


def _all_spec_ids():
    import fit_pipeline as fp
    return list(fp.ALL_SPECS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="small grid for a ~10 min smoke test")
    ap.add_argument("--deep", action="store_true",
                    help="5x steps, 5 seeds, wider lr grid (~24-30 h on 10 "
                         "workers). Adds to existing results, never replaces.")
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="ignore existing results and re-run everything")
    ap.add_argument("--workers", type=int, default=0,
                    help="launch N sharded worker processes (0 = run here)")
    ap.add_argument("--shard", type=int, default=None)
    ap.add_argument("--nshards", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    if args.summary_only:
        import sweep_summary
        sweep_summary.build()
        return

    if args.workers > 0:
        return launch_workers(args)

    # One thread per worker: 8 workers x torch's 12 default threads would
    # oversubscribe 14 cores badly.
    import torch
    torch.set_num_threads(1)

    import progress
    hot = progress.keep_hot()
    print("power: throttling_disabled=%(throttling_disabled)s "
          "execution_state_set=%(execution_state_set)s" % hot, flush=True)

    ensure_prereqs()

    import train

    grid = (QUICK["grid"] if args.quick else
          DEEP["grid"] if args.deep else GRID)
    seeds = (QUICK["seeds"] if args.quick else
          DEEP["seeds"] if args.deep else SEEDS)
    n_folds = (QUICK["n_folds"] if args.quick else
          DEEP["n_folds"] if args.deep else N_FOLDS)
    steps = (QUICK["steps"] if args.quick else
          DEEP["steps"] if args.deep else STEPS)
    max_rows = (QUICK["max_rows"] if args.quick else
          DEEP["max_rows"] if args.deep else MAX_ROWS)
    protocols = (QUICK["protocols"] if args.quick else
          DEEP["protocols"] if args.deep else PROTOCOLS)
    families = [f.strip() for f in args.families.split(",") if f.strip()]

    done = set() if args.force else load_done()
    keys = sorted(grid)
    combos = [dict(zip(keys, v)) for v in itertools.product(*[grid[k]
                                                             for k in keys])]

    # Count first so progress is meaningful.
    planned = []
    for family in families:
        recs = train.load_family(family)
        for protocol in protocols:
            splits = list(train.make_splits(recs, protocol, n_folds, seed=0))
            for (fold_name, split), seed, combo in itertools.product(
                    splits, seeds, combos):
                cfg = dict(combo, seed=seed, steps=steps, max_rows=max_rows,
                           clip=1.0)
                key = (family, protocol, fold_name, seed, cfg["lr"],
                       tuple(cfg["hidden"]), bool(cfg["normalize"]))
                planned.append((family, protocol, fold_name, split, recs, cfg,
                                key))

    if args.nshards:
        planned = [p for i, p in enumerate(planned)
                   if i % args.nshards == args.shard]

    todo = [p for p in planned if p[-1] not in done]
    out_path = shard_results(args.shard)
    print("\n%d runs planned, %d already done, %d to run\n"
          % (len(planned), len(planned) - len(todo), len(todo)), flush=True)

    t_start = time.time()
    for i, (family, protocol, fold_name, split, recs, cfg, key) in \
            enumerate(todo, 1):
        t0 = time.time()
        try:
            rec = train.train_one(family, protocol, fold_name, split, recs,
                                  cfg)
        except Exception:
            rec = {"status": "error", "family": family, "protocol": protocol,
                   "fold": fold_name, "message": traceback.format_exc(),
                   "wall_clock_s": time.time() - t0, **cfg}
        append(rec, out_path)

        elapsed = time.time() - t_start
        rate = elapsed / i
        eta = rate * (len(todo) - i)
        rmse = rec.get("heldout_rmse_N")
        print("[%4d/%4d] %-13s %-8s %-22s lr=%-6g %-8s seed=%d  "
              "%-9s rmse=%s  %5.1fs  eta %.0f min"
              % (i, len(todo), family, protocol, fold_name, cfg["lr"],
                 str(tuple(cfg["hidden"])), cfg["seed"], rec["status"],
                 ("%9.2f" % rmse) if rmse is not None else "      n/a",
                 time.time() - t0, eta / 60.0), flush=True)

    print("\nsweep finished in %.1f min" % ((time.time() - t_start) / 60.0))
    progress.release_hot()

    import sweep_summary
    sweep_summary.build()


def launch_workers(args):
    """Fan the sweep out across N processes, then summarise once."""
    import subprocess
    import sys

    os.makedirs(OUT_DIR, exist_ok=True)
    import progress
    hot = progress.keep_hot()
    print("power: throttling_disabled=%(throttling_disabled)s "
          "execution_state_set=%(execution_state_set)s" % hot, flush=True)

    ensure_prereqs()          # do this once, not in every worker

    base = [sys.executable, "-u", "sweep.py", "--families", args.families]
    if args.quick:
        base.append("--quick")
    if args.deep:
        base.append("--deep")
    if args.force:
        base.append("--force")

    procs = []
    for i in range(args.workers):
        log = open(os.path.join(OUT_DIR, "worker%d.log" % i), "w")
        p = subprocess.Popen(base + ["--shard", str(i),
                                     "--nshards", str(args.workers)],
                             stdout=log, stderr=subprocess.STDOUT)
        procs.append((p, log))
        print("launched shard %d/%d (pid %d)" % (i, args.workers, p.pid),
              flush=True)

    t0 = time.time()
    while any(p.poll() is None for p, _ in procs):
        time.sleep(30)
        done = sum(1 for p, _ in procs if p.poll() is not None)
        n = _count_results()
        print("  [%5.1f min] %d/%d shards finished, %d runs recorded"
              % ((time.time() - t0) / 60.0, done, len(procs), n), flush=True)

    for p, log in procs:
        log.close()
    print("\nall shards finished in %.1f min" % ((time.time() - t0) / 60.0))
    progress.release_hot()

    import sweep_summary
    sweep_summary.build()


def _count_results():
    n = 0
    for path in all_result_files():
        try:
            with open(path) as fh:
                n += sum(1 for line in fh if line.strip())
        except OSError:
            pass
    return n


if __name__ == "__main__":
    main()
