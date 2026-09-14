"""Run every spec through fit_pipeline and assemble the baseline artifacts.

One subprocess per spec (see run_spec_worker.py), at most --workers at a time.
The real bound on a fit is fit_pipeline.CAP_NFEV, a deterministic evaluation
cap -- a wall-clock kill would stop the reference and the pipeline at
different nfev and make the bitwise gate unpassable. --cap-minutes is only a
backstop against a genuine hang. G_x waits for its long_* spec and G_y for its
lat_* spec; a spec whose upstream failed is recorded skipped_upstream rather
than fitted against a garbage vector.

Results are written per spec as they finish, so a crash late in the run does
not lose the specs that already completed. Re-running skips specs that already
have a result unless --force is given.

  python run_baseline.py                          # all 18
  python run_baseline.py --only lat_180X60_R20_70,GY_180X60_R20_70
  python run_baseline.py --assemble-only          # rebuild parquet from json
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

import fit_pipeline as fp
from run_spec_worker import result_path, vector_path, OUT_DIR

LOG_DIR = os.path.join("outputs", "logs")
ARTIFACT_DIR = "outputs"

# Family ordering for launch priority: the deep chains start first.
_FAMILY_ORDER = {"lateral": 0, "longitudinal": 1, "gx": 2, "gy": 2}

# Caveats attached to the diagnostics table so a number is never read clean.
CAVEATS = {
    "gy": ("first_pass_GY reads camber from the SA channel, not IA "
           "(magic.py:1364). Known defect, reported not patched -- these "
           "R-params carry it."),
    "gx": ("The G_x>1 penalty branch at magic.py:1102 is dead "
           "(.any() > 1 is always False), so G_x is unconstrained above 1."),
    "longitudinal": ("second_pass_x segment 0 lacks the 1e-8 guard the loop "
                     "body has (magic.py:736 vs :769)."),
}


def load_record(spec_id):
    path = result_path(spec_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except json.JSONDecodeError:
        return None


def write_record(spec_id, record):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(result_path(spec_id), "w") as fh:
        json.dump(record, fh, indent=2)


def launch(spec_id):
    os.makedirs(LOG_DIR, exist_ok=True)
    fh = open(os.path.join(LOG_DIR, spec_id + ".log"), "w")
    env = dict(os.environ, MPLBACKEND="Agg", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTHONUNBUFFERED="1")
    proc = subprocess.Popen([sys.executable, "run_spec_worker.py", spec_id],
                            stdout=fh, stderr=subprocess.STDOUT, env=env)
    return proc, time.time(), fh


def schedule(spec_ids, workers, cap_seconds, force):
    specs = {sid: fp.ALL_SPECS[sid] for sid in spec_ids}
    pending, done = set(spec_ids), {}

    if not force:
        for sid in list(pending):
            rec = load_record(sid)
            if rec is not None and rec.get("status") == "ok":
                done[sid] = "ok"
                pending.discard(sid)
                print("skip (already done): " + sid, flush=True)

    running = {}
    t_start = time.time()

    def priority(sid):
        s = specs[sid]
        return (s.depends_on is not None, _FAMILY_ORDER[s.family], sid)

    while pending or running:
        for sid in list(running):
            proc, t0, fh = running[sid]
            rc = proc.poll()
            elapsed = time.time() - t0
            if rc is not None:
                fh.close()
                rec = load_record(sid)
                status = rec.get("status", "error") if rec else "error"
                if rec is None:
                    write_record(sid, {
                        "spec_id": sid, "family": specs[sid].family,
                        "status": "error", "wall_clock_s": elapsed,
                        "message": ("worker exited " + str(rc) +
                                    " without writing a result; see " +
                                    os.path.join(LOG_DIR, sid + ".log"))})
                done[sid] = status
                del running[sid]
                print("[%6.1fs] %-22s %-16s %.1fs"
                      % (time.time() - t_start, sid, status, elapsed),
                      flush=True)
            elif elapsed > cap_seconds:
                proc.kill()
                proc.wait()
                fh.close()
                write_record(sid, {
                    "spec_id": sid, "family": specs[sid].family,
                    "status": "timeout", "wall_clock_s": elapsed,
                    "message": ("killed at the %.0f min cap; partial output in "
                                % (cap_seconds / 60.0)
                                + os.path.join(LOG_DIR, sid + ".log"))})
                done[sid] = "timeout"
                del running[sid]
                print("[%6.1fs] %-22s TIMEOUT after %.1fs -- killed"
                      % (time.time() - t_start, sid, elapsed), flush=True)

        progressed = False
        for sid in sorted(pending, key=priority):
            if len(running) >= workers:
                progressed = True          # a slot will free up; not stuck
                break
            dep = specs[sid].depends_on
            # Satisfied if it succeeded this run, or -- when it was not
            # scheduled at all (--only) -- if its vector is already on disk.
            satisfied = (dep is None or done.get(dep) == "ok"
                         or (dep not in specs
                             and os.path.exists(vector_path(dep))))
            if satisfied:
                running[sid] = launch(sid)
                pending.discard(sid)
                progressed = True
                print("[%6.1fs] %-22s launched" % (time.time() - t_start, sid),
                      flush=True)
            elif dep in done or dep not in specs:
                # Upstream finished badly, or was never scheduled and has no
                # vector on disk. Either way there is nothing to fit against.
                write_record(sid, {
                    "spec_id": sid, "family": specs[sid].family,
                    "status": "skipped_upstream", "wall_clock_s": 0.0,
                    "depends_on": dep,
                    "message": ("upstream " + str(dep) + " finished as " +
                                str(done.get(dep, "not scheduled")) +
                                "; not fitting against a garbage vector")})
                done[sid] = "skipped_upstream"
                pending.discard(sid)
                progressed = True
                print("[%6.1fs] %-22s skipped_upstream (%s)"
                      % (time.time() - t_start, sid, dep), flush=True)
            else:
                progressed = True          # dep is still running; just wait

        if running:
            time.sleep(2)
        elif pending and not progressed:
            # Nothing running, nothing launchable, nothing skippable: a
            # dependency cycle or a spec id typo. Fail loudly, do not spin.
            raise SystemExit("scheduler deadlock; cannot resolve: "
                             + ", ".join(sorted(pending)))

    return done, time.time() - t_start


def assemble(spec_ids):
    params, diags, uncert = [], [], []

    for sid in spec_ids:
        rec = load_record(sid)
        if rec is None:
            continue
        family = rec.get("family")

        for name, value in (rec.get("params") or {}).items():
            params.append({"spec_id": sid, "family": family,
                           "param_name": name, "value": value})

        for name, value in (rec.get("std_err") or {}).items():
            uncert.append({"spec_id": sid, "family": family,
                           "param_name": name, "std_err": value})

        diags.append({
            "spec_id": sid,
            "family": family,
            "block_line": rec.get("block_line"),
            "depends_on": rec.get("depends_on"),
            "status": rec.get("status"),
            "n_cases": rec.get("n_cases"),
            "n_rows": rec.get("n_rows"),
            "fz0_used": rec.get("fz0_used"),
            "solver_status": rec.get("solver_status"),
            "success": rec.get("success"),
            "nfev": rec.get("nfev"),
            "njev": rec.get("njev"),
            "cost": rec.get("cost"),
            "optimality": rec.get("optimality"),
            "jac_rank": rec.get("jac_rank"),
            "jac_cond": rec.get("jac_cond"),
            "jac_shape": (json.dumps(rec.get("jac_shape"))
                          if rec.get("jac_shape") else None),
            "residual_variance": rec.get("residual_variance"),
            "wall_clock_s": rec.get("wall_clock_s"),
            "caveat": CAVEATS.get(family),
            "message": rec.get("message"),
        })

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    written = []
    for frame, name in ((pd.DataFrame(params), "baseline_params.parquet"),
                        (pd.DataFrame(diags), "baseline_diagnostics.parquet"),
                        (pd.DataFrame(uncert),
                         "baseline_param_uncertainty.parquet")):
        if frame.empty:
            print("nothing to write for " + name)
            continue
        path = os.path.join(ARTIFACT_DIR, name)
        frame.to_parquet(path, index=False)
        written.append((path, len(frame)))
        print("wrote " + path + "  (" + str(len(frame)) + " rows)")
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None,
                    help="comma-separated spec ids (default: all 18)")
    ap.add_argument("--workers", type=int, default=10,
                    help="concurrent spec processes. Stage 1 has 10 independent "
                         "specs (6 lateral + 4 longitudinal), so 10 runs them in a "
                         "single wave; more than that buys nothing. Measured ~530MB "
                         "per worker, so 10 is ~5.3GB of 16GB.")
    ap.add_argument("--cap-minutes", type=float, default=360.0,
                    help="wall-clock backstop against a genuine hang only; "
                         "the real bound is the deterministic CAP_NFEV")
    ap.add_argument("--force", action="store_true",
                    help="re-run specs that already have an ok result")
    ap.add_argument("--assemble-only", action="store_true")
    args = ap.parse_args()

    spec_ids = (list(fp.ALL_SPECS) if args.only is None
                else [s.strip() for s in args.only.split(",") if s.strip()])
    unknown = [s for s in spec_ids if s not in fp.ALL_SPECS]
    if unknown:
        raise SystemExit("unknown spec ids: " + ", ".join(unknown))

    import progress
    hot = progress.keep_hot()
    print('power: throttling_disabled=%(throttling_disabled)s execution_state_set=%(execution_state_set)s' % hot, flush=True)

    if not args.assemble_only:
        print("scheduling " + str(len(spec_ids)) + " specs, " +
              str(args.workers) + " workers, " + str(args.cap_minutes) +
              " min cap\n", flush=True)
        done, total = schedule(spec_ids, args.workers,
                               args.cap_minutes * 60.0, args.force)

        print("\n" + "=" * 62)
        print("total wall clock: %.1fs (%.2f h)" % (total, total / 3600.0))
        by_status = {}
        for sid, st in done.items():
            by_status.setdefault(st, []).append(sid)
        for st in sorted(by_status):
            print("  %-18s %d" % (st, len(by_status[st])))
            for sid in sorted(by_status[st]):
                print("      " + sid)
        print("=" * 62 + "\n")

    # Always assemble from ALL specs, never just the --only subset,
    # otherwise a partial run silently truncates the parquet.
    assemble(list(fp.ALL_SPECS))
    progress.release_hot()


if __name__ == "__main__":
    main()
