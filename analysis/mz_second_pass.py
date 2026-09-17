"""MZ second pass: low-level per-segment coefficients -> 36 Q-parameters.

Runs every (tire, seed) combination in parallel -- 4 tires x 3 seeds = 12 jobs
on 12 physical cores -- and keeps the best cost per tire. The multi-start is not
just for CPU: these second passes are known to settle in flat regions (the
lateral one improved cost 0.021% over 27,108 evaluations), so several starts is
the cheap insurance, and it is P4 of the refinement plan.

magic.py has no x0 for this. Seed 0 is derived from the low-level fits rather
than guessed: each residual row is (low_level_param - model_of_it), so for rows
linear in their Q-parameters the leading coefficient reads straight off the
median of the low-level column --

    C_t  - QCZ1                          -> QCZ1 = median(C_t)
    S_ht - (QHZ1 + ...)                  -> QHZ1 = median(S_ht)
    S_arm- R_0*(QSZ1 + ...)              -> QSZ1 = median(S_arm)/R_0
    D_r  - F_z*R_0*((QDZ6 + ...) ...)    -> QDZ6 = median(D_r)/(F_z*R_0)
    D_t  - F_z*(R_0/F_z0)*(QDZ1 + ...)   -> QDZ1 = median(D_t)/(F_z*R_0/F_z0)

Seeds 1 and 2 scale that vector by 0.5 and 2.0.

CARRIED-FORWARD CAVEAT: B_t is set by its bound in 51% of segments, not by the
data, so QBZ1/2/3/5/6 inherit that. Everything here is conditional on that
bound until the owner decides how to constrain B_t. Running this was directed
explicitly; it does not resolve the caveat.

Only the 4 tires with longitudinal data can run: second_pass_MZ reads
R_0 = long_params[-2], so the cornering-only 160X75 pair has no source for it.

  python analysis/mz_second_pass.py
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")       # one core per worker,
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # no oversubscription
os.environ.setdefault("MKL_NUM_THREADS", "1")

import concurrent.futures as cf

import numpy as np
import pandas as pd

TIRES = ("205X70_R20_70", "205X70_R20_80", "180X60_R20_60", "180X60_R20_70")
SEED_SCALES = (1.0, 0.5, 2.0)
MZ_ORDER = ("D_t", "C_t", "B_t", "E_t", "D_r", "C_r", "B_r", "S_arm", "S_ht")
Q_NAMES = (["QDZ%d" % i for i in range(1, 12)] + ["QCZ1"]
           + ["QBZ%d" % i for i in range(1, 12)]
           + ["QEZ%d" % i for i in range(1, 6)]
           + ["QHZ%d" % i for i in range(1, 5)]
           + ["QSZ%d" % i for i in range(1, 5)])
TOL = 2.3e-16
CAP_NFEV = 60_000


def base_seed(med, F_z_mean, R_0, F_z0):
    x = np.zeros(36)
    i = {n: k for k, n in enumerate(Q_NAMES)}
    x[i["QDZ1"]] = med["D_t"] / max(F_z_mean * R_0 / F_z0, 1e-9)
    x[i["QDZ6"]] = med["D_r"] / max(F_z_mean * R_0, 1e-9)
    x[i["QCZ1"]] = med["C_t"]
    x[i["QBZ1"]] = med["B_t"]
    x[i["QEZ1"]] = med["E_t"]
    x[i["QBZ9"]] = med["B_r"]
    x[i["QHZ1"]] = med["S_ht"]
    x[i["QSZ1"]] = med["S_arm"] / max(R_0, 1e-9)
    return x


def run_one(job):
    """One (tire, seed) fit. Runs in its own process."""
    tire, si, scale = job

    # progress patches scipy.optimize.least_squares, so it must be installed
    # BEFORE the name is bound, and called through the module attribute. An
    # earlier version did `from scipy.optimize import least_squares` at module
    # top and got the unpatched function -- no heartbeats for 23 minutes.
    import progress
    progress.install(every=2000, cap_nfev=CAP_NFEV)
    progress.keep_hot()

    import scipy.optimize
    import fit_pipeline as fp
    import magic

    lat = np.load("outputs/specs/lat_%s.npy" % tire)
    lng = np.load("outputs/specs/long_%s.npy" % tire)
    gx = np.load("outputs/specs/GX_%s.npy" % tire)
    gy = np.load("outputs/specs/GY_%s.npy" % tire)
    cases = fp.build_cases_lateral(fp.ALL_SPECS["lat_%s" % tire])

    g = (pd.read_parquet("outputs/mz_first_pass.parquet")
           .query("tire == @tire").sort_values("segment"))
    if len(g) != len(cases):
        raise RuntimeError("%s: %d low-level rows vs %d segments"
                           % (tire, len(g), len(cases)))
    # (9, n_seg): second_pass_MZ indexes [0]..[8] then subscripts per segment.
    BCDE = np.vstack([g[p].to_numpy(dtype=float) for p in MZ_ORDER])

    F_z0 = float(lat[-1])
    R_0 = float(lng[-2]) * 0.5 * 0.0254
    F_z_mean = float(np.mean([np.abs(c["FZ"]).mean() for c in cases]))
    med = {p: float(g[p].median()) for p in MZ_ORDER}
    x0 = base_seed(med, F_z_mean, R_0, F_z0) * scale

    def resid(x):
        return magic.second_pass_MZ(cases, F_z0, 1, BCDE, x,
                                    lat, lng, gx, gy)

    r0 = np.asarray(resid(x0))
    res = scipy.optimize.least_squares(
        resid, x0, jac='3-point', method='lm',
        ftol=TOL, xtol=TOL, gtol=TOL, max_nfev=int(1e8), verbose=0)

    J = np.asarray(res.jac)
    out = {"tire": tire, "seed": si, "seed_scale": scale,
           "n_segments": len(cases), "n_resid": int(np.asarray(res.fun).size),
           "cost0": float(0.5 * r0 @ r0), "cost": float(res.cost),
           "nfev": int(res.nfev), "status": int(res.status),
           "optimality": float(res.optimality), "success": bool(res.success),
           "message": str(res.message)}
    try:
        sv = np.linalg.svd(J, compute_uv=False)
        out["jac_rank"] = int(np.linalg.matrix_rank(J))
        out["jac_cond"] = (float(sv.max() / sv.min()) if sv.min() > 0
                           else float("inf"))
    except np.linalg.LinAlgError:
        out["jac_rank"], out["jac_cond"] = None, None
    for n, v in zip(Q_NAMES, res.x):
        out[n] = float(v)
    return out


def main():
    jobs = [(t, i, s) for t in TIRES for i, s in enumerate(SEED_SCALES)]
    print("%d jobs (%d tires x %d seeds) on %d workers"
          % (len(jobs), len(TIRES), len(SEED_SCALES), len(jobs)), flush=True)

    rows = []
    with cf.ProcessPoolExecutor(max_workers=len(jobs)) as ex:
        futs = {ex.submit(run_one, j): j for j in jobs}
        for fut in cf.as_completed(futs):
            tire, si, scale = futs[fut]
            try:
                r = fut.result()
                rows.append(r)
                print("  %-16s seed %d (x%.1f)  cost %.6e -> %.6e  "
                      "nfev %6d  rank %s/36"
                      % (tire, si, scale, r["cost0"], r["cost"], r["nfev"],
                         r.get("jac_rank")), flush=True)
            except Exception as exc:
                print("  %-16s seed %d FAILED: %r" % (tire, si, exc),
                      flush=True)

    if not rows:
        print("no successful fits")
        return

    df = pd.DataFrame(rows)
    df.to_parquet("outputs/mz_second_pass_allseeds.parquet", index=False)

    best = df.loc[df.groupby("tire")["cost"].idxmin()].copy()
    best.to_parquet("outputs/mz_second_pass.parquet", index=False)
    for _, r in best.iterrows():
        np.save("outputs/specs/MZ_%s.npy" % r["tire"],
                np.hstack(([r[n] for n in Q_NAMES],
                           np.load("outputs/specs/long_%s.npy" % r["tire"])[-2],
                           r["n_segments"] * 0 + np.load(
                               "outputs/specs/lat_%s.npy" % r["tire"])[-1])))

    print("\nbest per tire:")
    print(best[["tire", "seed", "seed_scale", "cost0", "cost", "nfev",
                "jac_rank", "jac_cond", "success"]].to_string(index=False))
    spread = df.groupby("tire")["cost"].agg(["min", "max"])
    spread["spread_pct"] = 100 * (spread["max"] - spread["min"]) / spread["min"]
    print("\nmulti-start spread (does the seed matter?):")
    print(spread.to_string())
    print("\nwrote outputs/mz_second_pass.parquet and "
          "mz_second_pass_allseeds.parquet")


if __name__ == "__main__":
    main()
