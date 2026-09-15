"""MZ first pass across all 6 tires, as a sensitivity study.

Owner's direction, 2026-09-14: "you can neglect FX for the tires with no data,
the point of this first pass is to do a sensitivity study anyways. Into the
future it should be reintroduced once a PINN is utilized."

So F_x is neglected ONLY for the two 160X75 specs, which have no longitudinal
or G_x fit. The other four keep S_arm free, because they have the data.

Neglecting F_x means S_arm = 0, which zeroes the only term F_x enters. The two
tires still need *some* long/gx vector passed to first_pass_MX because it
computes G_x unconditionally and reads long_params[-1]. Rather than assume that
is harmless, prove_fx_independence() fits with two DIFFERENT donor tires'
vectors and checks the residuals are bit-identical.

Why the first pass only. The second pass fits 36 Q-params to these per-segment
results. Running it now would propagate a parameter the data does not constrain
(B_t -- see below) into 36 more. That waits on the owner's call.

Writes outputs/mz_first_pass.parquet and prints the sensitivity table.

  python analysis/mz_first_pass.py
"""
import os

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

import fit_pipeline as fp
import magic

NAMES = ("D_t", "C_t", "B_t", "E_t", "D_r", "C_r", "B_r", "S_arm", "S_ht")

# Invented seeds and bounds -- magic.py has none for any moment. Bounds are the
# widened set: the original narrow ones pinned 7 of 9 parameters and gave 23.6%
# error, these give 13.0% with one at a bound. See analysis/mz_identifiability.
X0 = np.array([0.025, 1.5, 10.0, 0.5, 10.0, 1.0, 10.0, 0.0, 0.0])
LOWER = np.array([0.0, 0.5, 0.0, -20.0, -500.0, 0.0, 0.0, -0.5, -0.2])
UPPER = np.array([0.20, 8.0, 200.0, 1.0, 500.0, 8.0, 200.0, 0.5, 0.2])

# B_t's upper bound is a STATED CHOICE, not a measurement. B_t follows whatever
# cap it is given across two orders of magnitude for a flat residual. Reported
# as unconstrained in the sensitivity output rather than presented as fitted.
B_T_CAP_IS_ARBITRARY = True

TIRES = ("160X75_R20_70", "160X75_R20_80", "205X70_R20_70",
         "205X70_R20_80", "180X60_R20_60", "180X60_R20_70")
NEGLECT_FX = ("160X75_R20_70", "160X75_R20_80")     # no long/gx fit exists
DONORS = ("205X70_R20_70", "180X60_R20_60")         # for the invariance proof


def load(prefix, tire):
    return np.load(os.path.join("outputs", "specs",
                                "%s_%s.npy" % (prefix, tire)))


def residual_fn(case, lat, lng, gx, gy, neglect_fx):
    """first_pass_MX's residual, optionally with S_arm pinned to 0."""
    if not neglect_fx:
        return lambda x: magic.first_pass_MX(case, x, lat, lng, gx, gy)

    def resid(z):
        x = np.concatenate([z[:7], [0.0], z[7:]])
        return magic.first_pass_MX(case, x, lat, lng, gx, gy)
    return resid


def prove_fx_independence(tire, case, lat, gy):
    """With S_arm = 0, the long/gx vectors must not affect the residual."""
    out = []
    for donor in DONORS:
        f = residual_fn(case, lat, load("long", donor), load("GX", donor), gy,
                        neglect_fx=True)
        out.append(np.asarray(f(np.delete(X0, 7))).ravel())
    same = np.array_equal(out[0], out[1])
    return same, float(np.abs(out[0] - out[1]).max())


def sensitivity(J, resid, theta, free):
    """Relative sensitivity per parameter, plus the unconstrained directions.

    rel_sens[j] = ||dr/dtheta_j|| * |theta_j| / ||r||, i.e. how much a 100%
    change in that parameter moves the residual relative to the residual's own
    size. Small means the data barely sees the parameter.
    """
    rn = float(np.linalg.norm(resid)) or 1.0
    rel = np.abs(theta[free]) * np.linalg.norm(J, axis=0) / rn
    sv = np.linalg.svd(J, compute_uv=False)
    cond = float(sv.max() / sv.min()) if sv.min() > 0 else float("inf")
    return rel, cond, int(np.linalg.matrix_rank(J))


def main():
    rows, notes = [], []
    for tire in TIRES:
        neglect = tire in NEGLECT_FX
        free = [i for i in range(9) if not (neglect and i == 7)]

        lat = load("lat", tire)
        gy_tire = tire if not neglect else None
        # G_y is identically 1 on this data (SL == 0), so for the cornering-only
        # tires any R-vector gives the same answer -- same argument as MX.
        gy = load("GY", gy_tire) if gy_tire else load("GY", DONORS[0])
        lng = load("long", tire if not neglect else DONORS[0])
        gx = load("GX", tire if not neglect else DONORS[0])

        cases = fp.build_cases_lateral(fp.ALL_SPECS["lat_%s" % tire])

        if neglect:
            ok, delta = prove_fx_independence(tire, cases[0], lat, gy)
            notes.append("%-16s F_x independence: %s (max|delta| = %.1e)"
                         % (tire, "EXACT" if ok else "FAILED", delta))
            if not ok:
                raise RuntimeError(tire + ": residual depends on the donor "
                                   "long/gx vectors even with S_arm = 0")

        x0 = X0[free]
        lo, hi = LOWER[free], UPPER[free]
        for si, case in enumerate(cases):
            f = residual_fn(case, lat, lng, gx, gy, neglect)
            r = least_squares(f, x0, jac='3-point', method='trf',
                              bounds=(lo, hi), ftol=1e-12, xtol=1e-12,
                              gtol=1e-12, max_nfev=8000)
            res = np.asarray(f(r.x)).ravel()
            theta = np.zeros(9)
            theta[free] = r.x
            rel, cond, rank = sensitivity(np.asarray(r.jac), res, theta, free)

            scale = float(np.percentile(np.abs(np.asarray(case["MZ"]).ravel()),
                                        95))
            rec = {"tire": tire, "segment": si, "neglect_fx": neglect,
                   "n_rows": len(res), "rmse_Nm": float(np.sqrt(np.mean(res**2))),
                   "p95_abs_MZ_Nm": scale, "jac_cond": cond, "jac_rank": rank,
                   "n_free": len(free)}
            for j, nm in enumerate(NAMES):
                rec[nm] = float(theta[j])
                rec["at_bound_" + nm] = bool(
                    j in free and (theta[j] <= LOWER[j] + 1e-7
                                   or theta[j] >= UPPER[j] - 1e-7))
            for k, j in enumerate(free):
                rec["sens_" + NAMES[j]] = float(rel[k])
            rows.append(rec)

    df = pd.DataFrame(rows)
    os.makedirs("outputs", exist_ok=True)
    df.to_parquet("outputs/mz_first_pass.parquet", index=False)

    print("\n".join(notes))
    print("\nwrote outputs/mz_first_pass.parquet (%d segments across %d tires)"
          % (len(df), df.tire.nunique()))

    print("\n" + "=" * 74)
    print("Fit quality per tire (in-sample -- first pass, per segment)")
    print("=" * 74)
    g = df.groupby("tire")
    for tire in TIRES:
        d = g.get_group(tire)
        print("  %-16s %3d seg  rmse %6.2f Nm / p95 %6.2f  -> %5.1f%%%s"
              % (tire, len(d), d.rmse_Nm.mean(), d.p95_abs_MZ_Nm.mean(),
                 100 * d.rmse_Nm.mean() / d.p95_abs_MZ_Nm.mean(),
                 "   (F_x neglected)" if tire in NEGLECT_FX else ""))

    print("\n" + "=" * 74)
    print("SENSITIVITY STUDY -- relative sensitivity, all segments pooled")
    print("  rel = ||dr/dtheta|| * |theta| / ||r||   (bigger = data sees it)")
    print("=" * 74)
    print("  %-7s %12s %12s %10s %s"
          % ("param", "median", "range", "at bound", "verdict"))
    print("  " + "-" * 70)
    for nm in NAMES:
        col = "sens_" + nm
        if col not in df or df[col].notna().sum() == 0:
            continue
        s = df[col].dropna()
        ab = 100.0 * df["at_bound_" + nm].mean()
        med = s.median()
        verdict = ("UNCONSTRAINED" if med < 1e-3 else
                   "weak" if med < 1e-2 else "constrained")
        if nm == "B_t":
            verdict += ", cap arbitrary"
        print("  %-7s %12.2e %5.0e-%-6.0e %9.0f%% %s"
              % (nm, med, s.min(), s.max(), ab, verdict))

    print("\n  Jacobian: median cond %.2e, rank %d of %d free params"
          % (df.jac_cond.median(), int(df.jac_rank.median()),
             int(df.n_free.median())))
    print("\n  Read B_t's row with the caveat in this file's header: its bound")
    print("  is a stated choice, so its fitted value is not a measurement.")


if __name__ == "__main__":
    main()
