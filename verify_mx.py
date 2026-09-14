"""Prove the MX pipeline solves magic.py's fit_MX, for every tire.

run_mx solves a linear system rather than calling fit_MX in a loop, because
fit_MX's residual is linear in QSX1/QSX2/QSX3. This script is what makes that
substitution safe. Three gates per tire:

  1. magic.fit_MX's own residual, evaluated at our solution, matches the linear
     system's residual. Same model, not merely a similar one.
  2. least_squares driven by magic.fit_MX from a NEUTRAL seed converges to the
     same parameters. So the closed form is the optimum fit_MX would reach,
     not a different point that happens to fit.
  3. G_y is exactly 1 on this data, which is what lets MX drop gy_params and
     cover all 6 tires instead of the 4 with straight-line runs.

Run:  python verify_mx.py
Exit status is non-zero if any gate fails.
"""
import sys

import numpy as np
from scipy.optimize import least_squares

import fit_pipeline as fp
import magic

TOL_RESID = 1e-8        # Nm, on residuals of order 10-100 Nm
TOL_PARAM = 1e-6        # relative, on parameters of order 0.01-2
NEUTRAL_SEED = [0.0, 1.0, 0.0]


def main():
    failures = []
    print("%-18s %14s %14s %10s" % ("tire", "max|dresid|", "max rel dparam",
                                    "G_y"))
    print("-" * 60)

    for spec in fp.MX_SPECS:
        tire = spec.spec_id[len("MX_"):]
        lat_vec = np.load("outputs/specs/" + spec.depends_on + ".npy")
        lat_spec = fp.ALL_SPECS[spec.lat_source]
        cases = fp.build_cases_lateral(lat_spec)

        # Gate 3 first: everything else depends on it.
        sl = np.concatenate([np.asarray(c["SL"]).ravel() for c in cases])
        core = fp.strip_geometry(lat_vec, "lateral")
        F_z0 = lat_vec[-1]
        c0 = cases[0]
        F_z = -np.asarray(c0["FZ"]).ravel()
        gamma = np.sin(np.asarray(c0["IA"]).ravel() * np.pi / 180)
        alpha = np.tan(np.asarray(c0["SA"]).ravel() * np.pi / 180)
        # Deliberately ZERO R-params: if G_y is 1 with these, it is 1 with any,
        # which is the claim being tested.
        G_y, _ = magic.GY(F_z, F_z0, np.zeros_like(F_z), alpha, gamma,
                          np.zeros(15), core)
        gy_dev = float(np.abs(np.asarray(G_y) - 1.0).max())

        A, y, _ = fp._mx_columns(cases, lat_vec, with_pressure=False)
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)

        # Gate 1: magic's residual at our solution.
        r_magic = np.asarray(magic.fit_MX(cases, beta, lat_vec,
                                          np.zeros(15))).ravel()
        r_linear = A @ beta - y
        dresid = float(np.abs(r_magic - r_linear).max())

        # Gate 2: NLLS on magic's own function, from a neutral seed.
        res = least_squares(
            lambda x: magic.fit_MX(cases, x, lat_vec, np.zeros(15)),
            NEUTRAL_SEED, jac='3-point', method='lm',
            ftol=1e-15, xtol=1e-15, gtol=1e-15)
        dparam = float(np.abs((res.x - beta) / beta).max())

        ok = (dresid < TOL_RESID and dparam < TOL_PARAM
              and gy_dev == 0.0 and np.abs(sl).max() == 0.0)
        print("%-18s %14.3e %14.3e %10.1e %s"
              % (tire, dresid, dparam, gy_dev, "" if ok else "  <-- FAIL"))
        if not ok:
            failures.append(tire)

    print()
    if failures:
        print("FAILED: " + ", ".join(failures))
        return 1
    print("All %d tires pass. The closed-form MX solve is magic.fit_MX's own "
          "optimum." % len(fp.MX_SPECS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
