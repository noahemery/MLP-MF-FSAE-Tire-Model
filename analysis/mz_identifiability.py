"""Why the MZ first pass is not finished: B_t is not identifiable.

magic.py has no x0 or bounds for any moment, so these seeds are invented (see
DECISIONS.md). They work in the sense that matters first -- D_t comes out a
physically plausible pneumatic trail, a few centimetres and positive.

The problem is B_t. Raise its upper bound and it chases it, an order of
magnitude, while the residual barely moves. B_t enters as arctan(B_t*alpha_t),
which saturates, so unless the data resolves the low-slip region where B_t sets
the initial slope of the trail curve, many B_t values fit equally well.

Picking a bound would therefore be picking a parameter value, not measuring
one. That is a physics decision for the owner, not something to settle here.

  python analysis/mz_identifiability.py
"""
import numpy as np
from scipy.optimize import least_squares

import fit_pipeline as fp
import magic

TIRE = "205X70_R20_70"          # the only geometry with all four force families
N_SEG = 12                      # enough to see the pattern; the full set is 51
NAMES = ("D_t", "C_t", "B_t", "E_t", "D_r", "C_r", "B_r", "S_ht")

# S_arm is dropped, not fitted. On cornering data SL is identically 0 and |FX|
# is 3-4% of |FY|, so S_arm * F_x carries no information -- fitting it freely
# just sends it to a bound. Dropping it also removes MZ's only dependence on
# long_params and gx_params on this data, which is what would let MZ cover all
# 6 tires rather than the 4 with straight-line runs. Owner's call.
X0 = np.array([0.025, 1.5, 10.0, 0.5, 10.0, 1.0, 10.0, 0.0])


def fit_segment(case, lo, hi, forces):
    lat, lng, gx, gy = forces

    def resid(z):
        x = np.concatenate([z[:7], [0.0], z[7:]])       # S_arm = 0
        return magic.first_pass_MX(case, x, lat, lng, gx, gy)

    r = least_squares(resid, X0, jac='3-point', method='trf', bounds=(lo, hi),
                      ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=8000)
    res = np.asarray(resid(r.x))
    return r.x, float(np.sqrt(np.mean(res ** 2)))


def main():
    forces = tuple(np.load("outputs/specs/%s_%s.npy" % (p, TIRE))
                   for p in ("lat", "long", "GX", "GY"))
    cases = fp.build_cases_lateral(fp.ALL_SPECS["lat_%s" % TIRE])[:N_SEG]
    scale = float(np.mean([np.percentile(np.abs(np.asarray(c["MZ"]).ravel()), 95)
                           for c in cases]))

    print("MZ first pass, %s, %d segments, p95|MZ| = %.1f Nm\n"
          % (TIRE, len(cases), scale))
    print("%10s %8s %12s %14s %10s"
          % ("B_t cap", "err %", "B_t mean", "B_t range", "D_t mean"))
    print("-" * 60)

    for cap in (200.0, 1000.0, 5000.0, 25000.0):
        lo = np.array([0.0, 0.5, 0.0, -20.0, -500.0, 0.0, 0.0, -0.2])
        hi = np.array([0.20, 8.0, cap, 1.0, 500.0, 8.0, cap, 0.2])
        B, rm = [], []
        for c in cases:
            x, r = fit_segment(c, lo, hi, forces)
            B.append(x)
            rm.append(r)
        B = np.array(B)
        print("%10g %7.1f%% %12.1f %6.0f-%-7.0f %10.4f"
              % (cap, 100 * np.mean(rm) / scale, B[:, 2].mean(),
                 B[:, 2].min(), B[:, 2].max(), B[:, 0].mean()))

    print("\nB_t tracks the cap across two orders of magnitude while the error")
    print("stays flat. It is not determined by this data. D_t meanwhile stays")
    print("in the 0.02-0.06 m band a real pneumatic trail occupies.")


if __name__ == "__main__":
    main()
