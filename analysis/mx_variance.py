"""Honest version of the MX geometry question.

The asymptotic SE from a single pooled fit assumes iid residuals. With ~90k
autocorrelated samples from a rig sweep that assumption is badly violated and
the SE comes out far too small -- the same near-duplicate problem that forces
GroupKFold by segment everywhere else in this project.

So: fit MX per segment, then compare variance BETWEEN tires against variance
WITHIN a tire across its own segments. No iid assumption anywhere.

Also checks pressure as a lurking variable: each spec pools 6-9 distinct
inflation pressures, so a "geometry effect" could really be a pressure effect.
"""
import numpy as np
import fit_pipeline as fp
import tire_predict as tp

NAMES = ("QSX1", "QSX2", "QSX3")


def fit_one(case, lat_full):
    F_z0 = lat_full[-1]
    R_0 = lat_full[-2] * 0.5 * 0.0254
    p, _ = tp._core(lat_full, "lateral")
    F_z = -case["FZ"].ravel()
    gamma = np.sin(case["IA"].ravel() * np.pi / 180)
    alpha = np.tan(case["SA"].ravel() * np.pi / 180)
    M_x = case["MX"].ravel()
    F_y, _, _ = tp._tm_lat(F_z, alpha, gamma, p, F_z0)
    A = np.column_stack([F_z * R_0, -F_z * R_0 * gamma,
                         F_z * R_0 * F_y / F_z0])
    beta, *_ = np.linalg.lstsq(A, M_x, rcond=None)
    P = case["P"].ravel()
    P = P[P > 1.0]
    return beta, (P.mean() if P.size else np.nan), A.shape[0], np.linalg.cond(A)


per_tire = {}
for sid, spec in fp.ALL_SPECS.items():
    if spec.family != "lateral":
        continue
    tire = sid[len("lat_"):]
    lat_full = np.asarray(tp.PER_TIRE["lat_" + tire], float)
    out = []
    for c in fp.build_cases_lateral(spec):
        b, pmean, n, cond = fit_one(c, lat_full)
        if np.isfinite(cond) and cond < 1e8 and n > 200:
            out.append((b, pmean, n))
    per_tire[tire] = out
    B = np.array([o[0] for o in out])
    print(f"{tire:16s} {len(out):3d} segments   "
          + "  ".join(f"{nm} {B[:, j].mean():+.5f} (sd {B[:, j].std(ddof=1):.5f})"
                      for j, nm in enumerate(NAMES)))

print()
print("=" * 74)
print("Variance components -- no iid assumption, segment is the unit")
print("=" * 74)
tires = list(per_tire)
for j, nm in enumerate(NAMES):
    means = np.array([np.mean([o[0][j] for o in per_tire[t]]) for t in tires])
    within = np.sqrt(np.mean([np.var([o[0][j] for o in per_tire[t]], ddof=1)
                              for t in tires]))
    between = means.std(ddof=1)
    # standard error of a tire-level mean, from its own segment scatter
    se_mean = np.sqrt(np.mean([np.var([o[0][j] for o in per_tire[t]], ddof=1)
                               / len(per_tire[t]) for t in tires]))
    print(f"\n{nm}")
    print(f"   tire means         {np.array2string(means, precision=5)}")
    print(f"   between-tire sd    {between:.6f}")
    print(f"   within-tire sd     {within:.6f}   (segment to segment)")
    print(f"   SE of tire mean    {se_mean:.6f}")
    print(f"   between / within   {between / within:6.2f}")
    print(f"   between / SE(mean) {between / se_mean:6.2f}  "
          f"<- the honest significance ratio")

print()
print("=" * 74)
print("Pressure as a lurking variable: does QSX track P within a tire?")
print("=" * 74)
for j, nm in enumerate(NAMES):
    print(f"\n{nm}")
    for t in tires:
        v = np.array([o[0][j] for o in per_tire[t]])
        P = np.array([o[1] for o in per_tire[t]])
        ok = np.isfinite(P) & np.isfinite(v)
        if ok.sum() < 4:
            print(f"   {t:16s} too few segments"); continue
        r = np.corrcoef(P[ok], v[ok])[0, 1]
        print(f"   {t:16s} corr(P, {nm}) = {r:+.3f}   "
              f"P range {P[ok].min():.1f}-{P[ok].max():.1f} kPa")
