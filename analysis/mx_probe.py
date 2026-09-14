"""Probe: is fit_MX solvable in closed form, for all 6 tires, with real error
bars? William asked for NLLS first, then a check on whether parameters vary
between geometries before committing to a PINN.

fit_MX's residual is LINEAR in QSX1, QSX2, QSX3:

    M_x = F_z * R_0 * (QSX1 - QSX2 * gamma + QSX3 * F_y / F_z0)

so there is no iteration, no seed, and no convergence question -- lstsq gives
the global optimum and the exact parameter covariance.

On cornering data SL is exactly 0, where G_y == 1 identically, so this needs
only lat_params. That means all 6 tires, not the 4 that MZ is limited to.
"""
import numpy as np
import fit_pipeline as fp
import tire_predict as tp

FAM = "lateral"


def design(case, lat_full):
    """Columns of the linear system, matching fit_MX term for term."""
    F_z0 = lat_full[-1]
    R_0 = lat_full[-2] * 0.5 * 0.0254          # in -> m
    p, _ = tp._core(lat_full, FAM)

    F_z = -case["FZ"].ravel()
    gamma = np.sin(case["IA"].ravel() * np.pi / 180)
    alpha = np.tan(case["SA"].ravel() * np.pi / 180)
    M_x = case["MX"].ravel()

    F_y, _, _ = tp._tm_lat(F_z, alpha, gamma, p, F_z0)   # G_y == 1 at SL == 0

    A = np.column_stack([F_z * R_0,
                         -F_z * R_0 * gamma,
                         F_z * R_0 * F_y / F_z0])
    return A, M_x


rows = []
for sid, spec in fp.ALL_SPECS.items():
    if spec.family != FAM:
        continue
    tire_key = sid[len("lat_"):]
    lat_full = np.asarray(tp.PER_TIRE["lat_" + tire_key], float)

    cases = fp.build_cases_lateral(spec)
    sl = np.concatenate([c["SL"].ravel() for c in cases])
    assert np.abs(sl).max() == 0.0, f"{sid}: SL not identically 0 ({np.abs(sl).max()})"

    A, y = map(np.vstack, zip(*[(a, b[:, None]) for a, b in
                                (design(c, lat_full) for c in cases)]))
    y = y.ravel()

    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = A @ beta - y
    n, k = A.shape
    dof = n - k
    sigma2 = resid @ resid / dof
    cov = sigma2 * np.linalg.pinv(A.T @ A)
    se = np.sqrt(np.diag(cov))

    rmse = np.sqrt(resid @ resid / n)
    scale = np.percentile(np.abs(y), 95)

    rows.append(dict(spec=sid, tire=tire_key, n=n, beta=beta, se=se,
                     rmse=rmse, scale=scale, pct=100 * rmse / scale,
                     cond=np.linalg.cond(A)))

print("=" * 78)
print("MX closed-form fit (QSX1, QSX2, QSX3) -- all cornering cases per tire")
print("=" * 78)
for r in rows:
    print(f"\n{r['tire']}   n={r['n']}  cond(A)={r['cond']:.3e}")
    print(f"   RMSE {r['rmse']:.3f} Nm  vs p95|MX| {r['scale']:.2f} Nm"
          f"  -> {r['pct']:.1f}%")
    for nm, b, s in zip(("QSX1", "QSX2", "QSX3"), r["beta"], r["se"]):
        print(f"   {nm} = {b:+.6f}  +/- {s:.6f}   ({abs(b/s):7.1f} sigma)")

print()
print("=" * 78)
print("Does MX vary between geometries? between-tire spread vs within-tire SE")
print("=" * 78)
for j, nm in enumerate(("QSX1", "QSX2", "QSX3")):
    vals = np.array([r["beta"][j] for r in rows])
    ses = np.array([r["se"][j] for r in rows])
    between = vals.std(ddof=1)
    within = np.sqrt((ses ** 2).mean())
    print(f"\n{nm}")
    print(f"   values        {np.array2string(vals, precision=5)}")
    print(f"   between-tire sd  {between:.6f}")
    print(f"   typical within-tire SE {within:.6f}")
    print(f"   ratio  {between / within:8.2f}   "
          f"{'VARIES between geometries' if between / within > 3 else 'no clear geometry effect'}")
