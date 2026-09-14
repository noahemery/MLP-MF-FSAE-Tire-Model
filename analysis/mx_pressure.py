"""Separate the pressure effect from the geometry effect for MX.

Per-segment fits are ill-conditioned: a segment is load-sorted, so F_z is
nearly constant inside it and the QSX1 / QSX3 columns go collinear. MX is only
identifiable when pooled across loads.

So bin by pressure instead. Within one (tire, pressure bin) we still span many
loads, so the fit stays identifiable, and we can ask two clean questions:

  across pressure bins, same tire  -> how much does pressure move QSX?
  across tires, same pressure bin  -> how much does geometry move QSX?

Also rules out the obvious confound: if P tracks F_z across segments, any
"pressure effect" could just be a load effect.
"""
import numpy as np
import fit_pipeline as fp
import tire_predict as tp

NAMES = ("QSX1", "QSX2", "QSX3")
# TTC Round 9 set points, roughly 8 / 10 / 12 / 14 psi
EDGES = np.array([0, 62, 76, 90, 1e9])
LABEL = ["~8psi", "~10psi", "~12psi", "~14psi"]


def columns(case, lat_full):
    F_z0 = lat_full[-1]
    R_0 = lat_full[-2] * 0.5 * 0.0254
    p, _ = tp._core(lat_full, "lateral")
    F_z = -case["FZ"].ravel()
    gamma = np.sin(case["IA"].ravel() * np.pi / 180)
    alpha = np.tan(case["SA"].ravel() * np.pi / 180)
    F_y, _, _ = tp._tm_lat(F_z, alpha, gamma, p, F_z0)
    A = np.column_stack([F_z * R_0, -F_z * R_0 * gamma,
                         F_z * R_0 * F_y / F_z0])
    return A, case["MX"].ravel(), F_z


def solve(A, y):
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = A @ beta - y
    return beta, np.sqrt(r @ r / len(y)), np.linalg.cond(A)


# ---- confound check: does pressure track load across segments? -------------
print("Confound check: corr(segment mean P, segment mean F_z)")
allP, allF = [], []
for sid, spec in fp.ALL_SPECS.items():
    if spec.family != "lateral":
        continue
    tire = sid[len("lat_")]
    for c in fp.build_cases_lateral(spec):
        P = c["P"].ravel(); P = P[P > 1.0]
        if P.size == 0:
            continue
        allP.append(P.mean()); allF.append(np.abs(c["FZ"]).mean())
allP, allF = np.array(allP), np.array(allF)
print(f"   n={len(allP)} segments   corr = {np.corrcoef(allP, allF)[0,1]:+.3f}")
print("   (near zero => the pressure effect is not a disguised load effect)\n")

# ---- pooled fit per (tire, pressure bin) ----------------------------------
grid = {}
for sid, spec in fp.ALL_SPECS.items():
    if spec.family != "lateral":
        continue
    tire = sid[len("lat_"):]
    lat_full = np.asarray(tp.PER_TIRE["lat_" + tire], float)
    buckets = {}
    for c in fp.build_cases_lateral(spec):
        P = c["P"].ravel(); Pp = P[P > 1.0]
        if Pp.size == 0:
            continue
        b = int(np.digitize(Pp.mean(), EDGES) - 1)
        A, y, Fz = columns(c, lat_full)
        keep = P > 1.0
        buckets.setdefault(b, []).append((A[keep], y[keep], Fz[keep]))
    for b, parts in sorted(buckets.items()):
        A = np.vstack([p[0] for p in parts])
        y = np.concatenate([p[1] for p in parts])
        Fz = np.concatenate([p[2] for p in parts])
        if len(y) < 2000 or np.ptp(Fz) < 200:      # need real load spread
            continue
        beta, rmse, cond = solve(A, y)
        grid[(tire, b)] = dict(beta=beta, rmse=rmse, cond=cond, n=len(y))

tires = sorted({k[0] for k in grid})
bins = sorted({k[1] for k in grid})

for j, nm in enumerate(NAMES):
    print("=" * 78)
    print(f"{nm}   rows = tire, cols = pressure bin")
    print("=" * 78)
    hdr = "".join(f"{LABEL[b]:>12s}" for b in bins)
    print(f"{'':18s}{hdr}")
    M = np.full((len(tires), len(bins)), np.nan)
    for i, t in enumerate(tires):
        cells = ""
        for k, b in enumerate(bins):
            g = grid.get((t, b))
            if g is None:
                cells += f"{'--':>12s}"
            else:
                M[i, k] = g["beta"][j]
                cells += f"{g['beta'][j]:12.5f}"
        print(f"{t:18s}{cells}")
    within_tire_across_P = np.nanmean(np.nanstd(M, axis=1, ddof=1))
    within_P_across_tire = np.nanmean(np.nanstd(M, axis=0, ddof=1))
    print(f"\n   spread across PRESSURE, holding tire  : {within_tire_across_P:.6f}")
    print(f"   spread across TIRE, holding pressure  : {within_P_across_tire:.6f}")
    ratio = within_tire_across_P / within_P_across_tire
    verdict = ("PRESSURE dominates" if ratio > 1.5 else
               "GEOMETRY dominates" if ratio < 0.67 else "comparable")
    print(f"   pressure / geometry = {ratio:6.2f}   -> {verdict}\n")

print("Fit quality of the binned fits:")
print(f"   cond(A)  {min(g['cond'] for g in grid.values()):.1f} - "
      f"{max(g['cond'] for g in grid.values()):.1f}")
print(f"   cells    {len(grid)} of {len(tires)*len(bins)} possible")
