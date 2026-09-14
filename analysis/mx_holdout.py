"""Held-out MX error, GroupKFold by segment, per CLAUDE.md -- in-sample numbers
are not a result.

Compares the model as fit_MX writes it (3 params) against a pressure-extended
version (6 params), which is what MF 6.x does and is still linear, so still a
closed-form solve:

    M_x = F_z*R_0*[ (QSX1 + QSX1p*dpi)
                   -(QSX2 + QSX2p*dpi)*gamma
                   +(QSX3 + QSX3p*dpi)*F_y/F_z0 ]      dpi = (P - P_nom)/P_nom

If the pressure terms cut held-out error, the answer to "do we need a PINN for
the moments" is no -- we need the physics to carry a pressure term.
"""
import numpy as np
import fit_pipeline as fp
import tire_predict as tp

P_NOM = 82.7           # kPa, ~12 psi, the modal TTC set point
NFOLD = 5


def build(spec, lat_full):
    F_z0 = lat_full[-1]
    R_0 = lat_full[-2] * 0.5 * 0.0254
    p, _ = tp._core(lat_full, "lateral")
    A3, Y, G, dpi = [], [], [], []
    for gi, c in enumerate(fp.build_cases_lateral(spec)):
        P = c["P"].ravel()
        keep = P > 1.0
        if keep.sum() < 50:
            continue
        F_z = -c["FZ"].ravel()[keep]
        gamma = np.sin(c["IA"].ravel()[keep] * np.pi / 180)
        alpha = np.tan(c["SA"].ravel()[keep] * np.pi / 180)
        F_y, _, _ = tp._tm_lat(F_z, alpha, gamma, p, F_z0)
        A3.append(np.column_stack([F_z * R_0, -F_z * R_0 * gamma,
                                   F_z * R_0 * F_y / F_z0]))
        Y.append(c["MX"].ravel()[keep])
        G.append(np.full(keep.sum(), gi))
        dpi.append((P[keep] - P_NOM) / P_NOM)
    return (np.vstack(A3), np.concatenate(Y), np.concatenate(G),
            np.concatenate(dpi))


def cv(A, y, groups):
    """GroupKFold by segment. Returns held-out RMSE."""
    uniq = np.unique(groups)
    rng = np.random.default_rng(0)
    order = rng.permutation(uniq)
    folds = np.array_split(order, NFOLD)
    se, n = 0.0, 0
    for f in folds:
        te = np.isin(groups, f)
        tr = ~te
        if tr.sum() < A.shape[1] * 10 or te.sum() == 0:
            continue
        beta, *_ = np.linalg.lstsq(A[tr], y[tr], rcond=None)
        r = A[te] @ beta - y[te]
        se += r @ r
        n += len(r)
    return np.sqrt(se / n)


print(f"{'tire':18s}{'p95|MX|':>9s}{'3-param':>10s}{'+pressure':>11s}"
      f"{'change':>9s}")
print("-" * 57)
rows = []
for sid, spec in fp.ALL_SPECS.items():
    if spec.family != "lateral":
        continue
    tire = sid[len("lat_"):]
    lat_full = np.asarray(tp.PER_TIRE["lat_" + tire], float)
    A3, y, g, dpi = build(spec, lat_full)
    A6 = np.column_stack([A3, A3 * dpi[:, None]])

    r3 = cv(A3, y, g)
    r6 = cv(A6, y, g)
    scale = np.percentile(np.abs(y), 95)
    rows.append((tire, scale, r3, r6))
    print(f"{tire:18s}{scale:9.2f}{100*r3/scale:9.1f}%{100*r6/scale:10.1f}%"
          f"{100*(r6-r3)/r3:+8.1f}%")

print()
a3 = np.mean([100 * r[2] / r[1] for r in rows])
a6 = np.mean([100 * r[3] / r[1] for r in rows])
print(f"mean held-out error   3-param {a3:.1f}%   +pressure {a6:.1f}%   "
      f"({100*(a6-a3)/a3:+.1f}%)")
print()
print("Pressure coefficients from the full 6-param fit (sign and consistency):")
for sid, spec in fp.ALL_SPECS.items():
    if spec.family != "lateral":
        continue
    tire = sid[len("lat_"):]
    lat_full = np.asarray(tp.PER_TIRE["lat_" + tire], float)
    A3, y, g, dpi = build(spec, lat_full)
    A6 = np.column_stack([A3, A3 * dpi[:, None]])
    b, *_ = np.linalg.lstsq(A6, y, rcond=None)
    print(f"   {tire:18s} QSX1p {b[3]:+.5f}  QSX2p {b[4]:+.5f}  "
          f"QSX3p {b[5]:+.5f}")
