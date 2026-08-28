"""The Magic Formula, transcribed from magic.py into differentiable torch.

Line-for-line from magic.py's tm_lat (:989-1034) and tm_long (:1037-1072).
Nothing is reordered, no sign is changed, no epsilon is added or removed
beyond the two notes below. verify_mf_torch() checks these against magic.py's
own numpy functions and is the gate on this file.

Two deliberate deviations, both forced and both already reported to the owner
in DECISIONS.md:

1. tm_long at magic.py:1063-1067 has trailing commas that make D_x, C_x, B_x,
   S_hx and E_x into 1-element tuples. numpy silently coerces them back, so
   the numbers are unaffected; torch raises. The commas are dropped here.
   This changes no value -- verify_mf_torch() proves it.

2. tm_lat:1027 computes B_y without the `+ 1e-8` guard that
   second_pass_y:296 has on the same expression. That is transcribed AS IS,
   guard absent, because it is magic.py's behaviour and CLAUDE.md forbids
   adding a flag that changes it. It is safe in practice here: F_z = -FZ is
   the applied load, order 1e3 N across every segment, so the denominator is
   never near zero for this data. If a NaN ever appears in training, this is
   the first place to look and it is evidence for the defect, not a licence
   to patch it.

Everything runs in float64. float32 would lose the comparison against numpy
and these parameters span ~7 orders of magnitude (PKY1 ~ 4.8e3 vs PHY1 ~ 1e-3).
"""

import numpy as np
import torch

DTYPE = torch.float64


def tm_lat(F_z, alpha, gamma, lambda_mu_y, x):
    """magic.py:989-1034. Returns (Y, BCD_y, D_y, S_hy, S_vy, mu_y).

    F_z, alpha, gamma : tensors, broadcastable
    x                 : tensor of 23 -- 22 P-params then F_z0
    """
    PDY1, PDY2, PDY3 = x[0], x[1], x[2]
    PCY1 = x[3]
    PKY1, PKY2, PKY3, PKY4, PKY5, PKY6, PKY7 = (x[4], x[5], x[6], x[7],
                                                x[8], x[9], x[10])
    PHY1, PHY2 = x[11], x[12]
    PEY1, PEY2, PEY3, PEY4, PEY5 = x[13], x[14], x[15], x[16], x[17]
    PVY1, PVY2, PVY3, PVY4 = x[18], x[19], x[20], x[21]
    F_z0 = x[-1]

    df_z = (F_z - F_z0) / F_z0

    mu_y = (PDY1 + PDY2 * df_z) * lambda_mu_y / (1 + PDY3 * gamma ** 2)
    BCD_y = (PKY1 * F_z0
             * torch.sin(PKY4 * torch.atan(
                 F_z / ((PKY2 + PKY5 * gamma ** 2) * F_z0)))
             / (1 + PKY3 * gamma ** 2))
    S_vy_gamma = F_z * (PVY3 + PVY4 * df_z) * gamma
    K_y_gamma_0 = F_z * (PKY6 + PKY7 * df_z)

    D_y = (mu_y * F_z)
    C_y = PCY1
    B_y = BCD_y / (mu_y * F_z * PCY1)                     # no guard: see note 2
    S_hy = ((PHY1 + PHY2 * df_z)
            + (K_y_gamma_0 * gamma - S_vy_gamma) / (BCD_y + 1e-8))
    E_y = ((PEY1 + PEY2 * df_z)
           * (1 + PEY5 * gamma ** 2
              - (PEY3 + PEY4 * gamma) * torch.sign(alpha + S_hy)))
    S_vy = ((PVY1 + PVY2 * df_z) * F_z + S_vy_gamma)

    u = B_y * (alpha + S_hy)
    Y = D_y * torch.sin(C_y * torch.atan(u - E_y * (u - torch.atan(u)))) + S_vy
    return Y, BCD_y, D_y, S_hy, S_vy, mu_y


def tm_long(F_z, s, lambda_mu_x, x):
    """magic.py:1037-1072, trailing commas dropped (note 1). Returns Y."""
    PDX1, PDX2 = x[0], x[1]
    PCX1 = x[2]
    PKX1, PKX2, PKX3 = x[3], x[4], x[5]
    PHX1, PHX2 = x[6], x[7]
    PEX1, PEX2, PEX3, PEX4 = x[8], x[9], x[10], x[11]
    PVX1, PVX2 = x[12], x[13]
    F_z0 = x[-1]

    df_z = (F_z - F_z0) / F_z0

    mu_x = (PDX1 + PDX2 * df_z) * lambda_mu_x
    BCD_x = F_z * (PKX1 + PKX2 * df_z) * torch.exp(PKX3 * df_z)

    D_x = (mu_x * F_z)
    C_x = PCX1
    B_x = BCD_x / (mu_x * F_z * PCX1)
    S_hx = (PHX1 + PHX2 * df_z)
    E_x = ((PEX1 + PEX2 * df_z + PEX3 * df_z ** 2)
           * (1 - PEX4 * torch.sign(s + S_hx)))
    S_vx = (F_z * (PVX1 + PVX2 * df_z))

    u = B_x * (s + S_hx)
    Y = D_x * torch.sin(C_x * torch.atan(u - E_x * (u - torch.atan(u)))) + S_vx
    return Y


def GX(F_z, F_z0, s, alpha, gamma, x):
    """magic.py:1655-1681. Longitudinal combined-slip correction factor."""
    RBX1, RBX2, RBX3 = x[0], x[1], x[2]
    RCX1 = x[3]
    REX1, REX2 = x[4], x[5]
    RHX1 = x[6]

    df_z = (F_z - F_z0) / F_z0

    B_gx = (RBX1 + RBX3 * gamma ** 2) * torch.cos(torch.atan(RBX2 * s))
    C_gx = RCX1
    E_gx = REX1 + REX2 * df_z
    S_hgx = RHX1

    u0 = B_gx * S_hgx
    G_x0 = torch.cos(C_gx * torch.atan(u0 - E_gx * (u0 - torch.atan(u0))))
    u = B_gx * (alpha + S_hgx)
    G_x = torch.cos(C_gx * torch.atan(u - E_gx * (u - torch.atan(u)))) / G_x0
    return G_x


def GY(F_z, F_z0, s, alpha, gamma, x, lat_params):
    """magic.py:1684-1721. Returns (G_y, S_vgy).

    Note magic.py names the returned factor G_x inside this function; it is
    the lateral correction. Naming preserved in behaviour, not in spelling.
    """
    RBY1, RBY2, RBY3, RBY4 = x[0], x[1], x[2], x[3]
    RCY1 = x[4]
    REY1, REY2 = x[5], x[6]
    RHY1, RHY2 = x[7], x[8]
    RVY1, RVY2, RVY3, RVY4, RVY5, RVY6 = (x[9], x[10], x[11], x[12],
                                          x[13], x[14])

    df_z = (F_z - F_z0) / F_z0

    _, _, _, _, _, mu_y = tm_lat(F_z, alpha, gamma, 1, lat_params)

    B_gy = ((RBY1 + RBY4 * gamma ** 2)
            * torch.cos(torch.atan(RBY2 * (alpha - RBY3))))
    C_gy = RCY1
    E_gy = REY1 + REY2 * df_z
    S_hgy = RHY1 + RHY2 * df_z

    D_vgy = (mu_y * F_z * (RVY1 + RVY2 * df_z + RVY3 * gamma)
             * torch.cos(torch.atan(RVY4 * alpha)))
    S_vgy = D_vgy * torch.sin(RVY5 * torch.atan(RVY6 * s))

    u0 = B_gy * S_hgy
    G_y0 = torch.cos(C_gy * torch.atan(u0 - E_gy * (u0 - torch.atan(u0))))
    u = B_gy * (s + S_hgy)
    G_y = torch.cos(C_gy * torch.atan(u - E_gy * (u - torch.atan(u)))) / G_y0
    return G_y, S_vgy


# ---------------------------------------------------------------------------
# The gate: these torch functions must reproduce magic.py's numpy ones.
# ---------------------------------------------------------------------------

def verify_mf_torch(n=4000, seed=0, verbose=True):
    """Compare against magic.py's numpy originals on realistic inputs.

    Importing magic is safe now that all 18 blocks are commented out.
    Returns a dict of max absolute and relative differences.
    """
    import magic

    rng = np.random.default_rng(seed)
    # Realistic ranges: loads 200-2000 N, slip +-15 deg, camber 0-4 deg.
    F_z = rng.uniform(200.0, 2000.0, n)
    alpha = np.tan(rng.uniform(-15.0, 15.0, n) * np.pi / 180.0)
    gamma = np.sin(rng.uniform(0.0, 4.0, n) * np.pi / 180.0)
    s = rng.uniform(-0.2, 0.2, n)
    F_z0 = 750.0

    lat = np.concatenate([rng.uniform(-1.0, 1.0, 22), [F_z0]])
    lat[3] = 1.4            # PCY1, keep the shape factor physical
    lat[4] = 4000.0         # PKY1
    lat[5] = 0.5            # PKY2
    lat[0] = 2.5            # PDY1
    lng = np.concatenate([rng.uniform(-1.0, 1.0, 14), [F_z0]])
    lng[2] = 1.5            # PCX1
    lng[0] = 2.5            # PDX1
    lng[3] = 12.0           # PKX1
    gx = rng.uniform(-1.0, 1.0, 7)
    gx[3] = 1.5             # RCX1
    gy = rng.uniform(-1.0, 1.0, 15)
    gy[4] = 1.4             # RCY1

    T = lambda a: torch.as_tensor(a, dtype=DTYPE)
    out = {}

    def cmp(name, got_np, got_t):
        got_np = np.asarray(got_np, dtype=np.float64).ravel()
        got_t = got_t.detach().numpy().ravel()
        absd = float(np.max(np.abs(got_np - got_t)))
        scale = np.maximum(np.abs(got_np), 1e-30)
        reld = float(np.max(np.abs(got_np - got_t) / scale))
        out[name] = {"max_abs": absd, "max_rel": reld}
        if verbose:
            print("  %-14s max_abs=%.3e  max_rel=%.3e" % (name, absd, reld))

    n_lat = magic.tm_lat(F_z, alpha, gamma, 1, lat)
    t_lat = tm_lat(T(F_z), T(alpha), T(gamma), 1, T(lat))
    for label, a, b in zip(("tm_lat.Y", "tm_lat.BCD_y", "tm_lat.D_y",
                            "tm_lat.S_hy", "tm_lat.S_vy", "tm_lat.mu_y"),
                           n_lat, t_lat):
        cmp(label, a, b)

    cmp("tm_long.Y",
        np.asarray(magic.tm_long(F_z, s, 1, lng)).ravel(),
        tm_long(T(F_z), T(s), 1, T(lng)))

    cmp("GX", magic.GX(F_z, F_z0, s, alpha, gamma, gx),
        GX(T(F_z), T(F_z0), T(s), T(alpha), T(gamma), T(gx)))

    n_gy = magic.GY(F_z, F_z0, s, alpha, gamma, gy, lat)
    t_gy = GY(T(F_z), T(F_z0), T(s), T(alpha), T(gamma), T(gy), T(lat))
    cmp("GY.G_y", n_gy[0], t_gy[0])
    cmp("GY.S_vgy", n_gy[1], t_gy[1])

    return out


if __name__ == "__main__":
    import sys
    print("verifying mf_torch against magic.py's numpy functions\n")
    res = verify_mf_torch()
    worst = max(v["max_rel"] for v in res.values())
    print("\nworst relative difference: %.3e" % worst)
    TOL = 1e-10
    if worst < TOL:
        print("PASS -- transcription matches magic.py within %g" % TOL)
        sys.exit(0)
    print("FAIL -- transcription does NOT match magic.py")
    sys.exit(1)
