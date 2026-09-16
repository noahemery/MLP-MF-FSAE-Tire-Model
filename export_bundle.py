"""Assemble one bundle to hand to William: code, all parameters, all figures.

Covers both levels of parameter:

  LOW LEVEL   the per-segment first-pass coefficients. For MZ these are the 9
              (D_t, C_t, B_t, E_t, D_r, C_r, B_r, S_arm, S_ht) per segment, the
              direct input to second_pass_MZ. Shipped now.
  HIGH LEVEL  the second-pass P/R/Q vectors that describe how the low-level
              coefficients move with load and camber. Shipped for the four
              force families and MX; MZ's 36 Q-params are not fitted yet.

*** THE ONE THING THAT WILL BITE ***

second_pass_MZ indexes BCDE_params[0] .. [8] and then subscripts per segment
(D_t[0], D_t[i]). So it needs shape (9, n_segments).

Every other second pass takes (n_segments, 6) and indexes [:, 0].

So the MZ matrix is the TRANSPOSE of the layout the other families use. The
.npy files written here are already (9, n_seg) and can be passed straight in.
Hand it (n_seg, 9) instead and, above 9 segments, D_t[i] raises IndexError; at
or below 9 it silently fits nonsense.

  python export_bundle.py
"""
import os
import shutil
import zipfile

import numpy as np
import pandas as pd

import fit_pipeline as fp

OUT = os.path.join("outputs", "for_william")
MZ_ORDER = ("D_t", "C_t", "B_t", "E_t", "D_r", "C_r", "B_r", "S_arm", "S_ht")

# Fitting implementation plus the gates. magic.py is William's own file and is
# included because it carries the authorised fixes needed to reproduce any of
# this. The raw TTC data is deliberately NOT bundled -- it may be
# licence-restricted (see CLAUDE.md) and it is ~400 MB.
CODE = ("magic.py", "fit_pipeline.py", "run_baseline.py", "run_spec_worker.py",
        "progress.py", "audit_literals.py", "verify_mx.py", "export_params.py",
        "export_bundle.py", "tire_predict.py", "USAGE.md", "DECISIONS.md",
        "NEXT_STEPS.md", "requirements.txt")
ANALYSIS = ("mx_probe.py", "mx_variance.py", "mx_pressure.py",
            "mx_holdout.py", "mz_identifiability.py", "mz_first_pass.py")


def fresh(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path)


def copy_code():
    fresh(os.path.join(OUT, "code"))
    os.makedirs(os.path.join(OUT, "code", "analysis"), exist_ok=True)
    got = []
    for name in CODE:
        if os.path.exists(name):
            shutil.copy2(name, os.path.join(OUT, "code", name))
            got.append(name)
    for name in ANALYSIS:
        src = os.path.join("analysis", name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(OUT, "code", "analysis", name))
            got.append("analysis/" + name)
    return got


def high_level():
    """Second-pass vectors: the four force families plus MX."""
    d = os.path.join(OUT, "high_level_params")
    fresh(d)
    df = pd.read_parquet(os.path.join("outputs", "baseline_params.parquet"))
    df.to_csv(os.path.join(d, "all_params_long.csv"), index=False)

    wide = (df.pivot_table(index=["spec_id", "family"], columns="param_name",
                           values="value").reset_index())
    wide.to_csv(os.path.join(d, "all_params_wide.csv"), index=False)

    # Raw vectors in magic.py's own order, ready to pass to tm_lat/tm_long/GX/GY.
    nd = os.path.join(d, "npy")
    os.makedirs(nd, exist_ok=True)
    n = 0
    for sid in fp.ALL_SPECS:
        src = os.path.join("outputs", "specs", sid + ".npy")
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(nd, sid + ".npy"))
            n += 1

    for extra in ("baseline_diagnostics.parquet",
                  "baseline_param_uncertainty.parquet"):
        p = os.path.join("outputs", extra)
        if os.path.exists(p):
            pd.read_parquet(p).to_csv(
                os.path.join(d, extra.replace(".parquet", ".csv")), index=False)
    return df.spec_id.nunique(), n


def low_level():
    """MZ per-segment first-pass coefficients, shaped for second_pass_MZ."""
    d = os.path.join(OUT, "low_level_params_MZ")
    fresh(d)
    src = os.path.join("outputs", "mz_first_pass.parquet")
    if not os.path.exists(src):
        return {}, 0
    mz = pd.read_parquet(src).sort_values(["tire", "segment"])
    mz.to_csv(os.path.join(d, "mz_first_pass_all_segments.csv"), index=False)

    shapes = {}
    for tire, g in mz.groupby("tire"):
        g = g.sort_values("segment")
        # (9, n_seg): row j is parameter j across segments. See module docstring.
        M = np.vstack([g[p].to_numpy(dtype=float) for p in MZ_ORDER])
        np.save(os.path.join(d, "BCDE_params_%s.npy" % tire), M)
        g[["segment"] + list(MZ_ORDER)].to_csv(
            os.path.join(d, "BCDE_params_%s.csv" % tire), index=False)
        shapes[tire] = M.shape
    return shapes, len(mz)


def copy_figures():
    src = os.path.join("outputs", "figures")
    if not os.path.isdir(src):
        return 0
    d = os.path.join(OUT, "figures")
    fresh(d)
    n = 0
    for f in sorted(os.listdir(src)):
        if f.lower().endswith((".png", ".pdf", ".svg")):
            shutil.copy2(os.path.join(src, f), os.path.join(d, f))
            n += 1
    return n


README = """# Tire parameters and fitting code

Everything from the least-squares fits, in one place. Calspan TTC Round 9.

## Read this first: the MZ matrix is transposed

`second_pass_MZ` indexes `BCDE_params[0]` .. `[8]` and then subscripts per
segment (`D_t[0]`, `D_t[i]`), so it needs shape **(9, n_segments)**.

Every other second pass takes **(n_segments, 6)** and indexes `[:, 0]`.

`low_level_params_MZ/BCDE_params_<tire>.npy` is already `(9, n_seg)`:

    import numpy as np, magic
    BCDE = np.load("low_level_params_MZ/BCDE_params_205X70_R20_70.npy")
    lat  = np.load("high_level_params/npy/lat_205X70_R20_70.npy")
    lng  = np.load("high_level_params/npy/long_205X70_R20_70.npy")
    gx   = np.load("high_level_params/npy/GX_205X70_R20_70.npy")
    gy   = np.load("high_level_params/npy/GY_205X70_R20_70.npy")
    r = magic.second_pass_MZ(cases, lat[-1], 1, BCDE, x0_Q, lat, lng, gx, gy)

Passing `(n_seg, 9)` instead raises IndexError above 9 segments and silently
fits nonsense at or below 9.

## Layout

    code/                      fitting implementation, gates, analysis scripts
    high_level_params/         second-pass vectors: lateral, longitudinal,
                               G_x, G_y, MX. CSV plus npy in magic.py's order
    low_level_params_MZ/       MZ per-segment coefficients, 429 segments,
                               all 6 tires, ready for second_pass_MZ
    figures/                   plots

`high_level_params/npy/<spec>.npy` are the vectors `tm_lat`, `tm_long`, `GX`
and `GY` take directly. Pure-slip vectors are `[params..., diameter_in, F_z0]`
(lateral 24 long, longitudinal 16); the G families are bare parameters.

## What is fitted and what is not

| family | state |
|---|---|
| lateral, longitudinal, G_x, G_y | second pass done, 18 specs |
| MX (`MX_*`) | done, all 6 tires, exactly `fit_MX` |
| MX + pressure (`MXP_*`) | done, an EXTENSION, see below |
| MZ low level | done, 429 segments, all 6 tires |
| MZ high level (36 Q-params) | **not fitted** |
| MY | does not exist, no MY channel in the data |

## MZ: two things to decide

**`B_t` is set by its bound, not by the data.** It sits at the upper bound in
51% of segments (32-67% by tire), and sweeping the cap moved it from 180 to
2,397 with the residual flat at 13-16%. It lives inside `arctan(B_t*alpha_t)`,
which saturates. So treat the shipped `B_t` as conditional on a stated bound,
not measured. The second pass is deliberately not run yet, because fitting 36
Q-params on top would spread that through all of them. Preference: tie `B_t` to
the lateral stiffness rather than fit it free.

**Pneumatic trail looks right**, which matters more: `D_t` medians are
0.030-0.042 m and never hit a bound in any of the 429 segments.

Sensitivity, median of `||dr/dtheta|*|theta|/||r||` over all segments:

    E_t 82.3   C_t 36.0   B_t 4.05   D_t 2.56   D_r 1.88
    C_r 1.45   S_arm 0.44   B_r 0.35   S_ht 0.29

Trail shape terms dominate by one to two orders of magnitude. `B_r`, `S_ht` and
`S_arm` are weakest, so cut there first if the second pass needs fewer.

`S_arm` is 0 for both 160X75 specs, per your call to neglect F_x where there is
no data. That is exact, not approximate: with `S_arm = 0` the residual is
bit-identical (`0.0e+00`) whichever donor long/gx vectors are passed, which is
why MZ covers all 6 tires rather than 4.

## Accuracy

Held out, GroupKFold by segment:

    MX                10.1% of p95 abs MX   (8.2% with pressure terms)

In-sample fit diagnostics, NOT performance:

    lateral 3.2-4.7%   longitudinal 6.1-7.9%   G_x 8.2-8.8%   G_y 10.0-18.3%
    MZ low level 13.3-16.8% per tire

MZ has no held-out number because the part that would generalise is the second
pass, which is not run.

## Pressure

Nothing in the force fits reads the `P` channel, and every lateral spec pools
6-9 distinct inflation pressures spanning ~45-101 kPa. `corr(P, F_z)` is +0.028
across 429 segments, so it is an independent variable, not a disguised load
effect. Adding three linear pressure coefficients to MX cut held-out error from
10.1% to 8.2%, with `QSX3p` negative in all six tires and `QSX2p` positive in
all six. So the force residuals carry unmodelled pressure variance too.

`MXP_*` is that extension. It is held in `fit_pipeline`, NOT in `magic.py`,
since the physics in that file is yours. Awaiting your review.

## magic.py fixes in this bundle

16 authorised fixes. The one that matters for the moments: `fit_MX` and
`fit_MY` built residuals with `np.vstack` over arrays already `.squeeze()`d to
1-D, so vstack read them as rows and demanded equal segment lengths. `fit_MX`
raised for all 6 tires. Now `np.concatenate`. The second passes are unaffected
because they stack un-squeezed `(n,1)` columns.

## Reproducing

    python run_baseline.py          # all 30 specs
    python audit_literals.py        # every literal in all 18 blocks vs source
    python verify_mx.py             # MX closed form == fit_MX's own optimum
    python tire_predict.py          # standalone model == magic.py (0.000e+00)
    python analysis/mz_first_pass.py

`tire_predict.py` is standalone, numpy only, parameters embedded. It needs no
other file in this bundle.
"""


def main():
    os.makedirs(OUT, exist_ok=True)
    code = copy_code()
    n_specs, n_npy = high_level()
    shapes, n_seg = low_level()
    n_fig = copy_figures()

    with open(os.path.join(OUT, "README.md"), "w", encoding="utf-8",
              newline="\n") as fh:
        fh.write(README)

    zpath = os.path.join("outputs", "tire_params_for_william.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(OUT):
            for f in files:
                full = os.path.join(root, f)
                z.write(full, os.path.relpath(full, OUT))

    print("bundle: %s" % OUT)
    print("  code            %d files" % len(code))
    print("  high level      %d specs, %d npy vectors" % (n_specs, n_npy))
    print("  low level (MZ)  %d segments" % n_seg)
    for t in sorted(shapes):
        print("      %-16s BCDE_params shape %s" % (t, shapes[t]))
    print("  figures         %d" % n_fig)
    print("\nzip: %s  (%.1f MB)" % (zpath, os.path.getsize(zpath) / 1e6))


if __name__ == "__main__":
    main()
