# Tire Model

Pacejka Magic Formula fits to Calspan TTC Round 9 rig data (Project 2356,
January 2022), for FSAE vehicle dynamics and lap simulation.

## What Is Here

| File | Purpose |
|---|---|
| `magic.py` | Canonical physics and the per-spec fitting blocks. Source of truth. |
| `tire_model.py` | Standalone predictor. Parameter lists in it are stale, see below. |
| `data_comber.py` | Earlier data handling utilities. |
| `data/cornering_SI/` | Lateral rig runs, `.mat` signals paired with `.dat` metadata. |
| `data/straight_SI/` | Longitudinal and combined-slip rig runs, same pairing. |

## Model Structure

Three fitted layers, each a two-pass nonlinear least squares.

**Pass one** fits B, C, D, E, S_h, S_v independently per constant-load segment.
**Pass two** fits the low-level P or R parameters against all segments at once,
enforcing the load and camber dependence of those intermediates.

| Layer | Intermediates | Low-level params | Count |
|---|---|---|---|
| Lateral pure slip | B, C, D, E, S_hy, S_vy | PDY1-3, PCY1, PKY1-7, PHY1-2, PEY1-5, PVY1-4 | 22 |
| Longitudinal pure slip | B, C, D, E, S_hx, S_vx | PDX1-2, PCX1, PKX1-3, PHX1-2, PEX1-4, PVX1-2 | 14 |
| Combined G_x | B_gx, C_gx, E_gx, S_hgx | RBX1-3, RCX1, REX1-2, RHX1 | 7 |
| Combined G_y | B_gy, C_gy, E_gy, S_hgy, S_vgy | RBY1-4, RCY1, REY1-2, RHY1-2, RVY1-6 | 15 |

Aligning torque and the Q-parameter family are developed separately and are not
in this repo.

## Tire Specs And Their Runs

Spec names encode size, compound code, and rim width in inches.

### Lateral (from `cornering_SI`)

| Spec | Tire | Rim | Runs |
|---|---|---|---|
| `160X75_R20_70` | Hoosier 16.0X7.5-10, R20, C2000 | 7.0" | 4, 5, 6 |
| `160X75_R20_80` | Hoosier 16.0X7.5-10, R20, C2000 | 8.0" | 7, 8, 9 |
| `205X70_R20_70` | Hoosier 20.5X7.0-13, R20 | 7.0" | 16, 17, 18 |
| `205X70_R20_80` | Hoosier 20.5X7.0-13, R20 | 8.0" | 19, 20, 21 |
| `180X60_R20_60` | Hoosier 18.0X6.0-10, R20 | 6.0" | 27, 28, 29 |
| `180X60_R20_70` | Hoosier 18.0X6.0-10, R20 | 7.0" | 30, 31, 32 |

### Longitudinal And Combined (from `straight_SI`)

| Spec | Tire | Rim | Runs |
|---|---|---|---|
| `205X70_R20_70` | Hoosier 20.5X7.0-13, R20 | 7.0" | 50, 51, 52 |
| `205X70_R20_80` | Hoosier 20.5X7.0-13, R20 | 8.0" | 53, 54, 55 |
| `180X60_R20_60` | Hoosier 18.0X6.0-10, R20 | 6.0" | 68, 69, 70 |
| `180X60_R20_70` | Hoosier 18.0X6.0-10, R20 | 7.0" | 71, 72, 73 |

There are no `straight_SI` runs for 16.0X7.5-10, so longitudinal and combined
slip cover two tire sizes while lateral covers three.

Pure-slip longitudinal fits use segments where `mean(|SA|)` is near zero.
Combined-slip fits reuse the same raw files but select the segments where slip
angle is meaningfully nonzero.

Note that the file set used to compute `F_z0` for a spec is not always the same
as the file set that gets segmented into fitted cases. This is intentional.

## Data Conventions

From the `.mat` channels:

| Channel | Meaning | Conversion |
|---|---|---|
| `FZ` | Vertical load | Sign inverted in use: `F_z = -FZ` |
| `FY` | Lateral force | Sign inverted in use: `F_y = -FY` |
| `FX` | Longitudinal force | Used as-is |
| `SA` | Slip angle, degrees | `alpha = tan(SA * pi / 180)` |
| `SL` | Slip ratio | Dimensionless, used as-is |
| `IA` | Inclination (camber), degrees | `gamma = sin(IA * pi / 180)` |
| `ET` | Elapsed time, seconds | Used for sweep-duration filtering |

`F_z0` is the mean absolute load across a spec's designated raw files.

Getting these conversions wrong is silent, not loud. Nothing raises, the curve
just comes out wrong. Check them first when a fit looks strange.

Tire geometry is not in the `.mat` files. It comes from line one of the paired
`.dat` file, which carries `Tire_Name` in the form
`<OuterDiameter>X<Width>-<RimDiameter>` plus a separate `Rim_Width` field, both
in inches.

`R20` is a compound code and is constant across every run in this dataset, so
it carries no information. `C2000` appears only on the 16.0X7.5-10 runs, which
means compound is confounded with size on that one tire.

## Running A Fit

`magic.py` executes its fitting blocks at import time. Most are commented out
because they were originally run one spec at a time. Uncomment the block you
want, or use the config-driven pipeline once it exists.

The fits are not fast. Tolerances are set to `2.3e-16` with `max_nfev` at
`1e8`, and each spec runs a per-segment first pass before the second pass.

## Known Issues

- `tire_model.py`'s hardcoded parameter lists are the older 20-parameter
  lateral form and do not match what `magic.py` currently fits. Do not use them
  as a reference.
- Fitting bounds and segmentation thresholds were tuned by trial and error
  against the data actually received, not derived from spec sheets.
- Several parameters are weakly identified. See `DECISIONS.md`.
