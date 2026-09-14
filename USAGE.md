# Tire model — usage

Standalone Pacejka tire model fitted to Calspan TTC Round 9 data.
Needs only numpy.

```python
import tire_predict as tp

tp.available_tires()
# ['16.0X7.5-10 R20 7.0', '16.0X7.5-10 R20 8.0', '18.0X6.0-10 R20 6.0',
#  '18.0X6.0-10 R20 7.0', '20.5X7.0-13 R20 7.0', '20.5X7.0-13 R20 8.0']

# pure lateral slip
F_y = tp.lateral_force(F_z=800, slip_angle_deg=5, camber_deg=0,
                       tire="16.0X7.5-10 R20 7.0")          # 1765 N

# pure longitudinal slip
F_x = tp.longitudinal_force(F_z=800, slip_ratio=0.1,
                            tire="20.5X7.0-13 R20 7.0")

# combined slip (applies the G corrections)
F_x, F_y = tp.combined_force(F_z=800, slip_angle_deg=3, slip_ratio=0.1,
                             camber_deg=0, tire="18.0X6.0-10 R20 6.0")

# overturning moment (pure slip — takes no slip ratio)
M_x = tp.overturning_moment(F_z=800, slip_angle_deg=5, camber_deg=2,
                            tire="20.5X7.0-13 R20 7.0")      # 11.9 N·m

# ...and with inflation pressure, which matters a lot for M_x
M_x = tp.overturning_moment(F_z=800, slip_angle_deg=5, camber_deg=2,
                            tire="20.5X7.0-13 R20 7.0",
                            pressure_kpa=55)                 # 18.3 N·m at 8 psi

# raw parameters, e.g. to hand to another tool
tp.get_params("16.0X7.5-10 R20 7.0", family="lateral")
tp.get_params("16.0X7.5-10 R20 7.0", family="mx")
```

All inputs accept numpy arrays, so you can vectorise over a whole lap.

`overturning_moment` is available for **all six tires**, including the two
16.0X7.5-10 specs that have no combined-slip data.

## Units and signs

| | |
|---|---|
| `F_z` | newtons, **positive in compression** — pass load as a positive number |
| `slip_angle_deg`, `camber_deg` | degrees |
| `slip_ratio` | dimensionless |
| returns | newtons, following `magic.py`'s conventions |

## Accuracy

RMSE as a percentage of 95th-percentile force, measured on the data each tire
was fitted to:

| family | range | verdict |
|---|---|---|
| lateral | 3.2 – 4.7% | usable |
| longitudinal | 6.1 – 7.9% | usable |
| G_x (combined) | 8.2 – 8.8% | usable |
| G_y (combined) | 10.0 – 18.3% | weakest |

`M_x` is scored differently, and better: **held-out** error, GroupKFold by
segment, so it is not measured on the data it was fitted to. The force
families above are in-sample fit quality and are not directly comparable.

| M_x | held-out, % of p95 abs M_x |
|---|---|
| without `pressure_kpa` | 10.1% mean (9.2 – 10.9) |
| with `pressure_kpa` | 8.2% mean (5.5 – 10.8) |

Inflation pressure matters more for `M_x` than tire size does. The
lateral-force coefficient falls about 35% from 8 to 14 psi, consistently in
every tire measured, so pass `pressure_kpa` if you know it. Omitting it gives
the model as `magic.py` defines it, valid near the ~83 kPa (12 psi) test
pressure.

There is no `M_z` yet, and no `M_y` at all — the TTC files carry no MY channel.

Per tire, worst first:

| spec | family | % of peak |
|---|---|---|
| GY_180X60_R20_60 | gy | 18.3% |
| GY_180X60_R20_70 | gy | 11.1% |
| GY_205X70_R20_70 | gy | 10.3% |
| GY_205X70_R20_80 | gy | 10.0% |
| GX_180X60_R20_70 | gx | 8.8% |
| GX_180X60_R20_60 | gx | 8.6% |
| GX_205X70_R20_80 | gx | 8.3% |
| GX_205X70_R20_70 | gx | 8.2% |
| long_180X60_R20_60 | longitudinal | 7.9% |
| long_180X60_R20_70 | longitudinal | 7.0% |
| long_205X70_R20_70 | longitudinal | 6.2% |
| long_205X70_R20_80 | longitudinal | 6.1% |
| lat_180X60_R20_60 | lateral | 4.7% |
| lat_160X75_R20_80 | lateral | 4.5% |
| lat_160X75_R20_70 | lateral | 4.5% |
| lat_205X70_R20_70 | lateral | 4.1% |
| lat_180X60_R20_70 | lateral | 3.8% |
| lat_205X70_R20_80 | lateral | 3.2% |

## Limits — read before trusting a number

**Untested tires.** Asking for a tire that was not measured **raises**. A pooled
fallback exists (`allow_fallback=True`) but it is not validated for a size
outside the dataset.

Leave-one-tire-out testing showed geometry-based prediction does not
generalise here: on lateral and G_y, a model ignoring geometry entirely beat
one using it (128 N vs 822 N on lateral). With 6 tires across 3 sizes there is
not enough spread to learn a size trend. The fallback is a best-available
guess, not a prediction.

**Two tires have no combined-slip data.** Both `16.0X7.5-10` specs were only
run on the cornering rig, so `combined_force` raises for them. Pure lateral
works.

**`18.0X6.0-10 R20 6.0` G_y is weak.** Its straight-line runs only swept slip
angle from −6° to −3°, so those parameters are poorly constrained. Calling
`combined_force` on it emits a warning.

**Steady-state only.** No transient or relaxation-length behaviour, no
temperature or pressure dependence.

## Files

| file | contents |
|---|---|
| `tire_predict.py` | the model, parameters embedded |
| `outputs/tire_parameters.csv` | per-tire parameters, wide format |
| `outputs/pooled_fallback.csv` | pooled fallback parameters |
| `outputs/baseline_params.parquet` | same, long format |
| `outputs/baseline_diagnostics.parquet` | fit diagnostics per spec |

## Verification

```
python tire_predict.py
```

Checks the embedded force functions reproduce `magic.py` exactly. Currently
`0.000e+00` relative difference. Run it after any change.

```
python verify_mx.py
```

The equivalent gate for `M_x`. `run_mx` solves a linear system instead of
iterating on `magic.fit_MX`, because that residual is linear in its three
parameters. This proves the substitution is sound, per tire: `magic.fit_MX`'s
own residual at the solution matches the linear system (~1e-14), and driving
`least_squares` on `magic.fit_MX` from a neutral seed lands on the same
parameters (~1e-13). All 6 tires pass.

Supersedes `tire_model.py`, whose parameters are the stale 20-parameter form.
