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

# raw parameters, e.g. to hand to another tool
tp.get_params("16.0X7.5-10 R20 7.0", family="lateral")
```

All inputs accept numpy arrays, so you can vectorise over a whole lap.

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

Checks the embedded functions reproduce `magic.py` exactly. Currently
`0.000e+00` relative difference. Run it after any change.

Supersedes `tire_model.py`, whose parameters are the stale 20-parameter form.
