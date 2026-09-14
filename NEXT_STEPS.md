# Next steps

Updated 2026-08-28 after domain review. See `DECISIONS.md` for why things are
the way they are. Figures: `python plots.py` writes `outputs/figures/`.

## Domain rulings that changed the priorities

Review feedback, which reframes several findings as non-issues:

- **High uncertainty in shift (H) and curve (E) parameters is expected** and
  acceptable. Not a defect.
- **K at 1.44-3.77 relative uncertainty is acceptable.** Discrepancies there
  are attributable to the already-identified loose parameters.
- **Rank deficiency is benign for curve and offset parameters.** It means those
  parameters have no independent solution and must be expressed in terms of
  others. That is a property of the parameterisation, not a broken fit.
- **G factors are less critical.** They only adjust the pure lateral and
  longitudinal fits for combined loading.
- **The criterion that matters is whether the fits are suffering**, i.e. force
  error against measured data, not parameter uncertainty.

Consequence: the parameter-uncertainty work is deprioritised, and the question
narrows to which fits are actually bad.

## Fit quality, all 18 specs

Force RMSE from the NLLS baseline, as a percentage of 95th-percentile force.

| spec | family | % peak | verdict |
|---|---|---|---|
| lat_205X70_R20_80 | lateral | 3.2% | good |
| lat_180X60_R20_70 | lateral | 3.8% | good |
| lat_205X70_R20_70 | lateral | 4.1% | good |
| lat_160X75_R20_70 | lateral | 4.5% | good |
| lat_160X75_R20_80 | lateral | 4.5% | good |
| lat_180X60_R20_60 | lateral | 4.7% | good |
| GX_180X60_R20_70 | gx | 11.9% | marginal, low priority |
| GY_205X70_R20_70 | gy | 11.9% | marginal, low priority |
| GX_180X60_R20_60 | gx | 12.0% | marginal, low priority |
| GY_180X60_R20_70 | gy | 12.6% | marginal, low priority |
| long_180X60_R20_70 | longitudinal | 13.9% | marginal |
| long_180X60_R20_60 | longitudinal | 14.2% | marginal |
| GX_205X70_R20_80 | gx | 21.6% | bad, but a G factor |
| GX_205X70_R20_70 | gx | 22.1% | bad, but a G factor |
| GY_205X70_R20_80 | gy | 23.3% | bad, but a G factor |
| **long_205X70_R20_70** | **longitudinal** | **27.1%** | **THE PROBLEM** |
| **long_205X70_R20_80** | **longitudinal** | **29.7%** | **THE PROBLEM** |
| GY_180X60_R20_60 | gy | 36.9% | bad, but a G factor |

**All six lateral fits are good.** Every failure is straight-line data, and
five of six trace to runs 51/52/54/55, the 20.5X7.0-13 tire.

Given G factors are less critical, **the two `long_205X70` fits are the only
genuinely blocking failures.** They are pure longitudinal slip, they feed the
G_x layer beneath them, and at 27-30% they are not usable for braking or
acceleration.

---

## P1. Fix the two long_205X70 fits

**The whole job, essentially. Runs 51, 52, 54, 55.**

Evidence that this is data, not code:

- The same code, same segmentation literals and same parameter families give
  14% on runs 69/70/72/73 and 27-30% on 51/52/54/55.
- Fitted parameters are near-identical across all four longitudinal specs
  (PDX1 2.2-2.4, PCX1 1.6-1.7, PKX1 50-60), so no fit diverged.
- The same source runs also produce the two worst G_x fits, so three
  independent fitting paths fail on the same files.
- Learning rate and capacity change nothing: 678.1 -> 677.4 -> 676.1 N across
  lr 1e-4 to 1e-2. Not an optimisation problem.
- A camber hypothesis was tested and **rejected** — error at IA~0 is worse
  (745 N) than at 2 or 4 degrees (641 / 655 N).

Do, in order:

1. **Look at `outputs/figures/fig4_problem_runs.png`.** It puts
   `long_205X70_R20_70` (27%) beside `long_180X60_R20_60` (14%): measured vs
   modelled F_x against slip ratio, residual against load, and load coverage.
   Whatever is different should be visible.
2. **Check what segmentation admits from each run** — segment count, rows per
   segment, load levels, slip-ratio range. If runs 51/52 need different
   `sort`/`bound` literals than the other straight runs, that is a finding for
   the owner, not something to change unilaterally.
3. **Read the channels nothing currently uses.** The `.mat` files carry `P`
   (pressure), `V` (speed) and four temperature channels. If 51/52/54/55 were
   run at a different pressure or ran hotter, a camber-blind, pressure-blind
   `tm_long` cannot represent it and the residual is structural.

Success: `long_205X70` reaching the ~14% the other two longitudinal fits
achieve. That would make pooled longitudinal usable and lift G_x with it.

## P2. Rulings still needed from the owner

Three block the differentiable transcription, one changes results:

1. `tm_long:1063-1067` trailing commas making five variables 1-tuples.
2. `tm_lat:1027` missing the `1e-8` guard that `second_pass_y:296` has. This is
   now on the neural training path.
3. `first_pass_GY:1379` multiplying a 6-tuple by an array.
4. **Which `E_y` is intended** — `second_pass_y:298` and `tm_lat:1029` use
   different `np.sign()` arguments. The force objective uses `tm_lat`'s.

`mf_torch.py` works around 1 and 3 and transcribes 2 as-is; all are documented
in its module docstring.

## P3. G_y, if it becomes worth it

`GY_180X60_R20_60` is the worst fit at 36.9% and the worst rank deficiency
(12/15). Its source runs 69/70 produce fine longitudinal and G_x fits, so this
is specific to G_y. Two candidate causes are already known: camber read from
the `SA` channel instead of `IA` (`first_pass_GY:1364`), and all four G_y
second passes stopping at `max_nfev=1e2` rather than converging.

Low priority while G factors are considered less critical, but it is the one
place where a known defect and a bad fit coincide.

## P4. More tires

Lateral has 3 tire sizes, longitudinal and combined have 2. `R20` is constant
across the dataset so it carries no information, and `C2000` appears only on
the 16.0X7.5-10, confounding compound with size.

The 5.1% unseen-tire lateral result is good *because* lateral has three sizes.
Two sizes is barely a slope. No training change fixes this.

## Deprioritised

- **Parameter-uncertainty analysis.** Per the domain review, loose H, E and K
  are expected. Not worth a day.
- **Deep sweep.** Evidence says the ceiling is not in training.
- **Pressure as a feature.** Still additive, but read P1 step 3 first — it may
  turn out to be the cause rather than an enhancement.

## How to see the results

```
python plots.py           # writes outputs/figures/
```

| figure | question it answers |
|---|---|
| `fig1_tire_curves` | Does the fitted curve match the measured tire? |
| `fig2_fit_quality` | Which of the 18 fits are good, which are suffering? |
| `fig3_pred_vs_meas` | How tight is the model overall, per family? |
| `fig4_problem_runs` | What is different about the straight runs that fail? |
| `fig5_neural_vs_nlls` | Does the geometry network beat per-tire fitting? |

## Verification

- P1 — recompute per-spec fit quality; success is `long_205X70` at ~14%.
- P2/P3 — `mf_torch.py` must still pass its gate at 1e-10, and
  `verify_acceptance.py` must still be bitwise if `magic.py` is touched.
- Always — `audit_literals.py` reports all 18 blocks matching, and no result is
  quoted as geometry generalisation unless it comes from the `spec` protocol.
