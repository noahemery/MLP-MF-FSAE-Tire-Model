# Decisions

Live record for the neural parameter-estimation work. Why things are the way
they are, what has been verified, and what is still open.

For the earlier NLLS plus Gaussian Process pipeline, see
`docs/DECISIONS-mf-gp.md`. That work targeted a 20-parameter lateral formula
and its numbers do not transfer to the current 22-parameter form.

Rewritten 2026-08-27, after the config-driven pipeline and the NLLS baseline
landed. The previous version asserted an identifiability pattern since partly
disproven, and contained a contradiction between decision 1 and the Goal
diagram. Both are resolved below.

## Goal

Predict Magic Formula parameters from tire geometry, instead of fitting each
tire spec independently.

An MLP maps geometry to P or R parameters. The Magic Formula itself is
hardcoded from `magic.py` and has zero trainable parameters. Gradients flow
through the physics to the MLP weights.

```
geometry -> [MLP with bounded heads] -> P/R params
         -> magic.py expressions -> B,C,D,E,S_h,S_v
         -> force -> loss against measured FX/FY
```

Loss is measured **against force**, not against the first pass's per-segment
B,C,D,E. See decision 2.

The deliverable is recoverable P and R parameters, not force predictions. A
black-box force regressor would not hand off to lap sim and is not what this
is.

Two independent networks: one lateral, one longitudinal. Combined slip is what
the G factors handle. Aligning torque is out of scope.

Scope for the first version is **all 18 specs** — lateral, longitudinal, G_x
and G_y.

## Decisions

**1. Build on `magic.py`, do not reimplement it.**
`magic.py` is the reviewed, canonical physics. The neural framework transcribes
its expressions into a differentiable form and changes nothing about them. Sign
conventions in particular have been verified by the author and are not revised
here. Anything that looks wrong gets reported, not patched.

Scope note, because the previous version of this file was contradictory: this
decision governs **the expressions**, not the objective. The objective is
decision 2, and it deliberately differs from what `magic.py`'s second pass
optimizes. Transcribing `tm_lat` faithfully while choosing a different quantity
to minimise are compatible; the old text read as if it promised both would be
unchanged, which was never achievable.

**2. Loss is measured against force. (William, 2026-08-27)**
`magic.py` fits in two stages: a per-segment NLLS first pass produces
B,C,D,E,S_hy,S_vy, and `second_pass_y` then fits the 22 P-parameters to *those
coefficients* rather than to measured force. The neural framework instead runs
`tm_lat` / `tm_long` end to end and compares predicted force to measured FY/FX.

Two reasons, one structural and one measured.

Structural: it removes the non-differentiable first pass from the middle of the
pipeline, and trains on every measured row rather than on six summary numbers
per segment.

Measured: the two-stage objective sums residuals living on incompatible scales.
From the fitting bounds at `magic.py:646-652`, D ranges over 0..~3000 while
S_hy is confined to ±0.075. Least squares squares those, so a D error of 1000
contributes 1e6 to the cost while an S_hy error of 0.05 contributes 0.0025. The
optimizer is effectively blind to the shift parameters. This is not a
hypothesis — the baseline measured H-family relative uncertainty at 167 against
D-family at 0.0022 (see Measured Constraints). In force space every residual is
in newtons, and the shift parameters directly control where the curve crosses
zero, which the rig measured thousands of times.

Consequence to keep in mind: results are **not** directly comparable to the
NLLS baseline, because the two minimise different quantities. The baseline
remains the reference for parameter *consistency* (decision 3), not for loss
value.

**3. Joint training is the point, not just convenience.**
The alternative is fitting each spec by NLLS as today, then regressing the
resulting parameter vectors against geometry. That baseline now exists
(`outputs/baseline_params.parquet`). The argument for joint training is that
several P parameters are weakly identified, so each per-spec vector is an
arbitrary point on a flat region of the loss surface rather than a determined
value. Regressing geometry against arbitrary points fits noise. Shared MLP
weights cannot pick a different branch per tire, which acts as a regularizer
forcing a consistent choice across specs.

This is measurable, and it should be measured: are recovered parameters more
consistent across specs than the independent NLLS fits are?

**4. Bounds come from `magic.py`, enforced via squashed output heads.**
Every bound, window, threshold, and x0 seed in `magic.py` was hand-tuned
against this data. None are invented or borrowed from other implementations.
Networks cannot take hard box bounds, so bounds are enforced by scaling sigmoid
or softplus heads into the fitted range. Note the behavioral difference: NLLS
can sit exactly on a bound, a squashed head only approaches it asymptotically.

**5. GroupKFold by segment, always.**
Adjacent raw points within one sweep are near-duplicates. A random row split
leaks and produces held-out error that looks good for the wrong reason.
In-sample RMSE and R-squared are never reported as performance.

**6. Parquet for artifacts, not pickles.**
Fitted parameters should be readable without importing this code.

**7. The Gaussian Process layer stays parallel, undecided.**
On the earlier pipeline the GP improved lateral held-out RMSE by 76 to 88
percent and did essentially nothing for longitudinal. Whether that still holds
once a parameter decoder exists is an empirical question to test after the
decoder works, not a decision to make now. Do not entangle the two.

## Measured Constraints

Measured on the 22-parameter form from `outputs/baseline_*.parquet`,
2026-08-27. Replaces the previous "Inherited Constraints", whose numbers came
from the 20-parameter work and which asserted one pattern since disproven.

**Parameter identifiability.** Median relative standard error
(`std_err / |value|`) across the 6 lateral specs, from
`residual_variance * pinv(J'J)`:

| family | median rel. std err | verdict |
|---|---|---|
| D | 2.20e-03 | well identified |
| V | 2.29e-02 | well identified |
| K | 1.44e+00 | **marginal** |
| C | 3.61e+00 | marginal |
| E | 1.98e+01 | poorly identified |
| H | 1.67e+02 | poorly identified |

The earlier claim that "the D, K, and V families were consistently well
identified" is **wrong for K** at 22 parameters — its uncertainty exceeds its
own magnitude. D and V hold. The H and E half of the claim holds and is if
anything understated.

The mechanism is the scale imbalance described in decision 2, which is why
decision 2 is expected to change this picture rather than inherit it.

**Rank deficiency at the solution.** Four of 18 specs have Jacobians of less
than full rank, meaning some parameter combinations are unrecoverable from this
data rather than merely uncertain:

| spec | rank | condition |
|---|---|---|
| GY_180X60_R20_60 | 12/15 | 4.02e+17 |
| GY_180X60_R20_70 | 14/15 | 1.37e+15 |
| GY_205X70_R20_80 | 14/15 | 7.16e+12 |
| long_180X60_R20_70 | 13/14 | 2.94e+10 |

Full rank but ill-conditioned above 1e9: `lat_205X70_R20_80` (1.29e+10),
`lat_205X70_R20_70` (9.81e+09), `long_180X60_R20_60` (2.51e+09),
`lat_180X60_R20_70` (2.14e+09).

Three of the four rank-deficient specs are G_y, which is also the family
carrying a known channel defect and a 100-evaluation limit. Treat G_y as the
least trustworthy layer.

**Sample size.** The lateral network has 6 training examples across 3 tire
sizes; longitudinal, G_x and G_y have 4 each across 2 sizes. Geometry supplies
roughly three informative dimensions — section width, diameter, rim width.
`R20` is constant across the entire dataset and carries no information, and
`C2000` appears only on the 16.0X7.5-10.

This is not enough to generalise to an untested tire size, and a good held-out
number must not be read that way. The defensible claim is decision 3's: shared
weights force one consistent parameter choice across the tires that *were*
tested.

**Other.** Fitting bounds and segmentation thresholds are hand-tuned against
received data, not derived from spec sheets, and one rim width and compound
combination the team wanted was never received. `160X75_R20_70` is the lateral
spec matching the tire the team actually runs; it is also the best-conditioned
lateral fit (1.80e+06) and converged in the fewest evaluations. Prioritize it
when judging real-world fit quality.

## Known Defects In `magic.py`

Reported, not patched. These go to the repo owner. The first three block the
differentiable transcription; the rest affect results.

| Defect | Effect |
|---|---|
| `tm_long:1063-1067` trailing commas make `D_x`, `C_x`, `B_x`, `S_hx`, `E_x` 1-tuples | Numerically identical today — numpy coerces back. **torch will not.** Blocks transcription. |
| `tm_lat:1027` `B_y` denominator lacks the `1e-8` guard `second_pass_y:296` has | Fit and predict disagree near zero load. **Decision 2 puts `tm_lat` directly in the training path, so this matters more now, not less.** |
| `first_pass_GY:1379` multiplies a 6-tuple by an array | numpy coerces it and it lands on its feet by accident. Will not transcribe. |
| `np.sign()` argument inside `E_y` differs between `second_pass_y:298` and `tm_lat:1029` | `:298` subtracts the full `S_vy`; `:1029` subtracts only `S_vy_gamma`. Decision 2 settles *which the network uses* — `tm_lat`'s — but the two still disagree, and the owner should confirm `tm_lat:1029` is intended. |
| `first_pass_GY:1364` reads gamma from the `SA` channel, not `IA` | Every other function reads camber from `IA`. Changes numbers; all G_y results carry it. |
| `if (G_x.any() > 1)` always False (`:1102`, `:1381`) | `.any()` returns a bool, so the penalty branch never runs and `G_x` is unconstrained above 1. |
| `second_pass_x:736` omits the `1e-8` guard that `:769` has | Segment 0 is treated differently from every other segment in all 4 longitudinal fits. `docs/DECISIONS-mf-gp.md` records this same defect as fixed in the earlier pipeline. |
| `second_pass_GY:1439` uses `alpha = np.tan(...).mean()` where `:1415` keeps the full array | Segment 0 differs in both shape and value from all later segments. |
| `GX_180X60_R20_70:1349` hstacks an `F_z0` its block never defines | Leaks the previous block's value — the wrong spec. Affects only the stored 8th element, not the fit. Baseline stores bare `result.x` for all 4 G_x specs and records `fz0_used` separately. |
| All four G_y second passes use `max_nfev=int(1e+2)` | Every other second pass uses `1e+8`. All four G_y fits stop on this limit, not on convergence. |
| `magic.py:619` comment reads `Rim_Width=6.0` | Block loads runs 30/31/32 and assigns `lat_180X60_R20_70` (7.0"). Cosmetic. |

## Open Questions

- **Which `E_y` is intended.** `second_pass_y:298` and `tm_lat:1029` disagree.
  Decision 2 means the network uses `tm_lat`'s; the owner should confirm that
  is the correct physics rather than the accident.
- Compound is confounded with size on the 16.0X7.5-10 tire, the only one
  carrying a `C2000` designation. Any geometry effect learned for it may partly
  be compound.
- Combined-slip fits derive from `straight_SI`, which covers two tire sizes
  while lateral pure slip covers three. The G_y layer therefore has less
  geometry coverage than the lateral pure-slip layer.
- Pressure is not a regression input. Adding it later is additive. The `.mat`
  files carry a `P` channel that nothing currently reads.
- No transient model exists anywhere in the stack. Both the Magic Formula and
  the GP are steady-state, so nothing describes relaxation-length dynamics or
  drift across a stint. Possible future work, deliberately not in scope.
- The raw TTC data is committed to this repo. Licensing needs confirming.
- `requirements.txt` pins `scipy>=1.11`. scipy changed the `method='lm'`
  default `x_scale` from `1.0` to `'jac'` in **1.16.0**, so that floor permits
  two different solver behaviours. It should be pinned exactly.

## Environment Of Record

The baseline was produced on Python 3.14.5, numpy 2.5.2, scipy 1.18.1,
pandas 3.0.5, pyarrow 25.0.1. torch is not yet installed and has no verified
3.14 wheel.

## Log

### 2026-08-27 — Config-driven pipeline (Task 0) and NLLS baseline (Task 1)

`magic.py`'s 18 hand-run blocks are now driven from config in
`fit_pipeline.py`. `magic.py` itself changed only by prefixing `# ` to the 75
driver lines of its two live blocks so the module is importable; a diff check
confirms 0 lines changed by anything other than that prefix.

**Acceptance: bitwise.** Both previously-live specs reproduce the reference
captured from unmodified `magic.py` exactly — all 38 values, every delta
`0.000000e+00`. `audit_literals.py` independently parses `magic.py`'s source
text and confirms all 18 blocks' literals match the config, which is what
covers the 16 specs that have no reference fit.

All 18 specs fit in 31 min wall clock, 10 parallel workers, all `ok`.

**Correction to an earlier read.** The second passes were initially thought not
to converge. They do. `lat_180X60_R20_70` terminates on `ftol` at 27,108
residual evaluations (~22 min at full CPU). The apparent non-convergence was
Windows Power Throttling starving the process to ~1.6% CPU — at that rate the
same fit needs ~23 hours. `progress.keep_hot()` opts the process out via
`SetProcessInformation(ProcessPowerThrottling)`. The deterministic `CAP_NFEV`
guard never bound on any second pass, so the baseline is `magic.py`'s natural
result, not a capped one.

Convergence varies enormously by spec: `lat_160X75_R20_70` converged in 739
residual evaluations, `lat_180X60_R20_70` needed 27,108. That spread tracks
Jacobian conditioning.

**Artifacts.** `outputs/baseline_params.parquet` (286 rows, long format),
`baseline_diagnostics.parquet` (18), `baseline_param_uncertainty.parquet`
(276). No RMSE or R-squared is reported; `cost` and `optimality` are fit
diagnostics only.

**New tooling.** `fit_pipeline.py` (config + runners), `run_spec_worker.py`,
`run_baseline.py` (scheduler + assembly), `audit_literals.py` (source-text
drift check), `verify_acceptance.py` (bitwise gate), `progress.py`
(instrumentation, evaluation cap, power-throttling opt-out),
`capture_reference.py`, `comment_out_live_blocks.py`.
