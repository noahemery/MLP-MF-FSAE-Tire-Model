# Decisions

Live record for the neural parameter-estimation work. Why things are the way
they are, what has been verified, and what is still open.

For the earlier NLLS plus Gaussian Process pipeline, see
`docs/DECISIONS-mf-gp.md`. That work targeted a 20-parameter lateral formula
and its numbers do not transfer to the current 22-parameter form.

## Goal

Predict Magic Formula parameters from tire geometry, instead of fitting each
tire spec independently.

An MLP maps geometry to P or R parameters. The Magic Formula itself is
hardcoded from `magic.py` and has zero trainable parameters. Gradients flow
through the physics to the MLP weights.

```
geometry -> [MLP with bounded heads] -> P/R params
         -> magic.py expressions -> B,C,D,E,S_h,S_v -> force -> loss
```

The deliverable is recoverable P and R parameters, not force predictions. A
black-box force regressor would not hand off to lap sim and is not what this
is.

Two independent networks: one lateral, one longitudinal. Combined slip is what
the G factors handle. Aligning torque is out of scope.

## Decisions

**1. Build on `magic.py`, do not reimplement it.**
`magic.py` is the reviewed, canonical physics. The neural framework transcribes
its expressions into a differentiable form and changes nothing about them. Sign
conventions in particular have been verified by the author and are not revised
here. Anything that looks wrong gets reported, not patched.

**2. Joint training is the point, not just convenience.**
The alternative is fitting each spec by NLLS as today, then regressing the
resulting parameter vectors against geometry. That baseline is cheap and worth
building first. The argument for joint training is that several P parameters
are weakly identified, so each per-spec vector is an arbitrary point on a flat
region of the loss surface rather than a determined value. Regressing geometry
against arbitrary points fits noise. Shared MLP weights cannot pick a different
branch per tire, which acts as a regularizer forcing a consistent choice across
specs.

This is measurable, and it should be measured: are recovered parameters more
consistent across specs than independent NLLS fits are?

**3. Bounds come from `magic.py`, enforced via squashed output heads.**
Every bound, window, threshold, and x0 seed in `magic.py` was hand-tuned
against this data. None are invented or borrowed from other implementations.
Networks cannot take hard box bounds, so bounds are enforced by scaling
sigmoid or softplus heads into the fitted range. Note the behavioral
difference: NLLS can sit exactly on a bound, a squashed head only approaches
it asymptotically.

**4. GroupKFold by segment, always.**
Adjacent raw points within one sweep are near-duplicates. A random row split
leaks and produces held-out error that looks good for the wrong reason.
In-sample RMSE and R-squared are never reported as performance.

**5. Parquet for artifacts, not pickles.**
Fitted parameters should be readable without importing this code.

**6. The Gaussian Process layer stays parallel, undecided.**
On the earlier pipeline the GP improved lateral held-out RMSE by 76 to 88
percent and did essentially nothing for longitudinal. Whether that still holds
once a parameter decoder exists is an empirical question to test after the
decoder works, not a decision to make now. Do not entangle the two.

## Inherited Constraints

Carried forward from the earlier work. These were measured against the
20-parameter lateral form, so the numbers need re-confirming, but the patterns
are expected to persist.

- The H-family (horizontal shift) and E-family (curvature) parameters were
  poorly identified in every spec, in both directions. The D, K, and V families
  were consistently well identified.
- At least one spec had a singular Jacobian at the solution, with first-order
  optimality far from zero despite scipy reporting clean termination. Warm
  restarting it from its own solution moved the parameter vector by roughly
  2e5 for near-zero change in cost.
- Fitting bounds and segmentation thresholds are hand-tuned against received
  data, not derived from spec sheets, and one rim width and compound
  combination the team wanted was never received.
- `160X75_R20_70` is the lateral spec matching the tire the team actually runs.
  Prioritize it when judging real-world fit quality.

## Known Defects In `magic.py`

Reported, not patched. These go to the repo owner.

| Defect | Effect |
|---|---|
| `tm_long` has trailing commas making `D_x`, `C_x`, `B_x`, `S_hx`, `E_x` 1-tuples | Numerically identical, shape `(1,N)` instead of `(N,)`. Numpy converts back silently, torch will not. |
| `first_pass_GY` reads gamma from the `SA` channel, not `IA` | Every other function reads camber from `IA`. |
| `tm_lat`'s `B_y` denominator lacks the `1e-8` guard that `second_pass_y` has | Fit and predict disagree near zero load. |
| `np.sign()` argument inside `E_y` differs between `second_pass_y` and `tm_lat` | Fit and predict use different expressions. |
| `if (G_x.any() > 1)` is always False | `.any()` returns a bool, so the penalty branch never runs and `G_x` is unconstrained. |

## Open Questions

- Compound is confounded with size on the 16.0X7.5-10 tire, the only one
  carrying a `C2000` designation. Any geometry effect learned for it may partly
  be compound.
- Combined-slip fits derive from `straight_SI`, which covers two tire sizes
  while lateral pure slip covers three. The G_y layer therefore has less
  geometry coverage than the lateral pure-slip layer.
- Pressure is not a regression input. Adding it later is additive.
- No transient model exists anywhere in the stack. Both the Magic Formula and
  the GP are steady-state, so nothing describes relaxation-length dynamics or
  drift across a stint. Possible future work, deliberately not in scope.
- The raw TTC data is committed to this repo. Licensing needs confirming.

## Log

_Add dated entries as work lands._
