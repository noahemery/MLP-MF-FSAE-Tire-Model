> **ARCHIVED.** This document records the earlier NLLS plus Gaussian Process
> pipeline, built against a **20-parameter** lateral Magic Formula and a
> refactored `magic/` package that no longer exists in the current repo.
>
> The current canonical physics is `magic.py`, which fits **22** lateral
> P-parameters and includes combined-slip G corrections. Two expressions differ
> structurally between the two versions (the camber divisor placement in
> `BCD_y`, and the operator on the `np.sign()` term inside `E_y`), so the
> identifiability results and cross-validation numbers below do **not** transfer
> numerically to the current form. The patterns they describe are still expected
> to hold and are carried forward in `DECISIONS.md` as inherited constraints.
>
> Historical record. Do not edit.

# Tire model: MF bug fixes + config refactor + GP residual layer

This doc explains what's changing in the tire fitting code and why, for anyone on the team who wants the reasoning behind the decisions below.

## Why touch this at all

`magic.py` fits the Pacejka Magic Formula (MF) to Calspan TTC Round 9 data per tire spec. Two things are true about it as it stood before this change:

1. It had two real correctness bugs in the second-pass P-parameter fit (not style issues — they changed the fitted numbers).
2. It produced no durable output — the six fitted parameter sets were computed and then discarded the moment the script exited. Nothing was saved, printed, or plotted.

Both needed addressing before anything else could be layered on top, GP included.

## Decisions, and the reasoning behind each

**1. Fix the two bugs, don't just leave them.**
- `second_pass_x`: the `i==0` block divided by `mu_x * F_z * PCX1` with no epsilon guard; the loop body a few lines later divided by the same thing `+ 1e-8`. Segment 0 got different divide-by-zero protection than every other segment.
- `second_pass_y`: the loop body added `S_vy_gamma` *after* subtracting from the residual instead of nesting it inside the model term like the `i==0` block correctly did. That meant every segment except the first was fit against a slightly wrong target value.
- Concrete impact: these biased the fitted P-parameters, which is the actual thing downstream lap-sim work consumes. Worth fixing regardless of anything else.

**2. Extract one shared formula function per direction (`_y_model_terms`, `_x_model_terms`) instead of patching the two spots separately.**
Both bugs above existed because the same B/C/D/E/S_h/S_v formula got hand-typed in more than one place (the `i==0` block vs. the loop body) and drifted apart over time. Patching both spots fixes the bug we found but leaves the same failure mode available for the next edit. One function, called from both the fit code and the predict code (`tm_lat`/`tm_long`), removes the class of bug, not just the instance.

**3. Persist the fit results.**
Previously `lat_160X75_R20_70` etc. vanished when the script exited. Before anything (GP included) can build on the MF fit, the fit itself needs to survive past one script run — P-params, `F_z0`, per-segment BCDE params, and the segmented `cases` are now saved to disk (`joblib`).

**4. Refactor to a config-driven pipeline, staged so it can't silently regress.**
The 2x (lateral) / 4x (longitudinal) tire-spec blocks were hand-duplicated ~60-line copies of each other, differing only in a handful of constants (file list, window/threshold values, ET-duration filter range, least-squares bounds). That duplication is *how* the two bugs above happened — the same logic re-typed enough times eventually drifts. Moving to one function driven by a `TireSpecConfig` per spec kills that pattern going forward and gives something to actually test.
- Risk: this touches hand-tuned segmentation thresholds that clearly took real trial-and-error to land on. To avoid silently changing what "160X75_R20_70" means, Stage 1's acceptance bar was: reproduce numerically identical `p_params`/`bcde_params` to the original script for all 6 specs, verified by diff, before anything else got trusted or built on top of it.

**5. Add the GP as a hybrid residual model, not a replacement.**
The MF is a fixed analytic curve shape. It gets the overall S-curve right but can't represent everything real rig data does — local asymmetries, peak-region deviations, load/camber effects the P-parameter interpolation smooths over. A GP fit on `residual = measured - MF_prediction`, as a function of `(F_z, slip, IA)`, picks up what's left over without touching the MF's own coefficients, which stay standard/portable for other tooling (and for anyone on the team who just wants the plain Pacejka numbers). It also gives something the MF can't: a calibrated uncertainty band, which is specifically relevant to the "rough around the edges when inclination angle is factored in" issue already flagged in the channel — the GP can pick up residual camber effects that the MF's camber terms don't fully capture.

**6. `sklearn.gaussian_process.GaussianProcessRegressor`, anisotropic RBF + `WhiteKernel`, `StandardScaler`'d features.**
`F_z` (~O(100-2000) N) and slip/`IA` (~O(±15°)) are on very different scales; without scaling and a per-feature length-scale, the GP effectively ignores whichever feature has the smallest raw magnitude. `WhiteKernel` models the actual noise floor explicitly so it isn't conflated with the numerical jitter term (`alpha`, kept small and fixed).

**7. Subsample each segment before pooling.**
Raw segments have thousands of highly autocorrelated points (up to ~127k-163k samples per raw file pre-segmentation). Exact GP training is O(n³); pooling everything is intractable and produces a near-singular kernel matrix from near-duplicate points regardless. Points are capped per segment before pooling across a tire spec.

**8. Validate with `GroupKFold` grouped by segment id, never a random row split.**
Adjacent raw points within one sweep are nearly identical. A random split leaks and produces a held-out RMSE that looks great for the wrong reason.

**9. Seed all randomness (`random_state`).**
For a team project, reproducibility matters as much as correctness — two people re-running the same fit should get the same numbers, not just "similar" ones.

**10. Fix `F_z0`'s sign convention.**
Found while wiring up prediction: raw `FZ` is negative under load, and every use of load inside the fit flips it positive (`F_z = -case["FZ"]`) — except the original `F_z0` computation, which averaged the raw, unflipped `FZ`. That means `df_z = (F_z - F_z0)/F_z0`, the normalized-load term nearly every P-parameter depends on, compared a positive `F_z` against a negative `F_z0`. Checked against real data: `df_z` averaged **-1.9** across the tested range with the original convention (never close to 0, even at the reference load) vs. **-0.09** once the signs match — the latter is what a normalized load deviation is supposed to look like. This affects the load-sensitivity terms (`PDY2`, `PKY2-5`, `PHY2`, `PVY2`, and the longitudinal equivalents) more than either of the two originally-scoped bugs, so it's fixed here too rather than left as-is.

## Known issues found but not fixed (flagged for the team, not resolved here)

- **Lateral second-pass fit is numerically fragile.** Even replaying the *original* (bug-preserving) formula through the refactored pipeline with identical inputs doesn't reproduce the original script's lateral P-params exactly (longitudinal reproduces exactly). The original lateral fits show poor convergence (`first-order optimality` well above zero), meaning the objective sits in a flat/ill-conditioned region where tiny floating-point differences move the optimizer to a different local point. This lines up with what's already been observed in `#modelling-lapsim` about the lateral model being rougher than the longitudinal one — it's a pre-existing fit-quality issue, not something introduced here.
- **`x_scale_jac` inconsistency**: only `160X75_R20_70` (lateral) has `x_scale="jac"` active in its second-pass `least_squares` call; every other spec has it commented out in the original script. Looks like a leftover from tuning one spec rather than a deliberate choice. Preserved per-spec as found.
- **Lateral `x0_P` has a spare, unused 21st element**: `second_pass_y` only names/uses 20 P-parameters, but the original `x0_P` list has 21 entries — `least_squares` was optimizing over an extra free parameter with zero effect on the residual. Preserved (only the first 20 are consumed) rather than silently dropped.
- **Fitting bounds appear to be hand-guessed, not derived from spec sheets.** Per the team, the least-squares bounds on B/C/D/E/S_h/S_v (and the ET-duration / threshold-factor segmentation constants) were tuned by trial and error against the data actually received, not against reference tire data — and one rim width/compound/aspect-ratio combination the team needed was never received from FSAE, so there's no ground truth to check those bounds against for that combination. Worth keeping in mind when judging fit quality on any tire spec derived from incomplete or substitute data.

## Error evaluation / risk register

| Risk | Cause | Mitigation |
|---|---|---|
| Cholesky/numerical failure fitting the GP | Near-duplicate points from autocorrelated raw segments | Per-segment subsampling cap, small fixed `alpha` jitter, `WhiteKernel` for real noise |
| GP effectively ignores slip/camber | Unscaled features, `F_z` dominates by magnitude | `StandardScaler` + anisotropic RBF (per-feature length-scale), log fitted length-scales as a sanity check |
| Misleadingly good held-out error | Random split leaks autocorrelated points across train/val | `GroupKFold` by segment id, always |
| GP silently "fixes" an upstream sign/formula bug instead of surfacing it | Residual target computed with a re-derived formula that diverges from what the MF was actually fit with | Residual target always computed via the shared `_y_model_terms`/`tm_lat` (or `_x`/`tm_long`) function — never a second hand-typed copy |
| GP produces a wiggly, non-physical correction that doesn't generalize | Length-scale bounds too permissive, GP interpolates through noise | Floored `length_scale_bounds`, only trust `GroupKFold` held-out RMSE, never in-sample error |
| Weak/overconfident GP for a thin tire spec | Too few accepted segments for that spec | Minimum-segment-count check; fall back to MF-only with a logged warning below threshold |
| Results differ between teammates re-running the same fit | Unseeded randomness in optimizer restarts / subsampling | Fixed `random_state` everywhere it's exposed |
| `ModuleNotFoundError` on someone else's machine | `scikit-learn`/`joblib` not previously a dependency anywhere in the project | Pinned in `requirements.txt` |
| Refactor silently changes what a tire spec's fit means | Moving hand-tuned segmentation thresholds into config | Stage 1 reproduces the original `p_params`/`bcde_params` exactly before Stage 2 (GP) begins |

## What actually changed (implementation summary)

**Stage 1 — config-driven refactor of the existing MF pipeline (equivalence-verified)**
- New package layout: `magic/segmentation.py` (sort/bound/bound2, moved as-is), `magic/pacejka.py` (first_pass_y/x, second_pass_y/x, tm_lat/tm_long — bugs fixed via shared `_y_model_terms`/`_x_model_terms`), `magic/config.py` (`TireSpecConfig` etc., encoding the original 6 hardcoded blocks), `magic/pipeline.py` (`fit_tire_spec_lateral`/`fit_tire_spec_longitudinal`), `magic/persistence.py` (save/load fitted results via `joblib`).
- `fit_all.py` replaces the top-level script: loops over configs, fits, saves.
- Renamed `shit` → `residual_terms` (it's the vector of per-segment fit errors `least_squares` actually minimizes).
- Acceptance check: numerically diffed `p_params`/`bcde_params` for all 6 tire specs against the original script's output.

**Stage 2 — GP residual layer (additive, only after Stage 1 passed)**
- New `magic/gp_residual.py`: `ResidualGPConfig`, `ResidualGP` (fit/predict), `build_gp_dataset(cases, fit_result, direction)`.
- Features `[F_z, slip, IA]`, `StandardScaler`, `ConstantKernel * RBF(anisotropic) + WhiteKernel`, `normalize_y=True`.
- `magic/predict.py`: `predict_fy`/`predict_fx` → `(MF + GP mean, GP std)`, while `TireFitResult.p_params` alone still exposes plain Pacejka coefficients.
- Fitted `ResidualGP`s persisted alongside the MF results.

## Verification performed

1. **Stage 1 equivalence.** Replayed the ORIGINAL (bug-preserving) second-pass formula through the refactored pipeline's cases/`bcde_params`/`F_z0`. Longitudinal: exact match (`diff = 0`) against the original script's output for all 4 specs — strong evidence the refactor's mechanics (segmentation, filtering, config values, fit loop) are correct. Lateral: small nonzero diffs even with the bug-preserving formula, traced to the original lateral fits not actually converging well (`first-order optimality` far from zero) — an ill-conditioned objective where tiny floating-point differences move the optimizer to a different point. Not a refactor bug; a pre-existing fragility in the lateral second pass, consistent with what's already been observed about the lateral model being rougher than longitudinal.
2. Confirmed persisted results (`models/mf_fits.joblib`, `models/tire_models.joblib`) exist and reload cleanly.
3. **Held-out cross-validation** (`GroupKFold`, grouped by segment so no sweep is split across train/val): MF-only RMSE vs. MF+GP RMSE, per spec —

   | Spec | Direction | MF-only RMSE | MF+GP RMSE | Improvement | 95% band coverage |
   |---|---|---|---|---|---|
   | `160X75_R20_70` | lateral | 803.9 | 190.4 | **+76.3%** | 94.9% |
   | `160X75_R20_80` | lateral | 872.1 | 105.8 | **+87.9%** | 68.3% (overconfident) |
   | `205X70_R20_70` | longitudinal | 877.3 | 834.3 | +4.9% | 89.4% |
   | `205X70_R20_80` | longitudinal | 971.5 | 988.5 | -1.7% | 88.2% |
   | `180X60_R20_60` | longitudinal | 387.4 | 402.2 | -3.8% | 97.3% |
   | `180X60_R20_70` | longitudinal | 380.8 | 378.8 | +0.5% | 97.0% |

   Pattern matches the physics: both lateral specs get large, genuine improvement (the MF's known weak point is exactly what the GP is designed to correct); all four longitudinal specs show ~0 or slightly negative improvement (the longitudinal MF is already sound, so the GP there is just fitting noise on held-out folds). **Decision: ship the GP for the 2 lateral specs only; keep the 4 longitudinal specs MF-only** rather than adding complexity that measurably doesn't help.
4. **GP length-scale collapse, found and partially fixed.** The first full fit (`length_scale_bounds` floor at `1e-2`) had 3 of 6 specs pin a length-scale (usually camber/IA) at that floor — visually confirmed as spiky, non-physical corrections chasing sample-to-sample noise, worst on `160X75_R20_70`. Raised the floor to `0.2` and refit those 3: `205X70_R20_80` fully resolved (length-scales now well within bounds, smooth curve); `180X60_R20_70` was visually fine even before (moot, and also excluded from shipping per the GP-scope decision above); `160X75_R20_70` improved (no more extreme spikes) but still pins its IA length-scale at the new floor.
5. **Round 2: diagnosed and fixed the amplitude bound + uncertainty calibration.** Inspecting the fitted kernels directly (correcting for `normalize_y=True`, which fits hyperparameters in normalized-residual space, not real Newtons) found two distinct problems:
   - `160X75_R20_70`'s kernel amplitude (`constant_value`) was pinned at `1e3`, the original bound — implying a real-world signal std of ~26,500 N, physically absurd (>10x the tire's whole force range). `constant_value_bounds` was never a meaningful bound to begin with: since `normalize_y=True` standardizes the target to unit variance first, an amplitude of even 1.0 already means "the GP explains 100% of the target's variance." Tightened to `(1e-3, 3.0)`. Refit: still pins at the new (much saner) ceiling, and IA length-scale still pins at its floor too — both point back to the same root cause found by diagnosing the MF fit directly (next bullet), not something bound-tuning alone resolves further.
   - **Root cause of `160X75_R20_70`'s roughness, found**: its Magic Formula second-pass fit has a **singular Jacobian at the solution** (condition number `inf`) and a first-order optimality of `~1.24e6` (should be ~0 at a real optimum) despite scipy reporting clean termination (`xtol` satisfied — it stopped because the step size shrank, not because it found a minimum). This means some combination of the ~20 P-parameters has no identifiable effect on the fit given this spec's data — the fit isn't wrong, it's non-unique, which explains both the original refactor-equivalence mismatch (point 1) and why the GP has real, non-noise structure to chase. This is an MF-level issue, not fixable from the GP side; flagged for the team, see below.
   - `160X75_R20_80`'s uncertainty band was overconfident (68.3% vs. 95% target) despite excellent point predictions — traced to `GroupKFold` holding out whole segments, which can carry systematic between-segment variation (rig drift, run-to-run differences) that a per-point `WhiteKernel` noise model structurally can't represent. Added post-hoc uncertainty calibration (`ResidualGP.std_scale`, `magic/gp_residual.py`'s `calibrate_std_scale()`): fits GPs on `GroupKFold` training folds, measures the RMS held-out z-score `(residual - gp_mean) / gp_std`, and scales predicted std by that factor at inference time. Re-verified on the same folds: `160X75_R20_80` coverage goes from 68.3% to **95.5%**; `160X75_R20_70` goes from 94.9% to 97.1% (already fine, now slightly conservative — the safer direction). Note this re-check reuses the folds the scale was derived from, so it confirms the mechanism is applied correctly rather than serving as fully independent validation — the underlying technique (variance/temperature scaling from held-out z-scores) is standard practice.
6. Reproduced the plot that was commented out at the bottom of the original `magic.py`: raw scatter, MF-only curve, MF+GP corrected curve, `mean ± 2*std` band, with the correct sign convention (`F_y = -data["FY"]`, `F_z = -data["FZ"]`). One per tire spec in `plots/`.
7. Added `scikit-learn`/`joblib` to `requirements.txt`.

## Parameter-level uncertainty on the P-parameters (in response to William's request)

William asked for a probability distribution over the fitted P-parameters themselves — different from the GP, which gives uncertainty on the *predicted force*, not on the parameters. Added `magic/param_uncertainty.py` (`param_covariance`) + `scripts/param_uncertainty.py`: computes the standard asymptotic covariance for nonlinear least squares, `cov ≈ residual_variance * pinv(J^T J)` at the fitted solution (same formula `scipy.optimize.curve_fit` uses internally). Cheap — reuses `magic/pipeline.py`'s new `refit_second_pass()` to rerun only the second-pass fit (~30s/spec, skips the expensive first pass), no GP involved. Uses `pinv`, not `inv`, since some specs are known singular/ill-conditioned — degrades gracefully (large/NaN std) instead of crashing.

**Correction to an earlier finding**: while implementing this, verified directly against the installed scipy 1.18.0 source (`check_x_scale` in `scipy/optimize/_lsq/least_squares.py`) that `x_scale=None` already resolves to `x_scale='jac'` whenever `method='lm'` — which every tire spec's second-pass fit uses. **The `x_scale_jac` per-spec inconsistency flagged earlier was a no-op, not a real inconsistency** — all 6 specs were always fitting with `x_scale='jac'` regardless of the flag. `magic/pipeline.py` now sets it unconditionally to make actual behavior explicit; `TireSpecConfig.x_scale_jac` is kept for backward compatibility only, no longer read.

This is a **linearized approximation** (assumes the residual surface is locally quadratic near the solution) — cheap, but not as trustworthy as bootstrapping (resample cases with replacement, refit repeatedly, look at the empirical spread) would be. Bootstrap is deferred, not done: measured cost is ~29s per single second-pass-only refit, so a reasonable 200-replicate bootstrap is ~9.5 hours across all 6 specs, or ~3.2 hours for just the 2 lateral specs — real compute, out of scope for a same-day pass.

**Two real, distinct reasons some specs come out singular/ill-conditioned, both handled explicitly rather than treated as a fitting failure**:
1. **Mechanical**: lateral's second-pass fit has a documented spare, unused 21st parameter (`x[20]`) — its Jacobian column is exactly zero by construction, which alone makes `J^T J` rank-deficient for both lateral specs. Dropped before computing (`drop_cols`), recorded in `ParamUncertainty.dropped_param_names`, never silently discarded.
2. **Physical, genuine non-identifiability** — see next section.

**Actual results, run against all 6 specs** (`models/param_uncertainty.joblib`) — a clean, consistent, generalizable pattern, not spec-specific noise:

| Spec | Rank (after dropping the spare) | Poorly-identified params (relative std > 50%) |
|---|---|---|
| `160X75_R20_70` | 17/20 | PCY1, PKY4, PHY1, PHY2, PEY2 |
| `160X75_R20_80` | 17/20 | PHY1, PHY2, PEY2 |
| `205X70_R20_70` | 13/14 | PDX2, PKX3, PHX1, PHX2, PEX1, PEX2, PEX3, PEX4 |
| `205X70_R20_80` | 13/14 | PHX1, PHX2, PEX1, PEX2, PEX3, PEX4 |
| `180X60_R20_60` | 13/14 | PKX2, PKX3, PHX1, PHX2, PEX1, PEX2, PEX3, PEX4 |
| `180X60_R20_70` | 13/14 | PHX1, PHX2, PEX1, PEX2, PEX3, PEX4 |

The **H-family (horizontal shift) and E-family (curvature) parameters are poorly identified in every single spec, both directions** — `PHX1`/`PHX2`/`PEX1-4` in all 4 longitudinal specs, `PHY1`/`PHY2`/`PEY2` in both lateral specs. The **D/K/V-family (peak grip, stiffness, vertical shift) parameters are consistently well identified** everywhere. This is the headline finding: it's not one bad spec, it's a systematic pattern across the whole dataset — likely means the test data doesn't have enough resolution right around the zero-slip transition region (where H and E terms are actually constrained) across any of the 6 specs. Both lateral specs are also missing one more identifiable direction than that pattern alone accounts for (rank 17/20, not 18/20) — `160X75_R20_70` uniquely also has `PCY1`/`PKY4` poorly identified, consistent with its already-documented extra fragility.

## Longitudinal GP, re-evaluated after the amplitude-bound/calibration fixes

The original "GP doesn't help longitudinal" decision (see the cross-validation table earlier in this doc) was measured *before* the `constant_value_bounds` fix and the `std_scale` calibration existed — worth checking whether it was stale. Refit all 4 longitudinal specs with the current (fixed) `ResidualGPConfig` and re-ran `GroupKFold` cross-validation:

| Spec | Improvement (original config) | Improvement (fixed config) |
|---|---|---|
| `205X70_R20_70` | +4.9% | +4.9% |
| `205X70_R20_80` | -1.7% | -1.8% |
| `180X60_R20_60` | -3.8% | -3.8% |
| `180X60_R20_70` | +0.5% | +0.5% |

Essentially unchanged, within rounding noise — the decision holds up, it wasn't stale. Worth noting for anyone revisiting this later: 3 of the 4 refits still pinned a length-scale at the `0.2` floor (same red flag as `160X75_R20_70`'s lateral fit) — `205X70_R20_80` was the one clean fit (all three length-scales comfortably within bounds). That the *accuracy* conclusion held anyway suggests the length-scale symptom isn't the actual reason GP fails to help here — more likely the longitudinal MF fit is just already accurate enough that there's genuinely little residual signal to correct, independent of any GP tuning. Longitudinal ships MF-only, confirmed a second time with the current code, not just inherited from an earlier decision.

**Two distinct failure modes, not one**, worth understanding separately rather than lumping together as "GP doesn't help":

| Spec | Pinned dimension | `constant_value` (amplitude) | Improvement |
|---|---|---|---|
| `205X70_R20_70` | F_z (floor) | 0.29 | +4.9% |
| `205X70_R20_80` | *none — clean fit* | 0.092 (very low) | -1.8% |
| `180X60_R20_60` | slip (floor) | 0.49 | -3.8% |
| `180X60_R20_70` | slip (floor) | 0.095 (very low) | +0.5% |

- **Pinned-length-scale specs** (`205X70_R20_70`, `180X60_R20_60`, `180X60_R20_70`): same symptom as `160X75_R20_70`'s lateral fit — the GP wants extreme local sensitivity along one axis, consistent with chasing sample-to-sample noise rather than real structure.
- **`205X70_R20_80` is the more interesting case**: it's the *only* one of the 4 with no pinning at all — every length-scale converged cleanly within bounds — and it's also the *worst* performer (-1.8%). Not a contradiction: its `constant_value` (amplitude) is small, meaning the GP itself correctly concluded there's very little real signal to explain. A well-behaved GP that honestly finds "nothing much here" will still slightly overfit its own training data by construction, and that doesn't generalize to held-out folds — hence the small negative. This isn't a broken fit; it's a *correct* fit reporting back "there's nothing to add," which is itself a useful sanity check that the GP is behaving sensibly rather than hallucinating structure.

## Follow-ups for the team (not resolved in this pass)

- **`160X75_R20_70`'s Magic Formula second-pass fit is non-identifiable** (singular Jacobian) — likely not enough distinct camber/load combinations in this spec's data to separately pin down every camber-related P-parameter. The GP still helps a lot despite this (76.3% held-out RMSE improvement) and now reports honestly wide uncertainty because of it, but the real fix is at the MF level (regularizing the second pass, or more camber-diverse data for this spec) — raised with William directly since it may connect to which parameters are identifiable in his fuller MF version. **Further evidence from the parameter-uncertainty work**: re-running just this spec's second-pass fit, warm-started from its own already-reported solution, with otherwise identical settings, lands on a meaningfully different parameter vector (`max|Δp_params| ≈ 2e5`) while barely changing the fit cost at all — the loss surface is flat enough in some direction that the optimizer drifts a long way for near-zero cost change. This is the same fragility already documented, now confirmed a second, independent way.
- Per the `#modelling-lapsim` discussion: `160X75_R20_70` (7" rim) is the lateral spec that matches the tire the team actually runs -- prioritize it over `160X75_R20_80` (8", substitute data) when judging real-world fit quality.
- Repo structure is still open: William's plan is a dedicated MF-fitting repo with his code (`tire_model.py`) on `main` and contributor branches off it. Whether this `magic/` package becomes that branch, or stays a parallel exploration, needs a team decision before pushing anywhere.
- Pressure isn't currently a regression input (only F_z, slip, camber). William's plan is to eventually fold pressure in alongside aspect ratio and rim size. The GP layer is already feature-based (`build_gp_dataset` -> `[F_z, slip, IA]`), so adding a 4th input dimension later is additive, not a rework.
