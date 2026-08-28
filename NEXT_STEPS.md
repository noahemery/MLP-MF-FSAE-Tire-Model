# Next steps

Written 2026-08-28, after the config-driven pipeline, the NLLS baseline and the
first full neural sweep landed. Ranked by value over effort. See `DECISIONS.md`
for why things are the way they are, and its Log for what has already happened.

## Where we actually are

| family | held-out force RMSE | vs peak force | usable? |
|---|---|---|---|
| lateral | 89 N segment / **141 N unseen tire** | 3.3% / **5.1%** | yes, for cornering |
| gy | 353 N / 568 N | 16% / 26% | no |
| gx | 488 N / 432 N | 17% / 15% | no |
| longitudinal | 676 N / 638 N | **24%** | no |

Two results narrow the problem sharply:

**The network is not the bottleneck.** Evaluating the NLLS baseline parameters
on the same data gives lateral 116 N, longitudinal 679 N, G_x 537 N, G_y 427 N.
The network matches or beats `magic.py`'s own fits everywhere. Longitudinal's
~677 N is a floor *both* methods hit.

**Training knobs are exhausted.** Longitudinal moves 678.1 -> 677.4 -> 676.1 N
across lr = 1e-4, 1e-3, 1e-2, and capacity does nothing (677.3 vs 677.1 for
32x32 vs 64x64). It is not undertrained. The cause is upstream of training.

A camber hypothesis was tested and **rejected**: `tm_long` has no camber terms
and the data spans 0/2/4 degrees, but error at IA~0 is *worse* (745 N) than at
2 or 4 degrees (641 / 655 N).

---

## P0. Commit the work

Roughly 3,000 lines across 13 new files are untracked and nothing is committed.
Before anything else:

- Confirm `.gitignore` covers `.venv/`, `outputs/`, `*.bak` (it does).
- Do **not** add more raw TTC data; licensing is still unconfirmed
  (`DECISIONS.md` open questions).
- Commit as the repo owner only.

## P1. Longitudinal: investigate the 20.5X7.0-13 straight runs

**~half a day. The only cheap item likely to move a headline number.**

Error is not spread evenly across specs — it is concentrated in one tire:

| spec | RMSE | % of peak | source runs |
|---|---|---|---|
| long_205X70_R20_70 | 821 N | 27% | 51, 52 |
| long_205X70_R20_80 | 936 N | 30% | 54, 55 |
| long_180X60_R20_60 | 383 N | 14% | 69, 70 |
| long_180X60_R20_70 | 348 N | 14% | 72, 73 |

The fitted parameters are near-identical across all four (PDX1 2.2-2.4, PCX1
1.6-1.7, PKX1 50-60), so this is not a diverged fit. It points at the data or
the segmentation for runs 51/52/54/55.

Do:
1. Plot measured vs predicted FX per segment for run 51 against run 69, using
   the baseline parameters. Look for whole segments that are structurally
   wrong rather than uniformly noisy.
2. Check what `sort`/`bound` admit for those runs — segment count, row counts,
   load levels, slip-ratio range. The straight-file segmentation literals are
   identical across all 12 straight blocks, so if 51/52 need different ones,
   that is a finding for the owner, not something to change unilaterally.
3. Check whether those runs contain conditions the others do not: pressure
   sweeps, temperature drift, different speeds. The `.mat` files carry `P`,
   `V`, and four temperature channels that nothing currently reads.

Success: if 205X70 comes up to the 180X60 fits' 14%, pooled longitudinal drops
from 24% to ~14% and becomes arguably usable.

## P2. Prove decision 2 actually fixed identifiability

**~1 day. This is the intellectual payoff of the whole approach.**

`DECISIONS.md` decision 2 argues the two-stage NLLS objective is blind to the
shift parameters because its residuals span 0.075 to 3000, and that fitting
force directly removes that. The baseline measured the symptom (H-family
relative uncertainty 167, D-family 0.0022). Nobody has yet measured whether the
neural version cured it.

Do:
- Compute per-parameter uncertainty from the trained network the same way the
  baseline did — `residual_variance * pinv(J'J)`, with J the Jacobian of force
  residuals with respect to the recovered parameters (torch can produce this
  directly).
- Put it beside `outputs/baseline_param_uncertainty.parquet`, per family.
- If H and E uncertainty collapses, decision 2 is vindicated and worth writing
  up. If it does not, decision 2's reasoning is wrong and should be revised.

This also explains the mixed decision-3 result: the network made parameters
more consistent for lateral (15/22) and G_y (14/15), but *less* consistent for
longitudinal (3/14) and G_x (2/7). Consistency and identifiability should track
each other; if they do not, that is informative.

## P3. Get William's rulings (blocking)

Already written up in `DECISIONS.md`. Three block the differentiable
transcription and one changes results:

1. `tm_long:1063-1067` trailing commas making five variables 1-tuples.
2. `tm_lat:1027` missing the `1e-8` guard that `second_pass_y:296` has. Now on
   the training path, so it matters more than before.
3. `first_pass_GY:1379` multiplying a 6-tuple by an array.
4. **Which `E_y` is intended** — `second_pass_y:298` and `tm_lat:1029` use
   different `np.sign()` arguments. The force objective uses `tm_lat`'s;
   confirm that is the physics and not the accident.

`mf_torch.py` currently works around 1 and 3 and transcribes 2 as-is; all three
are documented in its module docstring.

## P4. G_y rank deficiency (blocked on P3)

Three of four G_y fits are rank-deficient — worst is `GY_180X60_R20_60` at
12/15 with condition 4.0e17, numerically singular. The same family also:

- reads camber from the `SA` channel instead of `IA` (`first_pass_GY:1364`),
- stops at `max_nfev=1e2` in all four blocks rather than converging,
- has the dead `G_x > 1` penalty branch (`:1381`).

These are plausibly one problem, not three. Do not touch until P3 comes back.

## P5. More tires (not a code problem)

Lateral has 3 tire sizes, longitudinal and combined have 2. `R20` is constant
across the entire dataset so it carries no information, and `C2000` appears
only on the 16.0X7.5-10, confounding compound with size.

The 5.1% unseen-tire lateral result is good *because* lateral has three sizes.
Two sizes is barely a slope. No training change fixes this; the options are
more tires or accepting that the model interpolates among tested tires rather
than predicting new ones. That framing is already written into `DECISIONS.md`
under Measured Constraints and should stay in any external claim.

## Cheap follow-ups, if time

- **Pressure as a feature.** The `.mat` files carry `P` and nothing reads it.
  `DECISIONS.md` calls this additive.
- **Deep sweep on one family.** `sweep.py --deep --families longitudinal` is
  ~4 h rather than the ~60 h for all four. Only worth it if P1 shows
  longitudinal is trainable at all; right now the evidence says the ceiling is
  not in training.
- **Pin the environment harder.** `requirements.txt` is now exact, but nothing
  verifies at runtime that the installed versions match. `mf_torch.py`'s gate
  catches physics drift, not version drift.

## Verification for each item

- **P1** — pooled longitudinal RMSE recomputed from
  `outputs/baseline_params.parquet`; success is 205X70 reaching ~14%.
- **P2** — a new parquet beside `baseline_param_uncertainty.parquet`, same
  estimator, compared per family.
- **P3/P4** — `mf_torch.py` gate must still pass at 1e-10 after any change,
  and `verify_acceptance.py` must still be bitwise if `magic.py` is touched.
- **Everything** — `audit_literals.py` must keep reporting all 18 blocks
  matching, and no result may be quoted as geometry generalisation unless it
  comes from the `spec` protocol.
