# Refinement pass

Written 2026-09-14, after `tire_predict.py` shipped. Supersedes the earlier
version of this file, which was written before William's new `magic.py` landed
and before the `SL == 0` finding.

## Where things stand

**Shipped and in use:** `tire_predict.py` — standalone numpy module, parameters
embedded, verified bitwise identical to `magic.py` (0.000e+00). The team can
use it today.

**Fit quality, all 18 specs, one clean run:**

| family | range |
|---|---|
| lateral | 3.2 - 4.7% |
| longitudinal | 6.1 - 7.9% |
| G_x | 8.2 - 8.8% |
| G_y | 10.0 - 18.3% |

**The finding that shapes everything below:** geometry-based prediction does
not generalise from this dataset. Leave-one-tire-out, held-out RMSE on a tire
the model never saw:

| family | null (ignores geometry) | linear | mlp |
|---|---|---|---|
| lateral | **128** | 822 | 157 |
| longitudinal | 194 | **190** | 201 |
| gx | 204 | 200 | **150** |
| gy | **176** | 641 | 270 |

A model that ignores tire geometry entirely wins on lateral and G_y. With 6
tires across 3 sizes there is not enough spread to learn a size trend. The
shipped module therefore offers per-tire parameters and a pooled fallback,
not a geometry predictor.

---

## P1. Hierarchical (partial pooling) model

**The one idea likely to beat what we have. ~1 day.**

The null model wins because pooling helps. Per-tire NLLS wins where a tire has
rich data. Those are not opposites — they are the two ends of a single
spectrum, and nothing has tried the middle.

Fit per-tire parameter vectors `theta_i` directly (no geometry input), with a
penalty pulling them toward their shared mean:

```
loss = sum_i  force_error(theta_i)  +  lambda * sum_i ||theta_i - theta_bar||^2
```

- `lambda -> 0` reproduces per-tire NLLS
- `lambda -> infinity` reproduces the null model
- the optimum is expected in between, and `lambda` is chosen by
  leave-one-tire-out

Why this should work: a well-measured tire keeps its own parameters, a
poorly-constrained one borrows from the pool. That is exactly the failure mode
we measured — the linear model scored 822 N on lateral against the null's 128
because it had no shrinkage at all and extrapolated wildly off three points.

**Implementation.** Reuses the existing training loop. Replace `ParamNet`'s
geometry input with a free embedding, one vector per tire, initialised at the
pooled mean, plus the pooling penalty. Scale the penalty per parameter by the
baseline spread already computed in `train.baseline_stats`, otherwise `PKY1`
(~4.8e3) dominates `PHY1` (~1e-3).

**Verification.** Leave-one-tire-out against both existing baselines. Success
is beating `min(null, per-tire NLLS)` on at least two families. If the best
`lambda` comes out at the extreme, that is also a result — it says the two
regimes do not blend and the current approach is already the right one.

**Optional extension, only if the above works:** add geometry back as a
correction on top of the shared mean, `theta_i = theta_bar + f(geom_i)`. Do
not start here; geometry has already been shown not to carry.

## P2. Moments — MX then MZ

**~1 day, but the two halves are very unequal. William is waiting on this.**

All four moment functions are fixed and callable. **None of them is called
anywhere in `magic.py`** — there is no fitting block for any moment, and no x0
seed or bound exists for any of them. Every other family has both in its block.
These have to be invented, and that should be flagged as mine, not William's.

**Read the function names carefully before starting.** They do not mean what
they say:

| function | actually fits | params | structure |
|---|---|---|---|
| `fit_MX` | overturning moment MX | 3 (QSX1-3) | single stage, all segments at once |
| `first_pass_MX` | **MZ**, not MX — it reads `data["MZ"]` | 9 per segment | first pass |
| `second_pass_MZ` | MZ | 36 Q-params | second pass |
| `fit_MY` | rolling resistance | 2 | **dead**, no `MY` channel |

**P2a. MX first — this is a genuinely cheap win.** `fit_MX` is linear in its
three parameters:

```
M_x = F_z * R_0 * (QSX1 - QSX2 * gamma + QSX3 * F_y / F_z0)
```

No `arctan`, no shape factors, no two-stage structure, no segment loop to
orchestrate. Standard MF starting values (`QSX1` ~ 0, `QSX2` ~ 1, `QSX3` ~ 0.01)
should be close, and because the residual is linear in `x` the Jacobian is
constant — none of the flat-region convergence trouble that plagues the force
second passes can occur here. Expect this in under an hour, with a real error
number at the end of it.

**P2b. MZ second — this is the hard one.** Nine parameters per segment, then 36
Q-params, with `np.sign()` inside both `alpha_t` and `alpha_r`. That is the same
construct already suspected of wrecking the lateral second pass's Jacobian under
finite differences, and here it appears twice. Budget generously and expect the
seeds to need several attempts.

Seeds to start from: pneumatic trail `D_t` ~ 0.03 m, shape factors `C_t`/`C_r`
~ 1.5, residual torque `D_r` scaled off measured peak MZ, `S_arm` and `S_ht`
~ 0.

**Coverage limit — both moments, not just MZ.** `fit_MX` needs `lat_params` and
`gy_params`; MZ needs all four force families. G_y is fitted from straight runs,
so both moments are limited to the same 4 of 6 tires: `180X60_R20_60`,
`180X60_R20_70`, `205X70_R20_70`, `205X70_R20_80`. Both `160X75` specs are
cornering-only — and that includes the tire the team actually runs. Say this
plainly when handing the numbers over; a moment model that excludes the car's
own tire needs to be labelled as such.

**Steps:**
1. MX block: case selection, seeds, bounds, run all 4 specs, report error
2. Extend `audit_literals.py` to cover the new block before moving on
3. MZ first pass per segment, then the 36-parameter second pass
4. Error quantification for both, in the same normalised form as the force
   families so the numbers sit in one table
5. Extend `tire_predict.py`, against whichever interface Session 1 settled on.
   Moments are the reason the interface question has to be answered first

**Also worth raising with William:** `first_pass_MX` fitting MZ is confusing
enough to cause a real mistake later. Ask whether it can be renamed
`first_pass_MZ`. That is a rename, not a physics change, so it needs his
approval but not a judgement call.

## P3. Pressure as a feature

**~half a day, genuinely unexploited.**

The `.mat` files carry a `P` channel that **nothing currently reads**. Later
Pacejka versions have explicit pressure terms.

**Check before building:** does `P` vary meaningfully within and across runs?
If it was held constant there is no signal and this stops immediately. If it
drifted, it is unmodelled variance currently being absorbed as fit error.

This will not help geometry prediction. It could reduce per-tire error, which
is what the shipped model actually uses.

## P4. Optimisation quality

**~half a day, uncertain payoff.**

The second passes terminate on `ftol` in flat regions — `lat_180X60_R20_70`
improved cost by 0.021% across 27,108 evaluations, and that is with tolerances
at 2.3e-16, below double epsilon. Multi-start from several seeds might find
better optima, particularly on G_y.

Cheap to test, and a null result is still worth recording.

## P5. Revisit the G-correction penalty

The penalty William authorised is a measured trade, not a clean win:

- `GY_180X60_R20_60` went from rank 12/15 with an **infinite** condition number
  to full rank 15/15
- two other G_y specs dropped 15/15 to 14/15, condition numbers rose 3-4 orders
- force error rose ~10%
- G_x is completely unaffected; the penalty never fires there

Worth revisiting once the hierarchical model exists, since shrinkage may make
the penalty unnecessary.

---

## The deliverable: `tire_predict.py` v2

Everything above is only worth doing if it lands in the module the team
imports. What v2 adds over what shipped today:

| addition | comes from | if it fails |
|---|---|---|
| `Tire.aligning_moment(...)` / `r.M_z` from `Tire.combined` | P2 | omitted, no other item depends on it |
| `Tire.overturning_moment(...)` | P2 | omitted |
| better per-tire parameters for weak specs | P1, P3, P4 | keep v1 values |
| `param_uncertainty(tire, family)` returning per-parameter std | already computed | keep, it is free |
| improved pooled fallback | P1 | keep v1 null-model fallback |

### The API question, settled

An earlier draft of this plan froze v1's interface outright. That was too
strong, for two reasons.

**First, the freeze is barely binding.** Of the five refinement items, only P3
touches a signature at all, and it does so with a defaulted keyword
(`pressure_kpa=<nominal test pressure>`) that reproduces v1 behaviour exactly
when omitted. P1, P4 and P5 change parameter *values*, not shapes. P2 and
`param_uncertainty` add new functions. Better numbers through the same five
functions is not an API change.

**Second, there is one genuine reason to break it, and it falls inside this
pass.** `combined_force` returns a bare tuple `(F_x, F_y)`. Once moments exist,
combined output naturally wants to carry `F_x, F_y, M_z, M_x` — which under the
current shape means returning a 4-tuple and silently breaking every
`fx, fy = combined_force(...)` already written. Holding v1's shape here would
force a worse design, not merely a smaller one.

**So: redesign is allowed, but it happens before P2, not after.** Adoption is
currently near zero — nothing in this repo imports `tire_predict`, and the only
copies are the two files sent to the team on 2026-09-14. That makes now the
cheapest moment this will ever be.

Order of operations:

1. Ask the team whether anyone has written lap-sim code against v1 yet
2. **If no:** change the interface outright, in Session 1, before moments land
3. **If yes:** keep the five v1 functions as three-line deprecation wrappers
   over the new design. ~15 lines, and no constraint on the new design
4. **Do not** ship a second module alongside the first. Two files both claiming
   to be the tire model means two sets of embedded literals to keep verified
   against `magic.py`, and they will drift

The intended v2 shape, resolving the tire once rather than per call:

```python
t = tp.Tire("205X70_R20_70")
fy = t.lateral_force(F_z, slip_angle_deg)
r  = t.combined(F_z, slip_angle_deg, slip_ratio)   # r.F_x, r.F_y, r.M_z
```

This drops the `tire=None, allow_fallback=False` pair from every signature,
moves the unknown-tire check and the low-confidence warning to construction
time instead of once per loop iteration, and gives combined output named fields
so moments can be added later without breaking callers. The lap sim calls these
in a tight loop, so hoisting the dict lookup out of it is a minor speed win as
well.

What does **not** change under any option: the parameters stay embedded as
literals, the module stays numpy-only with no `magic.py` import, and
`python tire_predict.py` still has to report 0.000e+00.

The `param_uncertainty` item is worth calling out: the asymptotic covariance
`residual_variance * pinv(J'J)` is already computed and recorded per spec. It
is not exposed. Surfacing it costs nearly nothing and tells the team which
parameters to distrust, which is more useful than a smaller error number.

## Sequencing

Four working sessions. Each ends with something committed and verified, so
stopping after any one of them leaves the repo in a shippable state.

**Session 1 — cheap checks first, before committing to anything expensive**
- Ask the team whether anyone has written code against v1 yet, then settle the
  interface per the section above. This has to happen before P2, because moments
  are what force the `combined_force` return shape to change
- Read the `P` channel, plot its distribution per run. Decide P3 go / no-go in
  the first 20 minutes
- Multi-start test (P4) on the three weakest G_y specs only, not all 18
- Send William the six questions below
- Commit whatever measurements come out, even null results

**Session 2 — the hierarchical model (P1)**
- Implement the embedding + pooling penalty in `train.py`
- Sweep `lambda` over a log grid under leave-one-tire-out
- Compare against both baselines on all four families
- Decision gate: if it does not beat `min(null, NLLS)` on at least two
  families, stop here and record the negative result. Do not iterate on it

**Session 3 — moments (P2)**
- MX first. It is 3 linear parameters and should be done inside an hour, which
  means the session produces a result even if everything after it stalls
- Then MZ: seeds, bounds, first pass, 36-parameter second pass
- Extend `audit_literals.py` to cover both new blocks
- This is the session most likely to overrun. MX is nearly free; MZ is
  unfitted work with invented seeds and `np.sign()` in the residual

**Session 4 — export and hand off**
- Regenerate all 18 specs in one clean run from the winning configuration
- Rebuild `tire_predict.py` v2, re-verify at 0.000e+00
- Update `USAGE.md` with moments, units and the new accuracy table
- Send the team the same two files as before

**Ordering rationale.** P1 before P2 because P1 might change every parameter
the module embeds, and rebuilding the export twice is wasted work. P3 and P4
first because they are cheap and their answers are inputs to P1.

## Decision gates

Stop and report rather than pushing through, at each of these:

- **P3, after 20 minutes** — if `P` is constant, abandon it entirely
- **P1, after the `lambda` sweep** — if the optimum sits at either extreme, the
  answer is that shrinkage does not blend these regimes. Record it and move on
- **P2, if seeds will not converge** — moments are genuinely unfitted work, not
  a refactor. If physically-motivated seeds do not produce sane pneumatic
  trail, this needs William, not more optimiser tuning
- **Any regression in the four force families** — v1 numbers are the floor.
  A change that improves moments and degrades lateral does not ship

## Open questions for William

1. **Which `E_y` is intended** — `second_pass_y:298` and `tm_lat:1029` use
   different `np.sign()` arguments. Asked three times, never answered. Affects
   every lateral fit, which is our best family.
2. **The penalty threshold** — he said penalise when G exceeds *zero*;
   implemented as `> 1` because the G corrections are positive by construction
   so `> 0` penalises every point. Never confirmed.
3. **Moment starting values** — none exist in his file, for MX or MZ. Confirm
   mine are acceptable once written, or supply his own.
4. **The dead `x[8]`** removed from `first_pass_MX` — `S_vy` was overwritten by
   the `tm_lat` unpacking before use. Confirm it was not meant to do something.
5. **Moments covering only 4 of 6 tires** — both MX and MZ need G_y, which the
   two cornering-only `160X75` specs do not have. Acceptable, given that is the
   tire the car runs?
6. **Rename `first_pass_MX` to `first_pass_MZ`** — it reads `data["MZ"]` and
   fits pneumatic trail. A rename only, no physics touched.

## Not doing, and why

- **More seeds, folds, or training steps** — measured repeatedly; the
  conclusions are already unambiguous and compute is not the constraint
- **Bigger networks** — the MLP already loses to a constant on half the
  families
- **Huber loss** — tested at William's suggestion; a wash (354 vs 371 on G_y,
  identical elsewhere). His instinct about outliers was right, but they were
  the `SL == 0` samples, and excluding those directly already captured the
  benefit
- **Transient / relaxation-length modelling** — nothing in the stack is
  anything but steady-state. Real future work, deliberately out of scope

## What would help most but cannot be done here

**More tire sizes.** Three sizes is precisely why geometry prediction fails.
Worth raising before the next TTC round, along with **wider slip-angle sweeps
on the straight runs** — the narrow -6 to -3 degree window is what limits
`GY_180X60_R20_60` to 18.3% and leaves its parameters unidentifiable.

## Verification standards — carry these forward

- `python tire_predict.py` must report 0.000e+00 against `magic.py`
- `python mf_torch.py` must pass at 1e-10
- `python audit_literals.py` must report all 18 blocks matching source
- Held-out metrics only, GroupKFold by segment. Never quote in-sample RMSE
- `spec` protocol (leave-one-tire-out) is the only protocol that speaks to
  geometry generalisation. Never quote `segment` numbers as evidence for it
