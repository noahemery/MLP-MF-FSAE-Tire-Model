"""Train an MLP that maps tire geometry to Magic Formula parameters.

Structure (DECISIONS.md Goal):

    geometry -> MLP -> P/R params -> mf_torch expressions -> force -> loss

The Magic Formula holds zero trainable parameters. All learning is in the MLP,
and gradients reach it only by flowing back through the physics.

Loss is measured against measured force (decision 2), not against the first
pass's per-segment B,C,D,E.

Two evaluation protocols, both reported, because they answer different
questions:

  segment  -- GroupKFold holding out whole segments, pooled across specs.
              Required by decision 5. Answers "does this predict force on a
              sweep it never saw, for a tire it has seen?"
  spec     -- leave-one-spec-out. Answers the actual goal: "does this predict
              parameters for a tire size it never saw?" With 6 lateral specs
              this is a brutal test and is expected to be poor. It is reported
              anyway because a good `segment` number must not be mistaken for
              this one.

Output scaling, not bounds. magic.py's second pass uses method='lm', which is
unbounded, so there are no P-parameter bounds in magic.py to inherit
(decision 4 concerns the first pass's B,C,D,E box). Raw network outputs are
therefore affine-mapped using the per-parameter mean and spread of the NLLS
baseline. That is numerical conditioning -- the parameters span PHY1 ~ 1e-3 to
PKY1 ~ 4.8e3 -- and it imposes no hard limit.
"""

import json
import os
import time

import numpy as np
import torch

import dataset
import mf_torch

DTYPE = torch.float64
BASELINE = os.path.join("outputs", "baseline_params.parquet")

# Which mf_torch entry point each family uses, and how many free parameters it
# has. Lateral/longitudinal carry a trailing F_z0 that is data, not fitted.
FAMILY = {
    "lateral":      {"n_params": 22, "trailing_fz0": True},
    "longitudinal": {"n_params": 14, "trailing_fz0": True},
    "gx":           {"n_params": 7,  "trailing_fz0": False},
    "gy":           {"n_params": 15, "trailing_fz0": False},
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_family(family):
    """Cached arrays plus geometry for every spec in a family."""
    import fit_pipeline as fp

    specs = [s for s in fp.ALL_SPECS.values() if s.family == family]
    specs.sort(key=lambda s: s.spec_id)

    out = []
    for spec in specs:
        d = dataset.load_spec(spec.spec_id)
        F_z = -d["FZ"]                                  # magic.py convention
        rec = {
            "spec_id": spec.spec_id,
            "geom": dataset.feature_vector(d["geometry"]),
            "segment_id": d["segment_id"],
            "F_z": F_z,
        }
        if family in ("lateral", "gy"):
            rec["F_y"] = -d["FY"]
            rec["alpha"] = np.tan(d["SA"] * np.pi / 180.0)
            rec["gamma"] = np.sin(d["IA"] * np.pi / 180.0)
        if family in ("longitudinal", "gx"):
            rec["F_x"] = d["FX"]
        if family in ("longitudinal", "gx", "gy"):
            rec["s"] = d["SL"]
        if family == "gx":
            rec["alpha"] = np.tan(d["SA"] * np.pi / 180.0)
            rec["gamma"] = np.sin(d["IA"] * np.pi / 180.0)
        rec["F_z0"] = float(np.abs(d["FZ"]).mean())

        if family in ("gx", "gy"):
            # The upstream pure-slip vector this spec's G layer is fitted
            # against, from the NLLS baseline. magic.py passes long_*[-1] as
            # F_z0 for G_x and lat_*[-1] for G_y -- note those come from
            # different run families, which is preserved here.
            import fit_pipeline as _fp
            raw = np.load(os.path.join("outputs", "specs",
                                       spec.depends_on + ".npy"))
            # Stored vectors may carry a diameter before F_z0; strip it so
            # tm_lat/tm_long see F_z0 as the last element.
            src = _fp.strip_geometry(
                raw, "longitudinal" if family == "gx" else "lateral")
            rec["source_params"] = src
            rec["F_z0_source"] = float(src[-1])
        out.append(rec)
    return out


def baseline_stats(family):
    """Per-parameter mean and spread from the NLLS baseline, for output scaling."""
    import pandas as pd
    import fit_pipeline as fp

    # Fitted parameters only: drop the trailing metadata (diameter, F_z0)
    # that William's vectors carry but the network does not predict.
    names = fp.PARAM_NAMES[family][:fp.N_FITTED[family]]
    df = pd.read_parquet(BASELINE)
    df = df[(df.family == family) & (df.param_name.isin(names))]
    piv = df.pivot(index="spec_id", columns="param_name", values="value")
    piv = piv[names]
    mean = piv.mean(axis=0).to_numpy()
    # Spread across specs, floored so a parameter that happens to be identical
    # everywhere does not collapse the scale to zero.
    spread = piv.std(axis=0).to_numpy()
    spread = np.maximum(spread, 0.1 * np.maximum(np.abs(mean), 1e-6))
    spread = np.maximum(spread, 1e-6)
    return names, mean, spread


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ParamNet(torch.nn.Module):
    """geometry -> [architecture] -> raw -> affine -> Magic Formula parameters.

    hidden controls capacity, and with 6 training specs capacity is the whole
    ballgame:

      ()        linear in geometry, ~5 weights per output
      (32, 32)  MLP, ~2000 weights fitted to 6 points
      None      'mean' model -- ignores geometry entirely and learns one
                parameter vector for all tires. This is the null hypothesis:
                if it wins on leave-one-spec-out, geometry is not learnable
                from this dataset and that is the finding.
    """

    def __init__(self, n_in, n_params, mean, spread, hidden=(32, 32), seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.ignore_geometry = hidden is None
        if self.ignore_geometry:
            # Bias only: same prediction regardless of input geometry.
            layers = [torch.nn.Linear(n_in, n_params, dtype=DTYPE)]
            self.net = torch.nn.Sequential(*layers)
            self.net[-1].weight.requires_grad_(False)
        else:
            layers, prev = [], n_in
            for h in hidden:
                layers += [torch.nn.Linear(prev, h, dtype=DTYPE),
                           torch.nn.Tanh()]
                prev = h
            layers += [torch.nn.Linear(prev, n_params, dtype=DTYPE)]
            self.net = torch.nn.Sequential(*layers)
        # Start at the baseline mean: the last layer is zeroed so the initial
        # prediction is exactly `mean`, which is a sane physical starting point
        # rather than noise.
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)
        self.register_buffer("mean", torch.tensor(np.ascontiguousarray(mean),
                                                  dtype=DTYPE))
        self.register_buffer("spread", torch.tensor(
            np.ascontiguousarray(spread), dtype=DTYPE))

    def forward(self, g):
        return self.mean + self.spread * self.net(g)


def predict_force(family, params, rec, idx):
    """Predicted force for one spec on the rows selected by idx."""
    T = lambda a: torch.tensor(np.ascontiguousarray(a[idx]), dtype=DTYPE)
    F_z = T(rec["F_z"])

    if family == "lateral":
        x = torch.cat([params, torch.tensor([rec["F_z0"]], dtype=DTYPE)])
        return mf_torch.tm_lat(F_z, T(rec["alpha"]), T(rec["gamma"]), 1, x)[0]

    if family == "longitudinal":
        x = torch.cat([params, torch.tensor([rec["F_z0"]], dtype=DTYPE)])
        return mf_torch.tm_long(F_z, T(rec["s"]), 1, x)[0]

    # Combined slip. magic.py feeds G_x the longitudinal vector for the same
    # spec and G_y the lateral one (magic.py:1198, :1638); those come from the
    # NLLS baseline here, exactly as they did there, so the G networks learn
    # only the R-parameters.
    #
    # gamma is taken from the IA channel. first_pass_GY:1364 reads it from SA
    # instead -- a reported defect -- but that is magic.py's *fitting* helper,
    # which this does not use. mf_torch.GY takes gamma as an argument, and
    # every other function in magic.py reads camber from IA.
    src = torch.tensor(np.ascontiguousarray(rec["source_params"]), dtype=DTYPE)
    F_z0 = torch.tensor(rec["F_z0_source"], dtype=DTYPE)

    if family == "gx":
        base = mf_torch.tm_long(F_z, T(rec["s"]), 1, src)[0]
        G = mf_torch.GX(F_z, F_z0, T(rec["s"]), T(rec["alpha"]),
                        T(rec["gamma"]), params)
        return base * G                                   # magic.py:1100

    if family == "gy":
        base = mf_torch.tm_lat(F_z, T(rec["alpha"]), T(rec["gamma"]), 1, src)[0]
        G, S_vgy = mf_torch.GY(F_z, F_z0, T(rec["s"]), T(rec["alpha"]),
                               T(rec["gamma"]), params, src)
        return base * G + S_vgy                           # magic.py:1379-1382

    raise ValueError("unknown family: " + family)


def target_force(family, rec, idx):
    arr = (rec["F_y"] if family in ("lateral", "gy") else rec["F_x"])[idx]
    return torch.tensor(np.ascontiguousarray(arr), dtype=DTYPE)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def make_splits(recs, protocol, n_folds, seed):
    """Yield (fold_name, {spec_id: (train_idx, test_idx)}) pairs."""
    rng = np.random.default_rng(seed)

    if protocol == "segment":
        # GroupKFold by segment, applied within each spec so every fold still
        # sees every tire. Holding out whole segments is what stops adjacent
        # near-duplicate rows leaking across the split (decision 5).
        per_spec_folds = []
        for rec in recs:
            segs = np.unique(rec["segment_id"])
            order = rng.permutation(segs)
            per_spec_folds.append(np.array_split(order, n_folds))
        for k in range(n_folds):
            split = {}
            for rec, folds in zip(recs, per_spec_folds):
                held = np.isin(rec["segment_id"], folds[k])
                split[rec["spec_id"]] = (np.flatnonzero(~held),
                                         np.flatnonzero(held))
            yield "segment_fold%d" % k, split

    elif protocol == "spec":
        # Leave-one-spec-out: the honest geometry-generalisation test.
        for rec in recs:
            split = {}
            for r in recs:
                n = r["segment_id"].shape[0]
                if r["spec_id"] == rec["spec_id"]:
                    split[r["spec_id"]] = (np.array([], dtype=np.int64),
                                           np.arange(n))
                else:
                    split[r["spec_id"]] = (np.arange(n),
                                           np.array([], dtype=np.int64))
            yield "holdout_" + rec["spec_id"], split
    else:
        raise ValueError("unknown protocol: " + protocol)


def subsample(idx, max_rows, rng):
    if max_rows is None or idx.size <= max_rows:
        return idx
    return np.sort(rng.choice(idx, size=max_rows, replace=False))


def train_one(family, protocol, fold_name, split, recs, cfg):
    """One training run. Returns a result dict."""
    names, mean, spread = baseline_stats(family)
    n_params = FAMILY[family]["n_params"]

    G = np.stack([r["geom"] for r in recs])
    g_mu, g_sd = G.mean(0), np.maximum(G.std(0), 1e-9)
    G_std = torch.tensor(np.ascontiguousarray((G - g_mu) / g_sd),
                         dtype=DTYPE)

    model = ParamNet(G.shape[1], n_params, mean, spread,
                     hidden=cfg["hidden"], seed=cfg["seed"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    rng = np.random.default_rng(cfg["seed"])

    train_idx = {r["spec_id"]: subsample(split[r["spec_id"]][0],
                                         cfg["max_rows"], rng) for r in recs}

    t0, history = time.time(), []
    for step in range(cfg["steps"]):
        opt.zero_grad()
        total, n_terms = torch.zeros((), dtype=DTYPE), 0
        params_all = model(G_std)
        for j, rec in enumerate(recs):
            idx = train_idx[rec["spec_id"]]
            if idx.size == 0:
                continue
            pred = predict_force(family, params_all[j], rec, idx)
            targ = target_force(family, rec, idx)
            resid = pred - targ
            scale = max(float(np.std(
                (rec["F_y"] if family in ("lateral", "gy")
                 else rec["F_x"])[idx])), 1e-6)
            if cfg["normalize"]:
                resid = resid / scale

            delta = cfg.get("huber_delta")
            if delta:
                # Huber: quadratic near zero, linear beyond delta, so a small
                # population of very wrong points (the SL==0 fill samples)
                # stops dominating the gradient. delta is in units of the
                # spec's own force spread so it means the same thing on every
                # tire.
                d = delta if cfg["normalize"] else delta * scale
                a = torch.abs(resid)
                term = torch.where(a <= d, 0.5 * resid ** 2,
                                   d * (a - 0.5 * d))
                total = total + term.mean()
            else:
                total = total + (resid ** 2).mean()
            n_terms += 1
        loss = total / max(n_terms, 1)
        if not torch.isfinite(loss):
            return {"status": "diverged", "step": step, "family": family,
                    "protocol": protocol, "fold": fold_name, **cfg}
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
        opt.step()
        if step % max(cfg["steps"] // 20, 1) == 0:
            history.append((step, float(loss.detach())))

    # ---- held-out evaluation, on FULL held-out rows (no subsampling) ----
    with torch.no_grad():
        params_all = model(G_std)
        per_spec, se, cnt = {}, 0.0, 0
        for j, rec in enumerate(recs):
            te = split[rec["spec_id"]][1]
            if te.size == 0:
                continue
            pred = predict_force(family, params_all[j], rec, te)
            targ = target_force(family, rec, te)
            r = (pred - targ).numpy()
            per_spec[rec["spec_id"]] = float(np.sqrt(np.mean(r ** 2)))
            se += float(np.sum(r ** 2)); cnt += r.size
        heldout_rmse = float(np.sqrt(se / cnt)) if cnt else float("nan")
        recovered = {r["spec_id"]: params_all[j].numpy().tolist()
                     for j, r in enumerate(recs)}

    return {
        "status": "ok", "family": family, "protocol": protocol,
        "fold": fold_name, "heldout_rmse_N": heldout_rmse,
        "heldout_rmse_by_spec": per_spec, "n_heldout_rows": cnt,
        "param_names": names, "recovered_params": recovered,
        "final_loss": history[-1][1] if history else None,
        "loss_history": history, "wall_clock_s": time.time() - t0, **cfg,
    }
