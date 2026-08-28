"""Turn outputs/sweep/results.jsonl into readable results.

Produces three things:

  summary.parquet      one row per run
  best_configs.parquet the best config per family and protocol
  consistency.parquet  the decision-3 test -- are network-recovered parameters
                       more consistent across specs than independent NLLS fits?

The consistency metric is the across-spec coefficient of variation,
std(param over specs) / |mean(param over specs)|, computed per parameter for
both the NLLS baseline and the network. Decision 3 predicts the network's
should be lower: shared weights cannot pick a different arbitrary branch per
tire, so the recovered vectors should agree more.
"""

import json
import os

import numpy as np
import pandas as pd

OUT_DIR = os.path.join("outputs", "sweep")
RESULTS = os.path.join(OUT_DIR, "results.jsonl")
BASELINE = os.path.join("outputs", "baseline_params.parquet")


def _result_files():
    if not os.path.isdir(OUT_DIR):
        return []
    return [os.path.join(OUT_DIR, f) for f in sorted(os.listdir(OUT_DIR))
            if f.startswith("results") and f.endswith(".jsonl")]


def load_runs():
    rows = []
    if not _result_files():
        return pd.DataFrame(), []
    raw = []
    for path in _result_files():
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    for r in raw:
        rows.append({
            "family": r.get("family"), "protocol": r.get("protocol"),
            "fold": r.get("fold"), "seed": r.get("seed"), "lr": r.get("lr"),
            "hidden": str(tuple(r["hidden"])) if r.get("hidden") else None,
            "normalize": r.get("normalize"), "status": r.get("status"),
            "heldout_rmse_N": r.get("heldout_rmse_N"),
            "n_heldout_rows": r.get("n_heldout_rows"),
            "final_loss": r.get("final_loss"),
            "wall_clock_s": r.get("wall_clock_s"),
        })
    return pd.DataFrame(rows), raw


def cv(values):
    """Across-spec coefficient of variation, robust to a near-zero mean."""
    v = np.asarray(values, dtype=float)
    m = np.mean(v)
    denom = max(abs(m), 1e-9)
    return float(np.std(v) / denom)


def consistency_table(raw):
    """Compare across-spec parameter spread: NLLS baseline vs network."""
    if not os.path.exists(BASELINE):
        return pd.DataFrame()
    base = pd.read_parquet(BASELINE)
    out = []

    ok = [r for r in raw if r.get("status") == "ok"
          and r.get("protocol") == "segment" and r.get("recovered_params")]
    if not ok:
        return pd.DataFrame()

    df = pd.DataFrame([{"family": r["family"], "rmse": r["heldout_rmse_N"],
                        "i": i} for i, r in enumerate(ok)])

    for family, grp in df.groupby("family"):
        best = ok[int(grp.loc[grp.rmse.idxmin(), "i"])]
        names = best["param_names"]
        rec = best["recovered_params"]
        spec_ids = sorted(rec)

        b = base[(base.family == family) & (base.param_name != "F_z0")]
        piv = b.pivot(index="spec_id", columns="param_name", values="value")

        for j, name in enumerate(names):
            net_vals = [rec[s][j] for s in spec_ids]
            if name not in piv.columns:
                continue
            nlls_vals = piv[name].to_numpy()
            out.append({
                "family": family, "param_name": name,
                "cv_nlls": cv(nlls_vals), "cv_network": cv(net_vals),
                "net_more_consistent": cv(net_vals) < cv(nlls_vals),
                "best_run_rmse_N": best["heldout_rmse_N"],
            })
    return pd.DataFrame(out)


def build(verbose=True):
    os.makedirs(OUT_DIR, exist_ok=True)
    runs, raw = load_runs()
    if runs.empty:
        print("no results yet in " + RESULTS)
        return

    runs.to_parquet(os.path.join(OUT_DIR, "summary.parquet"), index=False)

    ok = runs[runs.status == "ok"].copy()
    if verbose:
        print("\n" + "=" * 78)
        print("SWEEP SUMMARY   %d runs  (%d ok, %d failed/diverged)"
              % (len(runs), len(ok), len(runs) - len(ok)))
        print("=" * 78)

    if ok.empty:
        print("no successful runs")
        return

    # Best config per family x protocol, averaged over folds and seeds.
    grp = (ok.groupby(["family", "protocol", "lr", "hidden", "normalize"])
             .agg(rmse_mean=("heldout_rmse_N", "mean"),
                  rmse_std=("heldout_rmse_N", "std"),
                  n=("heldout_rmse_N", "size"))
             .reset_index())
    best = (grp.sort_values("rmse_mean")
               .groupby(["family", "protocol"], as_index=False).first())
    best.to_parquet(os.path.join(OUT_DIR, "best_configs.parquet"), index=False)

    if verbose:
        print("\nBest config per family and protocol "
              "(held-out force RMSE, newtons):\n")
        print("  %-13s %-8s %-7s %-9s %-6s %10s %9s %4s"
              % ("family", "protocol", "lr", "hidden", "norm", "rmse_mean",
                 "rmse_std", "n"))
        print("  " + "-" * 74)
        for _, r in best.sort_values(["family", "protocol"]).iterrows():
            print("  %-13s %-8s %-7g %-9s %-6s %10.2f %9.2f %4d"
                  % (r.family, r.protocol, r.lr, r.hidden, r.normalize,
                     r.rmse_mean, r.rmse_std if r.rmse_std == r.rmse_std
                     else float("nan"), r.n))

    cons = consistency_table(raw)
    if not cons.empty:
        cons.to_parquet(os.path.join(OUT_DIR, "consistency.parquet"),
                        index=False)
        if verbose:
            print("\nDecision 3 -- parameter consistency across specs")
            print("  (across-spec coefficient of variation; lower = the "
                  "parameter\n   agrees more between tires)\n")
            for family, g in cons.groupby("family"):
                won = int(g.net_more_consistent.sum())
                print("  %-13s network more consistent on %d of %d parameters"
                      "   median CV  nlls=%.3g  network=%.3g"
                      % (family, won, len(g), g.cv_nlls.median(),
                         g.cv_network.median()))

    if verbose:
        print("\nwrote %s/{summary,best_configs,consistency}.parquet\n"
              % OUT_DIR)
        print("NOTE: held-out RMSE under protocol 'segment' means unseen "
              "sweeps of a\n      tire the model HAS seen. Protocol 'spec' is "
              "the honest\n      generalisation test to an unseen tire size, "
              "and with 6 lateral\n      specs it is expected to be much "
              "worse. Do not quote 'segment'\n      as evidence of geometry "
              "generalisation.")


if __name__ == "__main__":
    build()
