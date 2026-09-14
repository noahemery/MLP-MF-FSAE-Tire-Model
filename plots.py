"""Figures that show whether the tire model actually works.

    python plots.py            # all figures to outputs/figures/

Each figure answers one question:

  fig1_tire_curves      Does the fitted curve match the measured tire?
  fig2_fit_quality      Which of the 18 fits are good and which are suffering?
  fig3_pred_vs_meas     How tight is the model overall, per family?
  fig4_problem_runs     What is different about the straight runs that fail?
  fig5_neural_vs_nlls   Does the geometry network beat per-tire fitting?
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import dataset
import train

FIG_DIR = os.path.join("outputs", "figures")
COLORS = {"lateral": "#2c7fb8", "longitudinal": "#d95f0e",
          "gx": "#31a354", "gy": "#756bb1"}


def _baseline_params(spec_id, family):
    import fit_pipeline as fp
    raw = np.load(os.path.join("outputs", "specs", spec_id + ".npy"))
    # strip_geometry drops any diameter metadata and keeps F_z0 last, so this
    # works whether the stored vector is the old or the new layout.
    v = fp.strip_geometry(raw, family)
    return torch.tensor(v[:-1] if family in ("lateral", "longitudinal") else v,
                        dtype=torch.float64)


def _predict(family, spec_id, rec, idx):
    with torch.no_grad():
        return train.predict_force(family, _baseline_params(spec_id, family),
                                   rec, idx).numpy()


# ---------------------------------------------------------------------------

def fig1_tire_curves(spec_id="lat_160X75_R20_70"):
    """The classic tire plot: lateral force vs slip angle, one curve per load.

    This is the figure a vehicle dynamics engineer looks at first. Measured
    rig data as faint dots, the fitted Magic Formula as a solid line.
    """
    recs = {r["spec_id"]: r for r in train.load_family("lateral")}
    rec = recs[spec_id]
    d = dataset.load_spec(spec_id)

    sa = d["SA"]                      # slip angle, degrees
    fy = -d["FY"]
    fz = -d["FZ"]
    ia = d["IA"]

    # Near-zero camber only, so the curves are not a mix of camber states.
    base = np.abs(ia) < 0.5
    loads = [(200, 500), (500, 800), (800, 1200), (1200, 1800)]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for (lo, hi), col in zip(loads, ["#c6dbef", "#6baed6", "#2171b5", "#08306b"]):
        m = base & (fz >= lo) & (fz < hi)
        if m.sum() < 500:
            continue
        idx = np.flatnonzero(m)
        ax.scatter(sa[idx], fy[idx], s=1, alpha=0.10, color=col)

        order = idx[np.argsort(sa[idx])]
        pred = _predict("lateral", spec_id, rec, order)
        # Smooth the model line by averaging predictions in slip-angle bins.
        bins = np.linspace(sa[order].min(), sa[order].max(), 120)
        which = np.digitize(sa[order], bins)
        bx = [sa[order][which == k].mean() for k in range(1, len(bins))
              if (which == k).sum() > 3]
        by = [pred[which == k].mean() for k in range(1, len(bins))
              if (which == k).sum() > 3]
        ax.plot(bx, by, color=col, lw=2.2,
                label="F_z %d-%d N" % (lo, hi))

    ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("Slip angle (degrees)")
    ax.set_ylabel("Lateral force F_y (N)")
    ax.set_title("Fitted Magic Formula vs measured rig data\n%s  "
                 "(dots = measurements, lines = model)" % spec_id)
    ax.legend(title="vertical load", loc="upper left", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    p = os.path.join(FIG_DIR, "fig1_tire_curves.png")
    fig.savefig(p, dpi=140); plt.close(fig)
    return p


def fig2_fit_quality():
    """Every fit, ranked by error as a percentage of peak force."""
    rows = []
    for fam in ("lateral", "longitudinal", "gx", "gy"):
        for rec in train.load_family(fam):
            idx = np.arange(rec["segment_id"].shape[0])
            pred = _predict(fam, rec["spec_id"], rec, idx)
            targ = train.target_force(fam, rec, idx).numpy()
            rmse = float(np.sqrt(np.mean((pred - targ) ** 2)))
            peak = float(np.percentile(np.abs(targ), 95))
            rows.append((rec["spec_id"], fam, 100 * rmse / peak))

    rows.sort(key=lambda r: r[2])
    names = [r[0] for r in rows]
    vals = [r[2] for r in rows]
    cols = [COLORS[r[1]] for r in rows]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(range(len(rows)), vals, color=cols)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(names, fontsize=8)
    ax.axvline(8, color="green", ls="--", lw=1, label="good (<8%)")
    ax.axvline(15, color="red", ls="--", lw=1, label="suffering (>15%)")
    ax.set_xlabel("Fit error as % of peak force  (lower is better)")
    ax.set_title("Fit quality, all 18 specs\n"
                 "every lateral fit is good; the failures are all straight-line data")
    ax.legend(loc="lower right")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS.values()]
    ax.legend(handles + [plt.Line2D([], [], color="green", ls="--"),
                         plt.Line2D([], [], color="red", ls="--")],
              list(COLORS) + ["good (<8%)", "suffering (>15%)"],
              loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    p = os.path.join(FIG_DIR, "fig2_fit_quality.png")
    fig.savefig(p, dpi=140); plt.close(fig)
    return p


def fig3_pred_vs_meas():
    """Predicted vs measured force. A perfect model sits on the diagonal."""
    fams = ("lateral", "longitudinal", "gx", "gy")
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4))
    rng = np.random.default_rng(0)

    for ax, fam in zip(axes, fams):
        P, T = [], []
        for rec in train.load_family(fam):
            idx = np.arange(rec["segment_id"].shape[0])
            sub = rng.choice(idx, size=min(6000, idx.size), replace=False)
            P.append(_predict(fam, rec["spec_id"], rec, sub))
            T.append(train.target_force(fam, rec, sub).numpy())
        P, T = np.concatenate(P), np.concatenate(T)
        lim = np.percentile(np.abs(T), 99.5)

        ax.scatter(T, P, s=1, alpha=0.05, color=COLORS[fam])
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=1)
        r = P - T
        ax.set_title("%s\nRMSE %.0f N  (%.1f%% of peak)"
                     % (fam, np.sqrt(np.mean(r ** 2)),
                        100 * np.sqrt(np.mean(r ** 2))
                        / np.percentile(np.abs(T), 95)), fontsize=10)
        ax.set_xlabel("measured force (N)"); ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim); ax.grid(alpha=0.25)
    axes[0].set_ylabel("predicted force (N)")
    fig.suptitle("Predicted vs measured. Points on the dashed line are perfect.",
                 y=1.02)
    fig.tight_layout()
    p = os.path.join(FIG_DIR, "fig3_pred_vs_meas.png")
    fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    return p


def fig4_problem_runs():
    """Why do the 20.5x7.0-13 straight runs fit so much worse?

    long_205X70_R20_70 (runs 51/52) errors at 27%; long_180X60_R20_60
    (runs 69/70) at 14%, from identical code and identical segmentation
    literals. Compare them directly.
    """
    pairs = [("long_205X70_R20_70", "SUFFERING  27% error  (runs 51, 52)"),
             ("long_180X60_R20_60", "OK  14% error  (runs 69, 70)")]
    recs = {r["spec_id"]: r for r in train.load_family("longitudinal")}

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for row, (sid, title) in enumerate(pairs):
        rec = recs[sid]
        d = dataset.load_spec(sid)
        s, fx, fz = d["SL"], d["FX"], -d["FZ"]
        idx = np.arange(s.shape[0])
        pred = _predict("longitudinal", sid, rec, idx)

        a = axes[row][0]
        a.scatter(s, fx, s=1, alpha=0.08, color="0.6", label="measured")
        a.scatter(s, pred, s=1, alpha=0.08, color="#d95f0e", label="model")
        a.set_xlabel("slip ratio"); a.set_ylabel("F_x (N)")
        a.set_title(sid + "\n" + title, fontsize=9)
        a.grid(alpha=0.25)

        a = axes[row][1]
        a.scatter(fz, fx - pred, s=1, alpha=0.08, color="#d95f0e")
        a.axhline(0, color="k", lw=0.8)
        a.set_xlabel("vertical load F_z (N)"); a.set_ylabel("residual (N)")
        a.set_title("residual vs load", fontsize=9); a.grid(alpha=0.25)

        a = axes[row][2]
        a.hist(fz, bins=60, color="#2c7fb8")
        a.set_xlabel("vertical load F_z (N)"); a.set_ylabel("samples")
        a.set_title("load coverage", fontsize=9); a.grid(alpha=0.25)

    fig.suptitle("Diagnosing the suffering longitudinal fits: same code, "
                 "same settings, very different result", y=1.00)
    fig.tight_layout()
    p = os.path.join(FIG_DIR, "fig4_problem_runs.png")
    fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    return p


def fig5_neural_vs_nlls():
    """Held-out error: geometry network vs fitting each tire independently."""
    import pandas as pd
    best = pd.read_parquet(os.path.join("outputs", "sweep",
                                        "best_configs.parquet"))
    fams = ("lateral", "longitudinal", "gx", "gy")
    nlls = {}
    for fam in fams:
        se = cnt = 0
        for rec in train.load_family(fam):
            idx = np.arange(rec["segment_id"].shape[0])
            r = _predict(fam, rec["spec_id"], rec, idx) - \
                train.target_force(fam, rec, idx).numpy()
            se += float(np.sum(r ** 2)); cnt += r.size
        nlls[fam] = np.sqrt(se / cnt)

    seg = [best[(best.family == f) & (best.protocol == "segment")]
           .rmse_mean.iloc[0] for f in fams]
    spec = [best[(best.family == f) & (best.protocol == "spec")]
            .rmse_mean.iloc[0] for f in fams]

    x = np.arange(len(fams)); w = 0.27
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w, [nlls[f] for f in fams], w, label="NLLS, fitted per tire",
           color="0.65")
    ax.bar(x, seg, w, label="network, unseen sweeps (known tire)",
           color="#2c7fb8")
    ax.bar(x + w, spec, w, label="network, UNSEEN TIRE", color="#08306b")
    ax.set_xticks(x); ax.set_xticklabels(fams)
    ax.set_ylabel("force RMSE (N)   lower is better")
    ax.set_title("Does predicting parameters from geometry beat fitting each "
                 "tire?\nLateral: yes. The rest: not yet.")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    p = os.path.join(FIG_DIR, "fig5_neural_vs_nlls.png")
    fig.savefig(p, dpi=140); plt.close(fig)
    return p


def fig6_error_distribution():
    """Log-scale error distributions, splitting valid data from SL==0 fill.

    Linear plots hide this: the SL==0 samples are a separate population with
    errors an order of magnitude larger, not a tail of the same distribution.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    sid = "long_205X70_R20_70"
    recs = {r["spec_id"]: r for r in train.load_family("longitudinal")}
    rec = recs[sid]
    d = dataset.load_spec(sid)
    idx = np.arange(rec["segment_id"].shape[0])
    pred = _predict("longitudinal", sid, rec, idx)
    targ = train.target_force("longitudinal", rec, idx).numpy()
    resid = np.abs(pred - targ)
    bad = d["SL"] == 0.0

    a = axes[0]
    bins = np.logspace(-1, 4, 70)
    a.hist(np.clip(resid[~bad], 1e-1, None), bins=bins, alpha=0.75,
           color="#2c7fb8", label="valid (SL != 0)")
    a.hist(np.clip(resid[bad], 1e-1, None), bins=bins, alpha=0.75,
           color="#d62728", label="SL == 0 fill value")
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel("|error| (N), log scale"); a.set_ylabel("samples, log scale")
    a.set_title("Two separate populations, not one tail"); a.legend(fontsize=8)
    a.grid(alpha=0.25, which="both")

    a = axes[1]
    a.scatter(np.abs(targ[~bad]), np.clip(resid[~bad], 1e-1, None), s=1,
              alpha=0.05, color="#2c7fb8")
    a.scatter(np.abs(targ[bad]), np.clip(resid[bad], 1e-1, None), s=1,
              alpha=0.15, color="#d62728")
    a.set_xscale("log"); a.set_yscale("log")
    a.plot([10, 4000], [10, 4000], "k--", lw=1, label="100% error")
    a.set_xlabel("|measured force| (N)"); a.set_ylabel("|error| (N)")
    a.set_title("Red sits on the 100% line: model predicts ~0")
    a.legend(fontsize=8); a.grid(alpha=0.25, which="both")

    a = axes[2]
    for lbl, m, c in (("all points", slice(None), "0.5"),
                      ("SL != 0 only", ~bad, "#2c7fb8")):
        r = np.sort(resid[m])
        a.plot(100 * np.arange(len(r)) / len(r), r, color=c, lw=2, label=lbl)
    a.set_yscale("log")
    a.set_xlabel("percentile of samples"); a.set_ylabel("|error| (N), log")
    a.set_title("Error percentiles, before and after filtering")
    a.legend(fontsize=8); a.grid(alpha=0.25, which="both")

    fig.suptitle(sid + " -- why linear plots understate the problem", y=1.02)
    fig.tight_layout()
    p = os.path.join(FIG_DIR, "fig6_error_distribution.png")
    fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    return p


def fig7_sl_dropout():
    """The SL channel itself: where the fill value lives."""
    import scipy.io
    fig, axes = plt.subplots(2, 2, figsize=(15, 7))
    for row, run in enumerate((51, 69)):
        m = scipy.io.loadmat("data/straight_SI/B2356raw%d.mat" % run)
        sl = np.asarray(m["SL"]).ravel()
        fx = np.asarray(m["FX"]).ravel()
        et = np.asarray(m["ET"]).ravel()
        n = min(40000, len(sl))
        a = axes[row][0]
        a.plot(et[:n], sl[:n], lw=0.4, color="#2c7fb8")
        z = sl[:n] == 0.0
        a.plot(et[:n][z], sl[:n][z], ".", ms=1, color="#d62728")
        a.set_xlabel("elapsed time (s)"); a.set_ylabel("slip ratio SL")
        a.set_title("raw%d  SL channel (red = exactly 0.0, %.0f%% of file)"
                    % (run, 100 * (sl == 0.0).mean()), fontsize=9)
        a.grid(alpha=0.25)

        a = axes[row][1]
        a.scatter(sl[::20], fx[::20], s=1, alpha=0.06, color="#2c7fb8")
        zz = sl == 0.0
        a.scatter(sl[zz][::20], fx[zz][::20], s=2, alpha=0.25, color="#d62728")
        a.set_xlabel("slip ratio SL"); a.set_ylabel("F_x (N)")
        a.set_title("raw%d  the red column at SL=0 carries up to %.0f N"
                    % (run, np.abs(fx[zz]).max()), fontsize=9)
        a.grid(alpha=0.25)
    fig.suptitle("The SL channel reads exactly zero while FX still records "
                 "thousands of newtons", y=1.00)
    fig.tight_layout()
    p = os.path.join(FIG_DIR, "fig7_sl_dropout.png")
    fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    return p


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    for fn in (fig1_tire_curves, fig2_fit_quality, fig3_pred_vs_meas,
               fig4_problem_runs, fig5_neural_vs_nlls,
               fig6_error_distribution, fig7_sl_dropout):
        try:
            print("wrote " + fn())
        except Exception as exc:
            print("FAILED %s: %s" % (fn.__name__, exc))
