# Overnight runbook

Everything is built and tested. This is what to run tonight.

## Before you leave this machine

The desktop needs the repo **and** the raw data. `data/` is committed to git
(1.37 GB), so a clone or pull brings it. Two optional shortcuts that save an
hour on the desktop:

- `outputs/cache/` (10 MB) — pre-segmented data. Copying it skips ~5 min of
  segmentation.
- `outputs/specs/` and `outputs/baseline_*.parquet` (~1 MB) — the NLLS
  baseline. Copying it skips a ~30 min re-run.

Both are gitignored, so copy them by hand if you want the shortcut. If you
don't, the sweep regenerates them automatically — it just takes longer.

## On the desktop

```powershell
cd <repo>
powershell -ExecutionPolicy Bypass -File setup_desktop.ps1
```

This builds `.venv`, installs pinned dependencies, and **verifies the torch
Magic Formula against `magic.py`**. If that gate fails, stop — nothing after it
is trustworthy.

Then smoke-test before committing the night to it:

```powershell
.\.venv\Scripts\python.exe -u sweep.py --quick --workers 4
```

~2 minutes. You should see four families reporting a held-out RMSE and a
"Decision 3" consistency block. If that works, start the real run:

```powershell
.\.venv\Scripts\python.exe -u sweep.py --workers 8 *> outputs\sweep\overnight.log
```

Set `--workers` to about `cores - 2`. Expect **~2 hours** on 8 workers, not a
full night — the margin is deliberate.

## In the morning

```powershell
.\.venv\Scripts\python.exe sweep.py --summary-only
```

Reads whatever finished and prints the summary. Artifacts:

| file | contents |
|---|---|
| `outputs/sweep/summary.parquet` | one row per run |
| `outputs/sweep/best_configs.parquet` | best config per family and protocol |
| `outputs/sweep/consistency.parquet` | the decision-3 result, per parameter |
| `outputs/sweep/results_shard*.jsonl` | raw records including recovered parameters |

## If something goes wrong

- **It died partway.** Re-run the same command. Every finished run is already
  on disk and is skipped.
- **A config diverged.** Recorded as `status="diverged"` and skipped; the sweep
  continues. Expected for `lr=1e-2` on some families.
- **It ran all night and barely progressed.** That is the power-throttling
  failure from 2026-08-26. `sweep.py` calls `progress.keep_hot()` to prevent
  it, and prints `throttling_disabled=True` at startup — check that line. If it
  says `False`, run the shell as Administrator.
- **`mf_torch.py` gate fails.** Do not proceed. It means the desktop's numpy or
  scipy differs enough to change the physics.

## What you will and will not have

**Will:** held-out force RMSE per family with a spread across seeds and folds,
under two protocols; recovered P/R parameters for all 18 specs; and a
per-parameter comparison of across-spec consistency against the NLLS baseline —
the decision-3 question.

**Will not:** evidence that this predicts an untested tire size. There are 6
lateral specs and ~3 informative geometry dimensions. The `spec` protocol
(leave-one-spec-out) measures that honestly and is expected to look bad. Quote
`segment` numbers as "unseen sweeps of a known tire", never as generalisation.

Early indication from the smoke test, for reference — lateral held-out RMSE
~91 N, and network parameters more consistent across specs than the NLLS fits
on 18 of 22 lateral parameters (median CV 0.76 to 0.13). The overnight sweep
tests whether that survives more seeds, folds and configs.
