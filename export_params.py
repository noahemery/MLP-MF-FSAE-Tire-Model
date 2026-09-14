"""Write the team-facing CSV exports from the parquet artifacts.

Two files, both regenerated from outputs/baseline_params.parquet so they can
never drift from the fits:

  outputs/tire_parameters.csv   one row per spec, one column per parameter
  outputs/pooled_fallback.csv   the pooled sets, long format

Run after run_baseline.py. Previously done by hand in a scratch script, which
is why this exists -- the CSVs the team was sent need to be reproducible.

  python export_params.py
"""
import os

import pandas as pd

import fit_pipeline as fp
import tire_predict as tp

OUT = "outputs"
PARQUET = os.path.join(OUT, "baseline_params.parquet")


def main():
    df = pd.read_parquet(PARQUET)

    wide = (df.pivot_table(index=["spec_id", "family"],
                           columns="param_name", values="value")
              .reset_index())
    # F_z0 and diameter_in are metadata, not fitted. Keep F_z0 up front where
    # it was, and diameter_in last, matching the file the team already has.
    cols = list(wide.columns)
    for meta in ("F_z0", "diameter_in"):
        if meta in cols:
            cols.remove(meta)
    ordered = (["spec_id", "family"]
               + (["F_z0"] if "F_z0" in wide.columns else [])
               + sorted(c for c in cols if c not in ("spec_id", "family"))
               + (["diameter_in"] if "diameter_in" in wide.columns else []))
    wide = wide[ordered]
    wide = wide.sort_values(["family", "spec_id"], kind="stable")

    pooled = pd.DataFrame(
        [{"family": fam, "param_name": name, "value": float(val)}
         for fam in sorted(tp.POOLED)
         for name, val in zip(fp.PARAM_NAMES[fam], tp.POOLED[fam])])

    os.makedirs(OUT, exist_ok=True)
    for frame, name in ((wide, "tire_parameters.csv"),
                        (pooled, "pooled_fallback.csv")):
        path = os.path.join(OUT, name)
        frame.to_csv(path, index=False)
        print("wrote %-34s %d rows x %d cols"
              % (path, frame.shape[0], frame.shape[1]))

    print("\nfamilies: " + ", ".join(sorted(df.family.unique())))
    print("specs:    %d" % df.spec_id.nunique())


if __name__ == "__main__":
    main()
