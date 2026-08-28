"""Fit one spec and write its parameters and diagnostics to outputs/specs/.

Run as:  python run_spec_worker.py <spec_id>

One spec per process. That is what makes the wall-clock cap enforceable --
scipy.optimize.least_squares has no callback or timeout hook, so the only way
to bound a runaway fit is to kill its process. It also isolates memory and
keeps each spec's verbose=1 output in its own log.

Dependencies are read from disk, not passed in: a G_x worker reads the
long_* vector that the longitudinal worker already wrote. No IPC.
"""

import json
import os
import sys
import time
import traceback

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

OUT_DIR = os.path.join("outputs", "specs")


def vector_path(spec_id):
    return os.path.join(OUT_DIR, spec_id + ".npy")


def result_path(spec_id):
    return os.path.join(OUT_DIR, spec_id + ".json")


def jacobian_diagnostics(result):
    """Rank, condition number and asymptotic parameter standard errors.

    cov ~= residual_variance * pinv(J.T @ J), the same estimator
    docs/DECISIONS-mf-gp.md:122-145 used, so the 22-parameter numbers are
    comparable to the archived 20-parameter ones.
    """
    out = {"jac_rank": None, "jac_cond": None, "jac_shape": None,
           "std_err": None, "residual_variance": None}
    J = getattr(result, "jac", None)
    if J is None:
        return out
    J = np.asarray(J, dtype=np.float64)
    if J.ndim != 2 or J.size == 0:
        return out

    m, n = J.shape
    out["jac_shape"] = [int(m), int(n)]
    try:
        sv = np.linalg.svd(J, compute_uv=False)
        out["jac_rank"] = int(np.linalg.matrix_rank(J))
        smin = float(sv.min())
        out["jac_cond"] = float(sv.max() / smin) if smin > 0 else float("inf")
    except np.linalg.LinAlgError:
        pass

    try:
        dof = max(m - n, 1)
        s2 = 2.0 * float(result.cost) / dof          # cost = 0.5 * sum(fun**2)
        cov = s2 * np.linalg.pinv(J.T @ J)
        out["residual_variance"] = s2
        out["std_err"] = [float(v) for v in
                          np.sqrt(np.clip(np.diag(cov), 0.0, None))]
    except (np.linalg.LinAlgError, ValueError):
        pass
    return out


def main(spec_id):
    os.makedirs(OUT_DIR, exist_ok=True)

    # Patch scipy before fit_pipeline (and through it magic.py) binds the
    # least_squares name. Same mechanism as capture_reference.py, so the
    # reference and the pipeline run through identical code paths.
    import progress
    progress.install(every=250, cap_nfev=50_000)
    progress.keep_hot()

    # Imported here so an import failure lands inside the try/except below and
    # is recorded rather than lost.
    import fit_pipeline as fp

    spec = fp.ALL_SPECS[spec_id]
    record = {"spec_id": spec_id, "family": spec.family,
              "block_line": spec.block_line, "depends_on": spec.depends_on}

    t0 = time.time()
    try:
        source_vector = None
        if spec.depends_on is not None:
            dep = vector_path(spec.depends_on)
            if not os.path.exists(dep):
                raise RuntimeError(
                    "upstream vector missing: " + dep +
                    " (spec " + spec.depends_on + " did not produce one)")
            source_vector = np.load(dep)

        vector, result, cases, F_z0 = fp.run_spec(spec, source_vector)

        vector = np.asarray(vector, dtype=np.float64).ravel()
        np.save(vector_path(spec_id), vector)

        names = fp.PARAM_NAMES[spec.family]
        if len(names) != vector.size:
            raise RuntimeError(
                "param name/vector length mismatch for " + spec_id +
                ": " + str(len(names)) + " names vs " + str(vector.size))

        diag = jacobian_diagnostics(result)
        record.update({
            "status": "ok",
            "params": {n: float(v) for n, v in zip(names, vector)},
            "n_cases": len(cases),
            "n_rows": int(sum(np.asarray(c["FZ"]).squeeze().shape[0]
                              for c in cases)),
            "fz0_used": float(F_z0),
            "solver_status": int(result.status),
            "message": str(result.message),
            "nfev": int(result.nfev),
            "njev": (int(result.njev) if getattr(result, "njev", None)
                     is not None else None),
            "cost": float(result.cost),
            "optimality": float(result.optimality),
            "success": bool(result.success),
            "wall_clock_s": time.time() - t0,
        })
        record.update({k: diag[k] for k in
                       ("jac_rank", "jac_cond", "jac_shape",
                        "residual_variance")})
        if diag["std_err"] is not None:
            # Fitted params only. A trailing F_z0 is not a fitted parameter.
            record["std_err"] = {n: float(v) for n, v
                                 in zip(names, diag["std_err"])}

    except Exception:
        record.update({"status": "error",
                       "message": traceback.format_exc(),
                       "wall_clock_s": time.time() - t0})

    with open(result_path(spec_id), "w") as fh:
        json.dump(record, fh, indent=2)

    print("\n=== " + spec_id + " -> " + record["status"] +
          " in " + format(record["wall_clock_s"], ".1f") + "s ===", flush=True)
    return 0 if record["status"] == "ok" else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python run_spec_worker.py <spec_id>")
    sys.exit(main(sys.argv[1]))
