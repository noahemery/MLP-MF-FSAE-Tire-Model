"""Capture the two live magic.py fits as the acceptance-gate reference.

Runs magic.py EXACTLY as committed, via runpy, so nothing in it is modified or
monkeypatched. Pulls the two module-level vectors the live blocks produce and
saves them to outputs/reference/.

This must run BEFORE the live blocks are commented out.
"""
import json
import os
import runpy
import time

os.environ.setdefault("MPLBACKEND", "Agg")   # no windows during an unattended run

import numpy as np

# Patch scipy.optimize.least_squares BEFORE runpy executes magic.py, so that
# magic.py's own `from scipy.optimize import least_squares` binds the
# instrumented name. magic.py itself is not touched.
import progress
progress.install(every=250, cap_nfev=50_000)
_hot = progress.keep_hot()
print('power: throttling_disabled=%(throttling_disabled)s execution_state_set=%(execution_state_set)s' % _hot, flush=True)

OUT = os.path.join("outputs", "reference")
TARGETS = ("lat_180X60_R20_70", "GY_180X60_R20_70")

os.makedirs(OUT, exist_ok=True)

print(f"[{time.strftime('%H:%M:%S')}] running magic.py unmodified ...", flush=True)
t0 = time.time()
g = runpy.run_path("magic.py", run_name="magic_reference")
elapsed = time.time() - t0
print(f"[{time.strftime('%H:%M:%S')}] magic.py finished in {elapsed:.1f}s", flush=True)

meta = {"total_wall_clock_s": elapsed, "vectors": {}}
missing = [n for n in TARGETS if n not in g]
if missing:
    raise SystemExit(f"magic.py did not define: {missing}")

for name in TARGETS:
    vec = np.asarray(g[name], dtype=np.float64)
    path = os.path.join(OUT, f"{name}.npy")
    np.save(path, vec)
    meta["vectors"][name] = {"len": int(vec.size), "path": path,
                             "values": [float(v) for v in vec.ravel()]}
    print(f"\n{name}  (n={vec.size})")
    for i, v in enumerate(vec.ravel()):
        print(f"  [{i:2d}] {v!r}")

with open(os.path.join(OUT, "reference_meta.json"), "w") as fh:
    json.dump(meta, fh, indent=2)

print(f"\nsaved to {OUT}/", flush=True)
