"""Tire geometry and cached per-spec training data.

Two jobs:

1. Parse geometry from line 1 of the .dat file paired with each .mat, which is
   the only place tire dimensions exist (README).
2. Segment the raw .mat files once via fit_pipeline and cache the result to
   outputs/cache/, so a sweep of many training runs does not re-segment 1.4 GB
   of raw data every time. Segmentation costs ~30 s per spec; loading the cache
   costs milliseconds.

The cache is keyed only by spec_id because segmentation is fully determined by
fit_pipeline's config, which audit_literals.py pins to magic.py's literals.
Delete outputs/cache/ to force a rebuild.
"""

import os
import re

import numpy as np

CACHE_DIR = os.path.join("outputs", "cache")

# Hoosier <overall_dia>X<section_width>-<rim_dia>, R20[, C2000]; Rim_Width=<w>
_HEADER = re.compile(
    r"Tire_Name=\s*(?P<brand>\w+)\s+"
    r"(?P<od>[\d.]+)X(?P<sw>[\d.]+)-(?P<rd>[\d.]+)"
    r"(?P<rest>[^;]*);.*?Rim_Width=\s*(?P<rw>[\d.]+)")

# Channels each family needs from the segmented data.
CHANNELS = {
    "lateral": ("FZ", "FY", "SA", "IA"),
    "longitudinal": ("FZ", "FX", "SL"),
    "gx": ("FZ", "FX", "SL", "SA", "IA"),
    "gy": ("FZ", "FY", "SL", "SA", "IA"),
}


def parse_dat_header(dat_path):
    """Geometry from the first line of a .dat file."""
    with open(dat_path, "r", errors="replace") as fh:
        line = fh.readline()
    m = _HEADER.search(line)
    if not m:
        raise ValueError("could not parse geometry from " + dat_path +
                         "\n  " + line[:200])
    return {
        "overall_dia_in": float(m.group("od")),
        "section_width_in": float(m.group("sw")),
        "rim_dia_in": float(m.group("rd")),
        "rim_width_in": float(m.group("rw")),
        # C2000 appears only on the 16.0X7.5-10, so it is confounded with size
        # (DECISIONS.md open questions). Recorded, not used as a feature.
        "compound_c2000": 1.0 if "C2000" in m.group("rest") else 0.0,
    }


def geometry_for_spec(spec):
    """Geometry for one spec, cross-checked across all its raw files.

    Every .mat used by a spec must report identical dimensions; a mismatch
    means the config points at files from different tires.
    """
    paths = []
    for cf in spec.case_files:
        paths.append(cf.path if hasattr(cf, "path") else cf)

    geos = []
    for p in paths:
        dat = os.path.splitext(p)[0] + ".dat"
        if not os.path.exists(dat):
            raise FileNotFoundError("no .dat sibling for " + p)
        geos.append(parse_dat_header(dat))

    first = geos[0]
    for g, p in zip(geos[1:], paths[1:]):
        if g != first:
            raise ValueError("geometry mismatch within " + spec.spec_id +
                             ": " + str(first) + " vs " + str(g) +
                             " from " + p)
    return first


# Feature order used by the networks. rim_dia is included even though it takes
# only two values (10, 13); R20 is NOT a feature because it is constant across
# the entire dataset and carries no information (DECISIONS.md).
FEATURES = ("overall_dia_in", "section_width_in", "rim_dia_in", "rim_width_in")


def feature_vector(geo):
    return np.array([geo[k] for k in FEATURES], dtype=np.float64)


def cache_path(spec_id):
    return os.path.join(CACHE_DIR, spec_id + ".npz")


def build_cache(spec, force=False):
    """Segment one spec and cache its channels. Returns the cache path."""
    import fit_pipeline as fp

    path = cache_path(spec.spec_id)
    if os.path.exists(path) and not force:
        return path

    os.makedirs(CACHE_DIR, exist_ok=True)
    cases = fp.build_cases(spec)
    if not cases:
        raise RuntimeError("no segments for " + spec.spec_id)

    wanted = CHANNELS[spec.family]
    cols, seg_id = {c: [] for c in wanted}, []
    for i, case in enumerate(cases):
        n = np.asarray(case["FZ"]).squeeze().shape[0]
        for c in wanted:
            cols[c].append(np.asarray(case[c], dtype=np.float64).reshape(-1))
        seg_id.append(np.full(n, i, dtype=np.int64))

    payload = {c: np.concatenate(cols[c]) for c in wanted}
    payload["segment_id"] = np.concatenate(seg_id)
    payload["n_cases"] = np.array([len(cases)])

    geo = geometry_for_spec(spec)
    for k, v in geo.items():
        payload["geo_" + k] = np.array([v], dtype=np.float64)

    np.savez_compressed(path, **payload)
    return path


def load_spec(spec_id):
    """Load one spec's cached arrays. Raises if the cache is missing."""
    path = cache_path(spec_id)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "no cache for " + spec_id + "; run build_all_caches() first")
    z = np.load(path)
    out = {k: z[k] for k in z.files}
    out["n_cases"] = int(out["n_cases"][0])
    out["geometry"] = {k[4:]: float(out[k][0]) for k in z.files
                       if k.startswith("geo_")}
    return out


def build_all_caches(families=None, force=False, verbose=True):
    """Segment and cache every spec. Idempotent; safe to re-run."""
    import fit_pipeline as fp

    made = []
    for spec_id, spec in fp.ALL_SPECS.items():
        if families and spec.family not in families:
            continue
        existed = os.path.exists(cache_path(spec_id)) and not force
        p = build_cache(spec, force=force)
        if verbose:
            z = np.load(p)
            print("  %-22s %-13s segments=%-4d rows=%-7d %s"
                  % (spec_id, spec.family, int(z["n_cases"][0]),
                     z["segment_id"].shape[0],
                     "(cached)" if existed else "BUILT"))
        made.append(p)
    return made


if __name__ == "__main__":
    import fit_pipeline as fp

    print("geometry per spec\n")
    seen = {}
    for spec_id, spec in fp.ALL_SPECS.items():
        g = geometry_for_spec(spec)
        key = tuple(sorted(g.items()))
        seen.setdefault(key, []).append(spec_id)
    for key, ids in seen.items():
        g = dict(key)
        print("  %.1fx%.1f-%.0f rim %.1f  C2000=%d   %s"
              % (g["overall_dia_in"], g["section_width_in"], g["rim_dia_in"],
                 g["rim_width_in"], g["compound_c2000"], ", ".join(ids)))
    print("\n%d distinct geometries across %d specs\n"
          % (len(seen), len(fp.ALL_SPECS)))

    print("building caches\n")
    build_all_caches()
