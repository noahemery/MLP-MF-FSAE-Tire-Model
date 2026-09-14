"""Config-driven driver for the 18 fitting blocks in magic.py.

magic.py holds 18 blocks at module level, written to be run one spec at a time
by hand. This module turns them into data plus four runners, so every spec can
be fit without retyping the loop body.

NOTHING NUMERIC CHANGES. Every window, threshold_factor, ET bound, SA
threshold, bounds tuple and x0 seed below is copied verbatim from the
corresponding block in magic.py. Values identical across a whole family live as
named constants; values that vary per spec live in the config. The physics is
imported from magic.py and is never redefined here.

Each spec carries the magic.py line number of the block it came from, so the
literals can be diffed against the source by hand.
"""

import numpy as np
import scipy.io
from dataclasses import dataclass
from scipy.optimize import least_squares

from magic import (
    sort, bound,
    first_pass_y, second_pass_y,
    first_pass_x, second_pass_x,
    first_pass_GX, second_pass_GX,
    first_pass_GY, second_pass_GY,
    tm_lat,                      # MX needs F_y; see _mx_columns
)

# Deterministic evaluation cap for the second passes. See progress.install().
# magic.py asks for max_nfev=1e8 with tolerances at 2.3e-16 (below double
# epsilon), which does not terminate: 12,250 evaluations on the live lateral
# spec gave 8 distinct cost values and a 0.021% reduction. The cap only ever
# lowers max_nfev, so magic.py's own 1e2 on the G_y second passes survives.
CAP_NFEV = 50_000

CORNERING = "data/cornering_SI/B2356raw{}.mat"
STRAIGHT = "data/straight_SI/B2356raw{}.mat"

# ---------------------------------------------------------------------------
# Family constants: literals identical in every block of that family.
# ---------------------------------------------------------------------------

# Solver settings, identical in all 18 blocks.
TOL = 2.3e-16
MAX_NFEV = int(1e+8)
JAC = '3-point'
VERBOSE = 1

# Lateral first pass, identical in all 6 lateral blocks (magic.py:642-652).
LAT_X0_BCDE_C0 = 1.45
LAT_X0_BCDE_D = 500
LAT_D_MARGIN = 100
LAT_SVY_BOUND = 100
LAT_LOWER = [0, 1, 0, 0, -0.075, -LAT_SVY_BOUND]

# straight_SI segmentation, identical in all 12 straight-file blocks
# (magic.py:790-793, 1166-1169, 1465-1468).
STRAIGHT_SORT_WINDOW = 100
STRAIGHT_SORT_TF = 10
STRAIGHT_BOUND_TF = 1
STRAIGHT_ET_LO = 5
STRAIGHT_ET_HI = 15

# Longitudinal pure slip keeps segments BELOW this mean|SA| (magic.py:793).
LONG_SA_MAX = 1e-1
# Longitudinal first pass, non-varying entries (magic.py:806-812).
LONG_X0_BCDE_C0 = 1.45
LONG_X0_BCDE_D = 500
LONG_B_LOWER, LONG_B_UPPER = 0, 20
LONG_C_LOWER, LONG_C_UPPER = 1.2, 2

# G_x first pass (magic.py:1182-1192).
GX_X0_BCES_B = 10
GX_X0_BCES_C0 = 1.65
GX_X0_BCES_E = 0.75
GX_LOWER = [0, 1, 0, -0.075]
GX_UPPER = [30, 2, 1, 0.075]
GX_FTOL = None                        # magic.py:1191 -- None here, not TOL

# G_y first pass (magic.py:1622-1632).
GY_X0_BCES_B = 10
GY_X0_BCES_C0 = 1.65
GY_X0_BCES_E = 0.75
GY_LOWER = [0, 1, 0, -0.075, -500]
GY_UPPER = [30, 2, 1, 0.075, 500]
GY_FTOL = None                        # magic.py:1631
# magic.py:1502, :1549, :1596, :1643 -- all four G_y second passes, not just
# the live one, cap at 1e2 where every other second pass uses 1e8.
GY_SECOND_PASS_MAX_NFEV = int(1e+2)


# ---------------------------------------------------------------------------
# Second-pass x0 seeds.
#
# Held as data rather than inline literals so audit_literals.py can check them
# against magic.py's source. Every entry is a literal except the one or two
# taken from a first-pass column mean, which stand in as ColMean placeholders.
# ---------------------------------------------------------------------------

class _ColMean:
    """An x0 entry taken from the mean of a first-pass parameter column."""
    __slots__ = ("col",)

    def __init__(self, col):
        self.col = col

    def __repr__(self):
        return "ColMean(" + str(self.col) + ")"


MEAN_C = _ColMean(1)      # BCDE_params[:,1].mean() / BCES_params[:,1].mean()
MEAN_SH = _ColMean(3)     # BCES_params[:,3].mean() -- G_x only


def seed(template, first_pass_params):
    """Resolve a template into a concrete x0 list."""
    return [first_pass_params[:, v.col].mean() if isinstance(v, _ColMean) else v
            for v in template]


# magic.py:656
LAT_X0_P = [1, 0, 0, MEAN_C, 10, 1.5, 0, 2, 0, 2.5, 0, 0, 0, -1, 0, 0, 0, 0,
            0, 0, 0.15, 0]
# magic.py:820
LONG_X0_P = [1, 0, MEAN_C, 12, 10, -0.6, 0, 0, -0.5, 0, 0, 0, 0, 0]
# magic.py:1196
GX_X0_R = [10, 8, 0, MEAN_C, 0, 0, MEAN_SH]
# magic.py:1636
GY_X0_R = [7, 2.5, 0, 0, MEAN_C, 0, 0, 0.02, 0, 0, 0, -0.2, 14, 1.9, 10]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FileSpec:
    """One raw file plus the segmentation literals applied to it."""
    path: str
    sort_window: int
    sort_tf: float
    bound_tf: float
    et_lo: float          # strict >
    et_hi: float          # strict <


@dataclass(frozen=True)
class LateralSpec:
    spec_id: str
    block_line: int
    fz0_files: tuple      # NOT always the same set as case_files
    case_files: tuple     # of FileSpec
    diameter_in: float = None   # William stores this before F_z0 in the vector
    family: str = "lateral"
    depends_on: str = None


@dataclass(frozen=True)
class LongitudinalSpec:
    spec_id: str
    block_line: int
    fz0_files: tuple
    case_files: tuple     # of str; segmentation literals are family constants
    d_margin: float       # the "+ N" in the D upper bound
    svx_bound: float      # the S_vx bound magnitude
    diameter_in: float = None   # William stores this before F_z0 in the vector
    family: str = "longitudinal"
    depends_on: str = None


@dataclass(frozen=True)
class CombinedSpec:
    spec_id: str
    block_line: int
    case_files: tuple
    sa_threshold: float   # keep segments with mean|SA| ABOVE this
    depends_on: str       # long_* for G_x, lat_* for G_y
    family: str           # 'gx' or 'gy'


LATERAL_SPECS = (
    LateralSpec(
        spec_id="lat_160X75_R20_70", diameter_in=16.0, block_line=349,
        fz0_files=(CORNERING.format(4), CORNERING.format(5), CORNERING.format(6)),
        case_files=(
            FileSpec(CORNERING.format(4), 200, 30, 0.86, 8, 12),
            FileSpec(CORNERING.format(5), 250, 60, 1, 8, 12),
            FileSpec(CORNERING.format(6), 250, 50, 1, 10, 13),
        ),
    ),
    LateralSpec(
        spec_id="lat_160X75_R20_80", diameter_in=16.0, block_line=415,
        # magic.py:416-418 stacks these in the order 8, 7, 9. Preserved.
        fz0_files=(CORNERING.format(8), CORNERING.format(7), CORNERING.format(9)),
        case_files=(
            FileSpec(CORNERING.format(8), 130, 90, 0.86, 6, 12),
            FileSpec(CORNERING.format(9), 130, 100, 0.86, 6, 12),
        ),
    ),
    LateralSpec(
        spec_id="lat_205X70_R20_70", diameter_in=20.5, block_line=465,
        fz0_files=(CORNERING.format(16), CORNERING.format(17), CORNERING.format(18)),
        case_files=(
            FileSpec(CORNERING.format(17), 130, 120, 0.86, 6, 12),
            FileSpec(CORNERING.format(18), 130, 90, 0.86, 6, 12),
        ),
    ),
    LateralSpec(
        spec_id="lat_205X70_R20_80", diameter_in=20.5, block_line=516,
        fz0_files=(CORNERING.format(19), CORNERING.format(20), CORNERING.format(21)),
        case_files=(
            FileSpec(CORNERING.format(20), 130, 90, 0.86, 6, 12),
            FileSpec(CORNERING.format(21), 130, 90, 0.86, 6, 12),
        ),
    ),
    LateralSpec(
        spec_id="lat_180X60_R20_60", diameter_in=18.0, block_line=567,
        fz0_files=(CORNERING.format(27), CORNERING.format(28), CORNERING.format(29)),
        case_files=(
            FileSpec(CORNERING.format(28), 190, 20, 0.86, 9, 12),
            FileSpec(CORNERING.format(29), 110, 18, 0.89, 10, 12),
        ),
    ),
    LateralSpec(
        spec_id="lat_180X60_R20_70", diameter_in=18.0, block_line=619,          # the live block
        fz0_files=(CORNERING.format(30), CORNERING.format(31), CORNERING.format(32)),
        case_files=(
            FileSpec(CORNERING.format(31), 110, 18, 0.89, 10, 12),
            FileSpec(CORNERING.format(32), 110, 18, 0.89, 10, 12),
        ),
    ),
)

LONGITUDINAL_SPECS = (
    LongitudinalSpec(
        spec_id="long_205X70_R20_70", diameter_in=20.5, block_line=783,
        fz0_files=(STRAIGHT.format(51), STRAIGHT.format(52)),
        case_files=(STRAIGHT.format(51), STRAIGHT.format(52)),
        d_margin=200, svx_bound=200,
    ),
    LongitudinalSpec(
        spec_id="long_205X70_R20_80", diameter_in=20.5, block_line=833,
        fz0_files=(STRAIGHT.format(54), STRAIGHT.format(55)),
        case_files=(STRAIGHT.format(54), STRAIGHT.format(55)),
        # magic.py:861-862 -- margin 100 but bound 150. The only spec where
        # these two differ from each other. Carried over verbatim.
        d_margin=100, svx_bound=150,
    ),
    LongitudinalSpec(
        spec_id="long_180X60_R20_60", diameter_in=18.0, block_line=883,
        fz0_files=(STRAIGHT.format(69), STRAIGHT.format(70)),
        case_files=(STRAIGHT.format(69), STRAIGHT.format(70)),
        d_margin=150, svx_bound=150,
    ),
    LongitudinalSpec(
        spec_id="long_180X60_R20_70", diameter_in=18.0, block_line=933,
        fz0_files=(STRAIGHT.format(72), STRAIGHT.format(73)),
        case_files=(STRAIGHT.format(72), STRAIGHT.format(73)),
        d_margin=50, svx_bound=50,
    ),
)

# The four straight-run pairs, shared by the longitudinal, G_x and G_y blocks.
_STRAIGHT_PAIRS = {
    "205X70_R20_70": (STRAIGHT.format(51), STRAIGHT.format(52)),
    "205X70_R20_80": (STRAIGHT.format(54), STRAIGHT.format(55)),
    "180X60_R20_60": (STRAIGHT.format(69), STRAIGHT.format(70)),
    "180X60_R20_70": (STRAIGHT.format(72), STRAIGHT.format(73)),
}

GX_SPECS = tuple(
    CombinedSpec(spec_id="GX_" + tag, block_line=line,
                 case_files=_STRAIGHT_PAIRS[tag], sa_threshold=0.5,
                 depends_on="long_" + tag, family="gx")
    for tag, line in (("205X70_R20_70", 1162), ("205X70_R20_80", 1209),
                      ("180X60_R20_60", 1256), ("180X60_R20_70", 1306))
)

GY_SPECS = tuple(
    CombinedSpec(spec_id="GY_" + tag, block_line=line,
                 case_files=_STRAIGHT_PAIRS[tag], sa_threshold=0.8,
                 depends_on="lat_" + tag, family="gy")
    for tag, line in (("205X70_R20_70", 1461), ("205X70_R20_80", 1508),
                      ("180X60_R20_60", 1555), ("180X60_R20_70", 1602))
)

# ---------------------------------------------------------------------------
# Overturning moment, MX.
#
# magic.py has NO fitting block for any moment -- fit_MX is defined and never
# called, and no x0 seed or bound exists for it anywhere. The two specs below
# are therefore NOT transcribed from magic.py; they are new, written 2026-09-14
# under the owner's authorisation to fix the moment code as needed. block_line
# is None to say so, and audit_literals.py skips them for that reason.
#
# Three facts make this family much cheaper than the force families:
#
# 1. fit_MX's residual is LINEAR in QSX1/QSX2/QSX3, so np.linalg.lstsq gives
#    the global optimum outright. No seed, no bounds, no convergence question.
#    least_squares is still run, from the closed-form point, purely so the
#    worker gets a real scipy result with a real Jacobian for the diagnostics
#    -- it terminates in single-digit evaluations.
#
# 2. MX reuses the LATERAL segmentation exactly, so there are no new
#    segmentation literals. The spec just names the lateral spec it borrows.
#
# 3. MX needs gy_params only through F_y = Y * G_y, and on cornering data SL is
#    identically 0, where G_y is identically 1 (u = B*(s + S_h) collapses to
#    u0, so the ratio is 1 for ANY R-params). So MX covers all 6 tires, not
#    just the 4 with straight-line data. run_mx asserts both facts rather than
#    trusting them.
MX_P_NOM = 82.7          # kPa, ~12 psi, the modal TTC Round 9 set point
MX_P_MIN = 1.0           # P is gated to exactly 0.0 in places, like SL


@dataclass(frozen=True)
class MXSpec:
    spec_id: str
    depends_on: str        # the lat_* spec supplying params, diameter and F_z0
    lat_source: str        # the lat_* spec supplying the segmentation
    family: str            # 'mx' (canonical) or 'mxp' (pressure-extended)
    block_line: int = None       # None: not transcribed from magic.py
    diameter_in: float = None


_MX_TIRES = ("160X75_R20_70", "160X75_R20_80", "205X70_R20_70",
             "205X70_R20_80", "180X60_R20_60", "180X60_R20_70")

MX_SPECS = tuple(
    MXSpec(spec_id="MX_" + tag, depends_on="lat_" + tag,
           lat_source="lat_" + tag, family="mx")
    for tag in _MX_TIRES
)

# Pressure-extended variant. This is an EXTENSION, not magic.py's model, and it
# is kept out of magic.py deliberately -- the owner owns that file's physics.
# Each of the three terms gains a linear pressure coefficient, which is what
# MF 6.x does:
#   M_x = F_z*R_0*[ (QSX1 + QSX1p*dpi) - (QSX2 + QSX2p*dpi)*gamma
#                  +(QSX3 + QSX3p*dpi)*F_y/F_z0 ],  dpi = (P - P_nom)/P_nom
# Still linear, so still a closed-form solve.
MXP_SPECS = tuple(
    MXSpec(spec_id="MXP_" + tag, depends_on="lat_" + tag,
           lat_source="lat_" + tag, family="mxp")
    for tag in _MX_TIRES
)

ALL_SPECS = {s.spec_id: s for s in
             LATERAL_SPECS + LONGITUDINAL_SPECS + GX_SPECS + GY_SPECS
             + MX_SPECS + MXP_SPECS}

# Parameter names in magic.py's own unpack order.
#   lateral      second_pass_y:255-276  + trailing F_z0 from the hstack
#   longitudinal second_pass_x:706-719  + trailing F_z0 from the hstack
#   gx           second_pass_GX:1115-1121   (bare result.x -- see DECISIONS)
#   gy           second_pass_GY:1395-1409   (bare result.x)
# NOTE: William's 2026-09 magic.py appends tire diameter before F_z0 on the
# pure-slip families, e.g. np.hstack((result.x, 16.0, F_z0)). Lateral vectors
# are therefore 24 long and longitudinal 16. tm_lat/tm_long are unaffected --
# they index x[0]..x[21] and x[-1] -- but anything reading the vector by
# length or slicing off the tail must account for the extra element.
# Use strip_geometry() to recover the [params..., F_z0] form.
PARAM_NAMES = {
    "lateral": ["PDY1", "PDY2", "PDY3", "PCY1", "PKY1", "PKY2", "PKY3", "PKY4",
                "PKY5", "PKY6", "PKY7", "PHY1", "PHY2", "PEY1", "PEY2", "PEY3",
                "PEY4", "PEY5", "PVY1", "PVY2", "PVY3", "PVY4",
                "diameter_in", "F_z0"],
    "longitudinal": ["PDX1", "PDX2", "PCX1", "PKX1", "PKX2", "PKX3", "PHX1",
                     "PHX2", "PEX1", "PEX2", "PEX3", "PEX4", "PVX1", "PVX2",
                     "diameter_in", "F_z0"],
    "gx": ["RBX1", "RBX2", "RBX3", "RCX1", "REX1", "REX2", "RHX1"],
    "gy": ["RBY1", "RBY2", "RBY3", "RBY4", "RCY1", "REY1", "REY2", "RHY1",
           "RHY2", "RVY1", "RVY2", "RVY3", "RVY4", "RVY5", "RVY6"],
    # fit_MX:1735-1737. diameter and F_z0 carried in the same layout as the
    # pure-slip families so tire_predict can read R_0 and F_z0 off the vector.
    "mx": ["QSX1", "QSX2", "QSX3", "diameter_in", "F_z0"],
    # Extension, not magic.py's model. The *p entries are the linear pressure
    # coefficients; P_nom is metadata, not fitted.
    "mxp": ["QSX1", "QSX2", "QSX3", "QSX1p", "QSX2p", "QSX3p",
            "diameter_in", "F_z0", "P_nom"],
}


# Fitted-parameter count per family, i.e. what least_squares actually solves
# for. Everything after this in a stored vector is metadata (diameter, F_z0).
N_FITTED = {"lateral": 22, "longitudinal": 14, "gx": 7, "gy": 15,
            "mx": 3, "mxp": 6}


def strip_geometry(vector, family):
    """Return [fitted params..., F_z0], dropping any geometry metadata.

    William's vectors are [params..., diameter, F_z0] for the pure-slip
    families and bare params for the G families. tm_lat and tm_long expect
    F_z0 as the LAST element, so the diameter has to come out rather than be
    sliced past.
    """
    import numpy as _np
    v = _np.asarray(vector).ravel()
    n = N_FITTED[family]
    if family in ("gx", "gy"):
        return v[:n]
    if family == "mx":
        return _np.concatenate([v[:n], v[-1:]])          # [.., diameter, F_z0]
    if family == "mxp":
        return _np.concatenate([v[:n], v[-2:-1]])        # trailing P_nom skipped
    return _np.concatenate([v[:n], v[-1:]])


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def _et_span(seg):
    return seg["ET"].max() - seg["ET"].min()


def compute_fz0(paths):
    """magic.py:620-622 -- mean of |FZ| stacked across the listed files."""
    return np.abs(np.vstack([scipy.io.loadmat(p)["FZ"] for p in paths])).mean()


def build_cases_lateral(spec):
    """magic.py:626-636 -- per-file sort/bound literals and ET window."""
    cases = []
    for fs in spec.case_files:
        split_raw = sort(scipy.io.loadmat(fs.path), load_key="FZ",
                         window=fs.sort_window, threshold_factor=fs.sort_tf)
        split = bound(split_raw, threshold_factor=fs.bound_tf)
        for i in range(len(split)):
            if (_et_span(split[i]) < fs.et_hi) & (_et_span(split[i]) > fs.et_lo):
                cases.append(split[i])
    return cases


# The SL channel is gated to exactly 0.0 between sweeps while FX keeps
# recording -- 53-59% of every straight file, with up to 4463 N present at
# those samples. They are not slip-ratio measurements, and at sweep boundaries
# the force has not decayed yet, so they dominate the residual: 2.4% of points
# caused 8.9% of squared error. Excluding them took long_205X70 from 27.1% to
# 6.8%. Owner authorised adjusting case selection 2026-09-09.
DROP_SL_ZERO = True


def build_cases_straight(case_files, sa_test):
    """magic.py:790-800 / 1166-1176 / 1465-1475.

    Segmentation is identical for all 12 straight-file blocks; only the mean|SA|
    test differs -- below 1e-1 for pure longitudinal slip, above 0.5 for G_x,
    above 0.8 for G_y.
    """
    cases = []
    for path in case_files:
        split_raw = sort(scipy.io.loadmat(path), load_key="FZ",
                         window=STRAIGHT_SORT_WINDOW,
                         threshold_factor=STRAIGHT_SORT_TF)
        split = bound(split_raw, slip_key="SL", threshold_factor=STRAIGHT_BOUND_TF)
        for i in range(len(split)):
            if ((_et_span(split[i]) < STRAIGHT_ET_HI)
                    & (_et_span(split[i]) > STRAIGHT_ET_LO)
                    & sa_test(np.abs(split[i]["SA"]).mean())):
                seg = split[i]
                if DROP_SL_ZERO:
                    keep = np.asarray(seg["SL"]).reshape(-1) != 0.0
                    if keep.sum() < 50:
                        continue
                    seg = {k: (v[keep] if hasattr(v, "shape")
                               and getattr(v, "shape", (0,))[:1] == keep.shape
                               else v) for k, v in seg.items()}
                cases.append(seg)
    return cases


def build_cases(spec):
    """Segmentation only, no fitting. Used by the equivalence check."""
    if spec.family == "lateral":
        return build_cases_lateral(spec)
    if spec.family == "longitudinal":
        return build_cases_straight(spec.case_files, lambda m: m < LONG_SA_MAX)
    return build_cases_straight(spec.case_files,
                                lambda m, t=spec.sa_threshold: m > t)


# ---------------------------------------------------------------------------
# Runners -- each a transcription of its block body in magic.py
# ---------------------------------------------------------------------------

def run_lateral(spec):
    """magic.py:619-666."""
    F_z0 = compute_fz0(spec.fz0_files)
    cases = build_cases_lateral(spec)

    BCDE_params = np.zeros((len(cases), 6))
    for i in range(len(cases)):
        if i == 0:
            x0_BCDE = [0, LAT_X0_BCDE_C0, LAT_X0_BCDE_D, 0, 0, 0]
        else:
            x0_BCDE = [0, BCDE_params[i - 1, 1], LAT_X0_BCDE_D, 0, 0, 0]
        fit_func = lambda x: (first_pass_y(cases[i], x))
        result = least_squares(fit_func, x0_BCDE, jac=JAC, method='trf',
                               bounds=(LAT_LOWER,
                                       [30, 2,
                                        LAT_D_MARGIN + np.abs(cases[i]["FY"]).max(),
                                        1, 0.075, LAT_SVY_BOUND]),
                               ftol=TOL, xtol=TOL, gtol=TOL,
                               max_nfev=MAX_NFEV, verbose=VERBOSE)
        BCDE_params[i] = result.x

    x0_P = seed(LAT_X0_P, BCDE_params)

    fit_func = lambda x: (second_pass_y(cases, F_z0, 1, BCDE_params, x))
    result = least_squares(fit_func, x0_P, jac=JAC, method='lm',
                           ftol=TOL, xtol=TOL, gtol=TOL,
                           max_nfev=MAX_NFEV, verbose=VERBOSE)

    return (np.hstack((result.x, spec.diameter_in, F_z0)),
            result, cases, F_z0)


def run_longitudinal(spec):
    """magic.py:783-829."""
    F_z0 = compute_fz0(spec.fz0_files)
    cases = build_cases_straight(spec.case_files, lambda m: m < LONG_SA_MAX)

    BCDE_params = np.zeros((len(cases), 6))
    for i in range(len(cases)):
        if i == 0:
            x0_BCDE = [0, LONG_X0_BCDE_C0, LONG_X0_BCDE_D, 0, 0, 0]
        else:
            x0_BCDE = [0, BCDE_params[i - 1, 1], LONG_X0_BCDE_D, 0, 0, 0]
        fit_func = lambda x: (first_pass_x(cases[i], x))
        result = least_squares(fit_func, x0_BCDE, jac=JAC, method='trf',
                               bounds=([LONG_B_LOWER, LONG_C_LOWER, 0, 0,
                                        -0.075, -spec.svx_bound],
                                       [LONG_B_UPPER, LONG_C_UPPER,
                                        spec.d_margin + np.abs(cases[i]["FX"]).max(),
                                        1, 0.075, spec.svx_bound]),
                               ftol=TOL, xtol=TOL, gtol=TOL,
                               max_nfev=MAX_NFEV, verbose=VERBOSE)
        BCDE_params[i] = result.x

    x0_P = seed(LONG_X0_P, BCDE_params)

    fit_func = lambda x: (second_pass_x(cases, F_z0, 1, BCDE_params, x))
    result = least_squares(fit_func, x0_P, jac=JAC, method='lm',
                           ftol=TOL, xtol=TOL, gtol=TOL,
                           max_nfev=MAX_NFEV, verbose=VERBOSE)

    return (np.hstack((result.x, spec.diameter_in, F_z0)),
            result, cases, F_z0)


def run_gx(spec, long_params):
    """magic.py:1162-1205.

    F_z0 for the second pass comes from long_params[-1], the straight-run value
    the longitudinal fit stored -- not from a locally computed one.
    """
    cases = build_cases_straight(spec.case_files,
                                 lambda m, t=spec.sa_threshold: m > t)

    BCES_params = np.zeros((len(cases), 4))
    for i in range(len(cases)):
        if i == 0:
            x0_BCES = [GX_X0_BCES_B, GX_X0_BCES_C0, GX_X0_BCES_E, 0]
        else:
            x0_BCES = [GX_X0_BCES_B, BCES_params[i - 1, 1], GX_X0_BCES_E, 0]
        fit_func = lambda x: (first_pass_GX(cases[i], x, long_params))
        result = least_squares(fit_func, x0_BCES, jac=JAC, method='trf',
                               bounds=(GX_LOWER, GX_UPPER),
                               ftol=GX_FTOL, xtol=TOL, gtol=TOL,
                               max_nfev=MAX_NFEV, verbose=VERBOSE)
        BCES_params[i] = result.x

    x0_R = seed(GX_X0_R, BCES_params)

    F_z0 = long_params[-1]
    fit_func = lambda x: (second_pass_GX(cases, F_z0, 1, BCES_params, x))
    result = least_squares(fit_func, x0_R, jac=JAC, method='lm',
                           ftol=TOL, xtol=TOL, gtol=TOL,
                           max_nfev=MAX_NFEV, verbose=VERBOSE)

    # Bare result.x for all four G_x specs. magic.py:1205/:1252 store bare
    # result.x while :1302/:1349 hstack an F_z0 -- and :1349's is a leaked
    # module global from the previous block, i.e. the wrong spec. F_z0 is
    # recorded in the diagnostics table instead. See DECISIONS finding B.
    return result.x, result, cases, F_z0


def run_gy(spec, lat_params):
    """magic.py:1602-1645.

    F_z0 for the second pass comes from lat_params[-1] -- the CORNERING-run
    value -- even though the cases are straight_SI. Verbatim from magic.py.
    """
    cases = build_cases_straight(spec.case_files,
                                 lambda m, t=spec.sa_threshold: m > t)

    BCES_params = np.zeros((len(cases), 5))
    for i in range(len(cases)):
        if i == 0:
            x0_BCES = [GY_X0_BCES_B, GY_X0_BCES_C0, GY_X0_BCES_E, 0, 0]
        else:
            x0_BCES = [GY_X0_BCES_B, BCES_params[i - 1, 1], GY_X0_BCES_E, 0, 0]
        fit_func = lambda x: (first_pass_GY(cases[i], x, lat_params))
        result = least_squares(fit_func, x0_BCES, jac=JAC, method='trf',
                               bounds=(GY_LOWER, GY_UPPER),
                               ftol=GY_FTOL, xtol=TOL, gtol=TOL,
                               max_nfev=MAX_NFEV, verbose=VERBOSE)
        BCES_params[i] = result.x

    x0_R = seed(GY_X0_R, BCES_params)

    F_z0 = lat_params[-1]
    fit_func = lambda x: (second_pass_GY(cases, F_z0, 1, BCES_params, x,
                                         lat_params))
    result = least_squares(fit_func, x0_R, jac=JAC, method='lm',
                           ftol=TOL, xtol=TOL, gtol=TOL,
                           max_nfev=GY_SECOND_PASS_MAX_NFEV, verbose=VERBOSE)

    return result.x, result, cases, F_z0


def _mx_columns(cases, lat_params, with_pressure):
    """Design matrix for fit_MX, which is linear in QSX1/QSX2/QSX3.

    Returns (A, y, groups). groups is the segment index per row, so held-out
    error can be scored by GroupKFold without re-deriving the segmentation.
    """
    F_z0 = lat_params[-1]
    R_0 = lat_params[-2] * 0.5 * 0.0254          # in -> m, as fit_MX:1740
    core = strip_geometry(lat_params, "lateral")

    A_all, y_all, g_all = [], [], []
    for gi, case in enumerate(cases):
        F_z = -np.asarray(case["FZ"]).ravel()
        gamma = np.sin(np.asarray(case["IA"]).ravel() * np.pi / 180)
        alpha = np.tan(np.asarray(case["SA"]).ravel() * np.pi / 180)
        M_x = np.asarray(case["MX"]).ravel()
        P = np.asarray(case["P"]).ravel()

        # G_y is identically 1 here (asserted by the caller), so F_y is the
        # bare tm_lat output and no gy_params are needed.
        F_y = tm_lat(F_z, alpha, gamma, 1, core)[0].ravel()

        A = np.column_stack([F_z * R_0,
                             -F_z * R_0 * gamma,
                             F_z * R_0 * F_y / F_z0])
        keep = np.ones(len(M_x), dtype=bool)
        if with_pressure:
            keep = P > MX_P_MIN          # P is gated to 0.0 in places, like SL
            dpi = (P - MX_P_NOM) / MX_P_NOM
            A = np.column_stack([A, A * dpi[:, None]])
        A_all.append(A[keep])
        y_all.append(M_x[keep])
        g_all.append(np.full(int(keep.sum()), gi))

    return (np.vstack(A_all), np.concatenate(y_all), np.concatenate(g_all))


def _assert_pure_slip(cases, spec_id):
    """MX's independence from gy_params rests on SL being identically 0."""
    sl = np.concatenate([np.asarray(c["SL"]).ravel() for c in cases])
    worst = float(np.abs(sl).max())
    if worst != 0.0:
        raise RuntimeError(
            spec_id + ": SL is not identically 0 (max|SL| = " + repr(worst) +
            "). G_y is only identically 1 at SL == 0, so this fit would "
            "silently depend on gy_params. Refusing to proceed.")


def _run_mx_family(spec, lat_params, with_pressure):
    lat_spec = ALL_SPECS[spec.lat_source]
    cases = build_cases_lateral(lat_spec)
    _assert_pure_slip(cases, spec.spec_id)

    A, y, groups = _mx_columns(cases, lat_params, with_pressure)

    # Closed form: the residual is linear, so this IS the global optimum.
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)

    # Re-run through least_squares from that point so the worker gets a real
    # scipy result -- status, Jacobian, optimality, parameter covariance. It
    # starts at the optimum, so it terminates in single-digit evaluations.
    result = least_squares(lambda x: A @ x - y, beta, jac=lambda x: A,
                           method='lm', ftol=TOL, xtol=TOL, gtol=TOL,
                           max_nfev=MAX_NFEV, verbose=VERBOSE)

    tail = [lat_params[-2], lat_params[-1]]
    if with_pressure:
        tail.append(MX_P_NOM)
    return (np.hstack((result.x, tail)), result, cases, float(lat_params[-1]))


def run_mx(spec, lat_params):
    """Canonical MX -- exactly fit_MX's model, 3 parameters.

    Equivalence to magic.fit_MX is proven by verify_mx.py, which checks that
    magic.fit_MX's own residual at this solution matches the linear system and
    that least_squares on magic.fit_MX from a neutral seed lands on the same
    point. It is not re-checked per run because fit_MX evaluates tm_lat and GY
    over every row on every call.
    """
    return _run_mx_family(spec, lat_params, with_pressure=False)


def run_mxp(spec, lat_params):
    """Pressure-extended MX. An EXTENSION, not magic.py's model."""
    return _run_mx_family(spec, lat_params, with_pressure=True)


def run_spec(spec, source_vector=None):
    """Dispatch. source_vector is the long_*/lat_* vector for G_x/G_y/MX."""
    if spec.family == "lateral":
        return run_lateral(spec)
    if spec.family == "longitudinal":
        return run_longitudinal(spec)
    if spec.family == "gx":
        return run_gx(spec, source_vector)
    if spec.family == "gy":
        return run_gy(spec, source_vector)
    if spec.family == "mx":
        return run_mx(spec, source_vector)
    if spec.family == "mxp":
        return run_mxp(spec, source_vector)
    raise ValueError("unknown family: " + str(spec.family))
