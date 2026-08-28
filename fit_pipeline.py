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
        spec_id="lat_160X75_R20_70", block_line=349,
        fz0_files=(CORNERING.format(4), CORNERING.format(5), CORNERING.format(6)),
        case_files=(
            FileSpec(CORNERING.format(4), 200, 30, 0.86, 8, 12),
            FileSpec(CORNERING.format(5), 250, 60, 1, 8, 12),
            FileSpec(CORNERING.format(6), 250, 50, 1, 10, 13),
        ),
    ),
    LateralSpec(
        spec_id="lat_160X75_R20_80", block_line=415,
        # magic.py:416-418 stacks these in the order 8, 7, 9. Preserved.
        fz0_files=(CORNERING.format(8), CORNERING.format(7), CORNERING.format(9)),
        case_files=(
            FileSpec(CORNERING.format(8), 130, 90, 0.86, 6, 12),
            FileSpec(CORNERING.format(9), 130, 100, 0.86, 6, 12),
        ),
    ),
    LateralSpec(
        spec_id="lat_205X70_R20_70", block_line=465,
        fz0_files=(CORNERING.format(16), CORNERING.format(17), CORNERING.format(18)),
        case_files=(
            FileSpec(CORNERING.format(17), 130, 120, 0.86, 6, 12),
            FileSpec(CORNERING.format(18), 130, 90, 0.86, 6, 12),
        ),
    ),
    LateralSpec(
        spec_id="lat_205X70_R20_80", block_line=516,
        fz0_files=(CORNERING.format(19), CORNERING.format(20), CORNERING.format(21)),
        case_files=(
            FileSpec(CORNERING.format(20), 130, 90, 0.86, 6, 12),
            FileSpec(CORNERING.format(21), 130, 90, 0.86, 6, 12),
        ),
    ),
    LateralSpec(
        spec_id="lat_180X60_R20_60", block_line=567,
        fz0_files=(CORNERING.format(27), CORNERING.format(28), CORNERING.format(29)),
        case_files=(
            FileSpec(CORNERING.format(28), 190, 20, 0.86, 9, 12),
            FileSpec(CORNERING.format(29), 110, 18, 0.89, 10, 12),
        ),
    ),
    LateralSpec(
        spec_id="lat_180X60_R20_70", block_line=619,          # the live block
        fz0_files=(CORNERING.format(30), CORNERING.format(31), CORNERING.format(32)),
        case_files=(
            FileSpec(CORNERING.format(31), 110, 18, 0.89, 10, 12),
            FileSpec(CORNERING.format(32), 110, 18, 0.89, 10, 12),
        ),
    ),
)

LONGITUDINAL_SPECS = (
    LongitudinalSpec(
        spec_id="long_205X70_R20_70", block_line=783,
        fz0_files=(STRAIGHT.format(51), STRAIGHT.format(52)),
        case_files=(STRAIGHT.format(51), STRAIGHT.format(52)),
        d_margin=200, svx_bound=200,
    ),
    LongitudinalSpec(
        spec_id="long_205X70_R20_80", block_line=833,
        fz0_files=(STRAIGHT.format(54), STRAIGHT.format(55)),
        case_files=(STRAIGHT.format(54), STRAIGHT.format(55)),
        # magic.py:861-862 -- margin 100 but bound 150. The only spec where
        # these two differ from each other. Carried over verbatim.
        d_margin=100, svx_bound=150,
    ),
    LongitudinalSpec(
        spec_id="long_180X60_R20_60", block_line=883,
        fz0_files=(STRAIGHT.format(69), STRAIGHT.format(70)),
        case_files=(STRAIGHT.format(69), STRAIGHT.format(70)),
        d_margin=150, svx_bound=150,
    ),
    LongitudinalSpec(
        spec_id="long_180X60_R20_70", block_line=933,
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

ALL_SPECS = {s.spec_id: s for s in
             LATERAL_SPECS + LONGITUDINAL_SPECS + GX_SPECS + GY_SPECS}

# Parameter names in magic.py's own unpack order.
#   lateral      second_pass_y:255-276  + trailing F_z0 from the hstack
#   longitudinal second_pass_x:706-719  + trailing F_z0 from the hstack
#   gx           second_pass_GX:1115-1121   (bare result.x -- see DECISIONS)
#   gy           second_pass_GY:1395-1409   (bare result.x)
PARAM_NAMES = {
    "lateral": ["PDY1", "PDY2", "PDY3", "PCY1", "PKY1", "PKY2", "PKY3", "PKY4",
                "PKY5", "PKY6", "PKY7", "PHY1", "PHY2", "PEY1", "PEY2", "PEY3",
                "PEY4", "PEY5", "PVY1", "PVY2", "PVY3", "PVY4", "F_z0"],
    "longitudinal": ["PDX1", "PDX2", "PCX1", "PKX1", "PKX2", "PKX3", "PHX1",
                     "PHX2", "PEX1", "PEX2", "PEX3", "PEX4", "PVX1", "PVX2",
                     "F_z0"],
    "gx": ["RBX1", "RBX2", "RBX3", "RCX1", "REX1", "REX2", "RHX1"],
    "gy": ["RBY1", "RBY2", "RBY3", "RBY4", "RCY1", "REY1", "REY2", "RHY1",
           "RHY2", "RVY1", "RVY2", "RVY3", "RVY4", "RVY5", "RVY6"],
}


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
                cases.append(split[i])
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

    return np.hstack((result.x, F_z0)), result, cases, F_z0


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

    return np.hstack((result.x, F_z0)), result, cases, F_z0


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


def run_spec(spec, source_vector=None):
    """Dispatch. source_vector is the long_*/lat_* vector for G_x/G_y."""
    if spec.family == "lateral":
        return run_lateral(spec)
    if spec.family == "longitudinal":
        return run_longitudinal(spec)
    if spec.family == "gx":
        return run_gx(spec, source_vector)
    if spec.family == "gy":
        return run_gy(spec, source_vector)
    raise ValueError("unknown family: " + str(spec.family))
