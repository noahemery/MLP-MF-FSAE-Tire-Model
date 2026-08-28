"""Progress instrumentation for long scipy.optimize.least_squares runs.

install() replaces scipy.optimize.least_squares with a wrapper that counts
residual evaluations and prints a heartbeat. The residual array is handed back
unchanged and no solver argument is altered, so fits stay numerically
identical -- this buys visibility, nothing else.

Why a monkeypatch rather than wrapping the call site: the reference run IS
magic.py's own code, executed through runpy, so there is no call site of ours
to wrap and magic.py must not be edited. magic.py does

    from scipy.optimize import least_squares

at import time, so install() must run BEFORE magic.py is imported or executed;
the name it binds is then the instrumented one, with the file untouched on
disk. Using the same mechanism for the pipeline keeps both code paths
identical, which matters when the acceptance gate is bitwise.
"""

import os
import time

import numpy as np
import scipy.optimize

_real_least_squares = scipy.optimize.least_squares
_installed = False


class _Counter:
    """Counts calls to a residual function and prints a periodic heartbeat."""

    __slots__ = ("_fun", "label", "every", "n", "t0")

    def __init__(self, fun, label, every):
        self._fun = fun
        self.label = label
        self.every = every
        self.n = 0
        self.t0 = time.time()

    def __call__(self, x, *args, **kwargs):
        r = self._fun(x, *args, **kwargs)
        self.n += 1
        if self.every and self.n % self.every == 0:
            elapsed = time.time() - self.t0
            try:
                a = np.asarray(r, dtype=float).ravel()
                cost = 0.5 * float(a @ a)
            except (TypeError, ValueError):
                cost = float("nan")
            print("    [%s] nfev=%-8d %8.1fs  %.4fs/eval  cost=%.6e"
                  % (self.label, self.n, elapsed, elapsed / self.n, cost),
                  flush=True)
        return r


def install(every=250, summary_after_s=10.0, cap_nfev=None):
    """Patch scipy.optimize.least_squares in place. Idempotent.

    every            -- heartbeat interval in residual evaluations
    summary_after_s  -- only print a completion line for calls at least this
                        long, so the many fast first-pass fits stay quiet
    cap_nfev         -- if set, clamp max_nfev to at most this

    On cap_nfev. magic.py's second passes specify max_nfev=int(1e+8) with all
    three tolerances at 2.3e-16, which is below double epsilon (2.220446e-16).
    Measured on the live lateral spec, that combination does not terminate:
    12,250 evaluations produced 8 distinct cost values and a 0.021% total
    reduction. Some bound is therefore required to produce any baseline at all.

    It has to be a max_nfev bound and NOT a wall-clock bound. A wall-clock kill
    stops the reference and the pipeline at different evaluation counts, giving
    different parameter vectors, so the bitwise acceptance gate could never
    pass. Capping evaluations is deterministic: identical cap, identical
    result.

    The clamp only ever lowers max_nfev, never raises it, so magic.py's own
    max_nfev=1e2 on all four G_y second passes is preserved. First passes
    terminate naturally in 33-82 evaluations and are unaffected.
    """
    global _installed
    if _installed:
        return

    def instrumented(fun, x0, *args, **kwargs):
        n_par = int(np.size(np.atleast_1d(x0)))
        label = "%s n=%d" % (kwargs.get("method", "trf"), n_par)

        if cap_nfev is not None:
            existing = kwargs.get("max_nfev")
            capped = cap_nfev if existing is None else min(existing, cap_nfev)
            if existing != capped:
                print("    [%s] max_nfev capped %s -> %d"
                      % (label, existing, capped), flush=True)
            kwargs["max_nfev"] = capped

        counter = _Counter(fun, label, every)
        t0 = time.time()
        result = _real_least_squares(counter, x0, *args, **kwargs)
        elapsed = time.time() - t0
        if elapsed >= summary_after_s:
            print("    [%s] DONE nfev=%d in %.1fs (%.4fs/eval) status=%s "
                  "cost=%.6e\n        %s"
                  % (label, counter.n, elapsed,
                     elapsed / max(counter.n, 1), result.status, result.cost,
                     result.message), flush=True)
        return result

    scipy.optimize.least_squares = instrumented
    _installed = True


def uninstall():
    global _installed
    scipy.optimize.least_squares = _real_least_squares
    _installed = False


# ---------------------------------------------------------------------------
# Keeping a long unattended run at full speed on Windows.
# ---------------------------------------------------------------------------

def keep_hot():
    """Opt this process out of power throttling and hold off idle standby.

    Measured on this machine: with the session idle, the fit process got
    ~1.6% of a core over 129 minutes -- 2.1 CPU-minutes of work for over two
    hours of wall clock. Setting standby-timeout-ac to 0 did NOT fix it, and
    the process was already at Normal priority on AC power, so idle standby
    was not the mechanism. Windows Power Throttling (EcoQoS), which is enabled
    by default, throttles background processes independently of standby.

    Two separate mitigations, both process-scoped and self-reverting:
      - SetProcessInformation(ProcessPowerThrottling) with ControlMask set and
        StateMask clear, which means "do not throttle execution speed".
      - SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED), so the
        system does not go idle underneath a long fit.

    Returns a dict of what actually succeeded. Never raises -- on a machine
    where these are unavailable the run should still proceed, just slowly.
    """
    result = {"throttling_disabled": False, "execution_state_set": False}
    if os.name != "nt":
        return result

    import ctypes
    from ctypes import wintypes

    class _PowerThrottlingState(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG),
                    ("ControlMask", wintypes.ULONG),
                    ("StateMask", wintypes.ULONG)]

    PROCESS_POWER_THROTTLING_CURRENT_VERSION = 1
    PROCESS_POWER_THROTTLING_EXECUTION_SPEED = 0x1
    ProcessPowerThrottling = 4

    try:
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        # argtypes/restype must be declared: HANDLE is 64-bit here, and the
        # GetCurrentProcess pseudo-handle (-1) is otherwise truncated to a
        # 32-bit int and the call fails with ERROR_INVALID_HANDLE.
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.GetCurrentProcess.argtypes = []
        k32.SetProcessInformation.restype = wintypes.BOOL
        k32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                              ctypes.c_void_p, wintypes.DWORD]
        state = _PowerThrottlingState(
            Version=PROCESS_POWER_THROTTLING_CURRENT_VERSION,
            ControlMask=PROCESS_POWER_THROTTLING_EXECUTION_SPEED,
            StateMask=0)                       # 0 = do not throttle
        ok = k32.SetProcessInformation(
            k32.GetCurrentProcess(),
            ProcessPowerThrottling,
            ctypes.byref(state),
            ctypes.sizeof(state))
        result["throttling_disabled"] = bool(ok)
        if not ok:
            result["throttling_error"] = ctypes.get_last_error()
    except (AttributeError, OSError):
        pass

    try:
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        prev = ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        result["execution_state_set"] = bool(prev)
    except (AttributeError, OSError):
        pass

    return result


def release_hot():
    """Drop the execution-state request. Safe to call unconditionally."""
    if os.name != "nt":
        return
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
    except (AttributeError, OSError):
        pass
