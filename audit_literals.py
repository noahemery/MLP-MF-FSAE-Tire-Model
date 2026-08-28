"""Check fit_pipeline's config against magic.py's source text.

This is the drift check for the 16 specs that have no reference fit.

It does NOT re-run segmentation and compare outputs -- that would test the
transcription against itself. Instead it reads magic.py as text, strips ONE
leading comment marker from each line of each block (so the double-commented
bound2 path at magic.py:355-360 stays excluded, exactly as it is when the
block is live), parses the result with ast, and pulls out every literal that
matters: loadmat paths, sort/bound arguments, ET windows, the mean|SA| test,
the first-pass bounds, and the second-pass max_nfev.

Those extracted values are then compared to fit_pipeline's config. Any
mismatch is a drift bug and is printed as a failure.

  python audit_literals.py
"""

import ast
import re
import sys

fp = None   # imported lazily in main(); importing it imports magic.py

SOURCE = "magic.py"


def _const(node):
    """Literal value of an ast node, or a marker describing it."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _const(node.operand)
        return -inner if isinstance(inner, (int, float)) else ("-" + str(inner))
    if isinstance(node, ast.List):
        return [_const(e) for e in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_const(e) for e in node.elts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        # the "MARGIN + np.abs(cases[i][...]).max()" D bound
        left = _const(node.left)
        if isinstance(left, (int, float)):
            return ("MARGIN", left)
    if isinstance(node, ast.Call):
        return ("call", ast.unparse(node.func))
    return ("expr", ast.unparse(node))


def _kw(call, name, default=None):
    for k in call.keywords:
        if k.arg == name:
            return _const(k.value)
    return default


def uncomment(lines):
    """Strip exactly one leading comment marker, preserving indentation.

    The file's convention is that disabling a block prefixes each line with
    '# '. Undoing that once leaves double-commented lines still commented,
    which is what we want.
    """
    out = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("# "):
            indent = len(line) - len(stripped)
            out.append(" " * indent + stripped[2:])
        elif stripped == "#":
            out.append("")
        elif stripped.startswith("#"):
            indent = len(line) - len(stripped)
            out.append(" " * indent + stripped[1:])
        else:
            out.append(line)
    return out


def block_ranges(lines):
    """Map spec_id -> (start, end) line indices, from its assignment line."""
    assign = {}
    for sid in fp.ALL_SPECS:
        pat = re.compile(r"^\s*#?\s*" + re.escape(sid) + r"\s*=")
        for i, line in enumerate(lines):
            if pat.match(line):
                assign[sid] = i
                break
    ordered = sorted(assign.items(), key=lambda kv: kv[1])
    ranges, prev_end = {}, 0
    # Every block opens with a "This is ..." comment naming the tire. Anchor on
    # it rather than on the previous block's end: the region before the first
    # block is magic.py's function definitions, and un-commenting their inline
    # comments would turn prose into code.
    header = re.compile(r"^[\s#]*This is\b")
    for sid, end in ordered:
        start = prev_end
        for i in range(end, prev_end - 1, -1):
            if header.match(lines[i]):
                start = i
                break
        ranges[sid] = (start, end + 1)
        prev_end = end + 1
    return ranges


def is_commented_block(block):
    """A block is disabled iff its own assignment line is commented out."""
    for line in reversed(block):
        if line.strip():
            return line.lstrip().startswith("#")
    return False


def extract(lines, start, end):
    """Pull the literals out of one block, live or commented."""
    block = lines[start:end]
    # Only un-comment a block that is actually disabled. A live block's header
    # is a genuine comment and must stay one.
    src = "\n".join(uncomment(block) if is_commented_block(block) else block)
    tree = ast.parse(src)

    found = {"fz0_files": [], "sorts": [], "bounds": [], "et": [], "sa": [],
             "first_bounds": None, "second_max_nfev": None, "x0_P": None}

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fname = ast.unparse(node.func)
            if fname.endswith("loadmat") and node.args:
                path = _const(node.args[0])
                if isinstance(path, str):
                    found.setdefault("_all_loadmat", []).append(path)
            elif fname == "sort":
                inner = [a for a in ast.walk(node.args[0])
                         if isinstance(a, ast.Call)
                         and ast.unparse(a.func).endswith("loadmat")]
                found["sorts"].append({
                    "path": _const(inner[0].args[0]) if inner else None,
                    "load_key": _kw(node, "load_key"),
                    "window": _kw(node, "window"),
                    "threshold_factor": _kw(node, "threshold_factor"),
                })
            elif fname == "bound":
                found["bounds"].append({
                    "slip_key": _kw(node, "slip_key", "SA"),
                    "threshold_factor": _kw(node, "threshold_factor"),
                })
            elif fname == "bound2":
                found.setdefault("bound2_seen", True)

        # F_z0 = np.abs(np.vstack((...))).mean()
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and getattr(node.targets[0], "id", None) == "F_z0"):
            found["fz0_files"] = [
                _const(c.args[0]) for c in ast.walk(node.value)
                if isinstance(c, ast.Call)
                and ast.unparse(c.func).endswith("loadmat")]

        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and getattr(node.targets[0], "id", None) in ("x0_P", "x0_R")):
            found["x0_P"] = _const(node.value)

        # the segment-selection if-test
        if isinstance(node, ast.If):
            text = ast.unparse(node.test)
            for m in re.finditer(r"ET'\]\.max\(\) - split\[i\]\['ET'\]\.min\(\)"
                                 r"\s*([<>])\s*([0-9.]+)", text):
                found["et"].append((m.group(1), float(m.group(2))))
            m = re.search(r"abs\(split\[i\]\['SA'\]\)\.mean\(\)\s*"
                          r"([<>])\s*([0-9.e-]+)", text)
            if m:
                found["sa"].append((m.group(1), float(m.group(2))))

        # least_squares(...) calls: first pass has bounds=, second has method=lm
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "least_squares":
            b = _kw(node, "bounds")
            if b is not None and found["first_bounds"] is None:
                found["first_bounds"] = b
            if _kw(node, "method") == "lm":
                found["second_max_nfev"] = _kw(node, "max_nfev")
                found["second_ftol"] = _kw(node, "ftol")
            elif b is not None:
                found["first_ftol"] = _kw(node, "ftol")
    return found


def norm_x0_from_source(x0):
    """Normalise an x0 extracted from magic.py for comparison.

    A column-mean entry comes out as ('call', 'BCDE_params[:, 1].mean');
    reduce it to ('MEAN', 1) so it can be matched against the config's
    ColMean placeholder.
    """
    if x0 is None:
        return None
    out = []
    for v in x0:
        if isinstance(v, tuple) and len(v) == 2 and v[0] == "call":
            m = re.search(r"\[:,\s*(\d+)\]\.mean$", str(v[1]))
            out.append(("MEAN", int(m.group(1))) if m else v)
        else:
            out.append(v)
    return out


def norm_x0_from_config(template):
    """Same normalisation for fit_pipeline's template."""
    return [("MEAN", v.col) if isinstance(v, fp._ColMean) else v
            for v in template]


def check(label, got, want, failures):
    ok = got == want
    if not ok:
        failures.append(label + ": magic.py has " + repr(got) +
                        ", config has " + repr(want))
    return ok


def main():
    global fp
    import fit_pipeline as fp

    with open(SOURCE) as fh:
        lines = fh.read().splitlines()

    ranges = block_ranges(lines)
    missing = [s for s in fp.ALL_SPECS if s not in ranges]
    if missing:
        raise SystemExit("no assignment line found in magic.py for: " +
                         ", ".join(missing))

    failures, checked = [], 0
    print("auditing " + str(len(ranges)) + " blocks in " + SOURCE + "\n")

    for sid, spec in fp.ALL_SPECS.items():
        start, end = ranges[sid]
        got = extract(lines, start, end)
        n0 = len(failures)

        if spec.family == "lateral":
            check(sid + " fz0_files", got["fz0_files"], list(spec.fz0_files),
                  failures)
            check(sid + " n_case_files", len(got["sorts"]),
                  len(spec.case_files), failures)
            for s, b, fs in zip(got["sorts"], got["bounds"], spec.case_files):
                check(sid + " " + fs.path + " sort",
                      (s["path"], s["load_key"], s["window"],
                       s["threshold_factor"]),
                      (fs.path, "FZ", fs.sort_window, fs.sort_tf), failures)
                check(sid + " " + fs.path + " bound",
                      (b["slip_key"], b["threshold_factor"]),
                      ("SA", fs.bound_tf), failures)
            want_et = []
            for fs in spec.case_files:
                want_et += [("<", float(fs.et_hi)), (">", float(fs.et_lo))]
            check(sid + " ET windows", got["et"], want_et, failures)
            check(sid + " SA filter", got["sa"], [], failures)
            check(sid + " D margin/S_vy bound",
                  (got["first_bounds"][0], got["first_bounds"][1][2],
                   got["first_bounds"][1][5]),
                  (fp.LAT_LOWER, ("MARGIN", fp.LAT_D_MARGIN),
                   fp.LAT_SVY_BOUND), failures)
            check(sid + " second max_nfev", got["second_max_nfev"],
                  ("call", "int"), failures)
            check(sid + " x0_P", norm_x0_from_source(got["x0_P"]),
                  norm_x0_from_config(fp.LAT_X0_P), failures)

        elif spec.family == "longitudinal":
            check(sid + " fz0_files", got["fz0_files"], list(spec.fz0_files),
                  failures)
            for s, b, path in zip(got["sorts"], got["bounds"], spec.case_files):
                check(sid + " " + path + " sort",
                      (s["path"], s["load_key"], s["window"],
                       s["threshold_factor"]),
                      (path, "FZ", fp.STRAIGHT_SORT_WINDOW,
                       fp.STRAIGHT_SORT_TF), failures)
                check(sid + " " + path + " bound",
                      (b["slip_key"], b["threshold_factor"]),
                      ("SL", fp.STRAIGHT_BOUND_TF), failures)
            check(sid + " ET windows", got["et"],
                  [("<", float(fp.STRAIGHT_ET_HI)),
                   (">", float(fp.STRAIGHT_ET_LO))] * len(spec.case_files),
                  failures)
            check(sid + " SA filter", got["sa"],
                  [("<", fp.LONG_SA_MAX)] * len(spec.case_files), failures)
            check(sid + " D margin", got["first_bounds"][1][2],
                  ("MARGIN", spec.d_margin), failures)
            check(sid + " S_vx bound",
                  (got["first_bounds"][0][5], got["first_bounds"][1][5]),
                  (-spec.svx_bound, spec.svx_bound), failures)
            check(sid + " B/C bounds",
                  (got["first_bounds"][0][0], got["first_bounds"][0][1],
                   got["first_bounds"][1][0], got["first_bounds"][1][1]),
                  (fp.LONG_B_LOWER, fp.LONG_C_LOWER, fp.LONG_B_UPPER,
                   fp.LONG_C_UPPER), failures)
            check(sid + " x0_P", norm_x0_from_source(got["x0_P"]),
                  norm_x0_from_config(fp.LONG_X0_P), failures)

        else:   # gx / gy
            for s, b, path in zip(got["sorts"], got["bounds"], spec.case_files):
                check(sid + " " + path + " sort",
                      (s["path"], s["load_key"], s["window"],
                       s["threshold_factor"]),
                      (path, "FZ", fp.STRAIGHT_SORT_WINDOW,
                       fp.STRAIGHT_SORT_TF), failures)
                check(sid + " " + path + " bound",
                      (b["slip_key"], b["threshold_factor"]),
                      ("SL", fp.STRAIGHT_BOUND_TF), failures)
            check(sid + " ET windows", got["et"],
                  [("<", float(fp.STRAIGHT_ET_HI)),
                   (">", float(fp.STRAIGHT_ET_LO))] * len(spec.case_files),
                  failures)
            check(sid + " SA threshold", got["sa"],
                  [(">", spec.sa_threshold)] * len(spec.case_files), failures)
            lower = fp.GX_LOWER if spec.family == "gx" else fp.GY_LOWER
            upper = fp.GX_UPPER if spec.family == "gx" else fp.GY_UPPER
            check(sid + " first bounds", got["first_bounds"],
                  (lower, upper), failures)
            check(sid + " first ftol", got.get("first_ftol"), None, failures)
            check(sid + " x0_R", norm_x0_from_source(got["x0_P"]),
                  norm_x0_from_config(fp.GX_X0_R if spec.family == "gx"
                                      else fp.GY_X0_R), failures)
            if spec.family == "gy":
                check(sid + " second max_nfev is 1e2",
                      "int(100.0)" if got["second_max_nfev"] else None,
                      "int(100.0)", failures)

        checked += 1
        status = "OK" if len(failures) == n0 else "MISMATCH"
        extra = ""
        if got.get("bound2_seen"):
            extra = "   (block also contains a bound2 call)"
        print("  %-22s %-9s cases=%d%s"
              % (sid, status, len(got["sorts"]), extra))

    print("\n" + "=" * 62)
    if failures:
        print("DRIFT DETECTED -- " + str(len(failures)) + " mismatch(es):\n")
        for f in failures:
            print("  " + f)
        print("=" * 62)
        return 1
    print("all " + str(checked) + " blocks match magic.py's source literals")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
