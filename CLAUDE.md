# CLAUDE.md

FSAE tire modeling. This repo fits the Pacejka Magic Formula to Calspan TTC
Round 9 (Project 2356) rig data, covering lateral pure slip, longitudinal pure
slip, and the combined-slip G corrections.

Current work: building a neural parameter-estimation framework on top of the
existing fits, so Magic Formula parameters can be predicted from tire geometry
instead of fit independently per tire spec.

## Repo Ownership And Boundaries

`magic.py` is William Young's, and it is the canonical physics. He owns it.
Everything this project adds sits on top of it.

**Do not change the physics in `magic.py`.** Not signs, not operators, not
parenthesization, not epsilon guards, not term ordering, not which data channel
feeds which variable. The sign conventions have been independently verified by
the author and are not open for revision here.

If something in `magic.py` looks wrong, stop and report it with reasoning. Do
not fix it, do not fix it temporarily, and do not add a flag that changes it.
Report it and let the owner decide.

Aligning torque, self-aligning moment, Mz, and the Q-parameter family are owned
separately and are out of scope. Do not implement, refactor, or reason about
them.

## Layout

- `magic.py`: canonical physics plus the per-spec fitting blocks. 18 blocks,
  most commented out because they were originally run one at a time by hand.
- `tire_model.py`: standalone predictor with hardcoded parameter lists.
  **These are stale**, they are the older 20-parameter lateral form and do not
  match `magic.py`'s current 22-parameter fit. Do not use them as a baseline.
- `data_comber.py`: earlier data handling utilities.
- `data/cornering_SI/` and `data/straight_SI/`: raw `.mat` signal files paired
  with `.dat` files whose first line carries tire name, dimensions, and rim
  width.
- `DECISIONS.md`: live decision record for current work. Read it before
  assuming project state from this file alone.
- `docs/DECISIONS-mf-gp.md`: archived record from the earlier NLLS plus
  Gaussian Process pipeline. Historical, do not edit, and note that its numbers
  were computed against the 20-parameter lateral formula.

## Teaching Mode

I am learning Python, venvs, and ML fundamentals. When you fix something, also
explain (1) what was broken and (2) why the fix works, so I can repeat it.
Prefer in-place edits to existing files over restructuring into new packages
unless I ask.

## Git And Commits

- Never add Claude as a co-author in commit messages. Commit as the repo owner
  only.
- Before any `git push`, scan staged files for secrets and confirm `.gitignore`
  covers `.env`, `venv/`, `.venv/`, `__pycache__/`.
- The raw TTC data is currently committed to this repo. That may be
  license-restricted. Do not add more of it, do not move it, and do not remove
  what is there without asking. This is a question for the repo owner.

## Language And Tooling

- Interpreter: TODO confirm. Before installing anything, check which
  interpreter is active with `python -c "import sys; print(sys.executable)"`.
- Core dependencies: `numpy`, `scipy`, `pandas`, `pyarrow`, `matplotlib`. The
  neural work adds `torch`. Pinned as minimum versions in `requirements.txt`.
- No test suite exists yet. Reproducing a known fit is the closest thing to a
  regression check.

## Runtime Expectations

`magic.py`'s fits are real `scipy.optimize.least_squares` runs with tolerances
at `2.3e-16` and `max_nfev` at `1e8`, not quick scripts. A single tire spec
involves a per-segment first pass followed by a second pass over all segments.
There are 18 such blocks. Confirm scope before launching a full run rather than
assuming it is cheap, and report actual wall-clock time when you do.

## Verification Before Claiming Success

- Do not describe warnings or errors as harmless without running the code and
  showing the output.
- After any fix, re-run the failing command and paste the real output before
  summarizing.
- Never present in-sample RMSE or R-squared as a performance result. Held-out
  metrics only, using GroupKFold grouped by segment id. Adjacent points within
  a sweep are near-duplicates, so a random row split leaks and produces a
  held-out error that looks good for the wrong reason.

## Workflow Reminders

### Use Task Agents For Multi-File Investigation

When debugging something that spans multiple files or systems, do not chase it
sequentially. Define the full scope of what needs checking, spawn an agent to
sweep all of it, then review the findings before changing anything.
