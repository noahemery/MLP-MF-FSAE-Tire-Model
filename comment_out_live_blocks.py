"""Comment out the two live fitting blocks in magic.py.

This is the ONLY change made to magic.py. It touches no literal, no operator,
no sign, no function body -- it prefixes driver lines with '# ' so that all 18
blocks are uniformly commented and `import magic` stops launching real fits.

It follows the file's own existing convention exactly: every non-blank line in
the block gets a '# ' prefix, blank lines are left blank. A block header that
was already '# This is the data for ...' becomes '# # This is the data ...',
which is precisely how the 16 already-disabled blocks look (see magic.py:349).

  python comment_out_live_blocks.py --check    # report, change nothing
  python comment_out_live_blocks.py            # apply
  python comment_out_live_blocks.py --revert   # undo

A .bak is written before the first edit.
"""

import argparse
import os
import shutil
import sys

TARGET = "magic.py"
BACKUP = "magic.py.bak"

# 1-indexed, inclusive. The two blocks that are live in magic.py as committed.
BLOCKS = (
    (619, 666, "lat_180X60_R20_70"),
    (1602, 1645, "GY_180X60_R20_70"),
)


def read_lines():
    with open(TARGET, newline="") as fh:
        return fh.read().splitlines(keepends=True)


def is_blank(line):
    return line.strip() == ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()

    if args.revert:
        if not os.path.exists(BACKUP):
            raise SystemExit("no " + BACKUP + " to revert from")
        shutil.copyfile(BACKUP, TARGET)
        print("reverted " + TARGET + " from " + BACKUP)
        return 0

    lines = read_lines()

    # Sanity-check the ranges before touching anything: the last line of each
    # block must be that spec's assignment.
    for start, end, spec_id in BLOCKS:
        tail = lines[end - 1]
        if not tail.lstrip().startswith(spec_id):
            raise SystemExit(
                "line " + str(end) + " is not the " + spec_id +
                " assignment; magic.py does not match the expected layout.\n"
                "  found: " + tail.rstrip())
        head = lines[start - 1]
        if head.lstrip().startswith("#") and "This is" not in head:
            raise SystemExit("line " + str(start) + " is not a block header: "
                             + head.rstrip())

    already = all(lines[s - 1].lstrip().startswith("# #") for s, _, _ in BLOCKS)
    if already:
        print("both blocks already commented out; nothing to do")
        return 0

    changed = 0
    for start, end, spec_id in BLOCKS:
        n = 0
        for i in range(start - 1, end):
            if is_blank(lines[i]):
                continue
            lines[i] = "# " + lines[i]
            n += 1
        changed += n
        print("block " + spec_id + "  lines " + str(start) + "-" + str(end) +
              ": " + str(n) + " lines prefixed")

    if args.check:
        print("\n--check: nothing written (" + str(changed) +
              " lines would change)")
        return 0

    if not os.path.exists(BACKUP):
        shutil.copyfile(TARGET, BACKUP)
        print("backup written to " + BACKUP)

    with open(TARGET, "w", newline="") as fh:
        fh.writelines(lines)
    print("\n" + TARGET + " updated: " + str(changed) + " lines commented, "
          "0 literals changed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
