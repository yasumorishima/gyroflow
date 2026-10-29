"""Sums cubecl's per-kernel profile lines ("| <duration> | <kernel name>") of the warm passes of a gmflow test log.

Usage: python profile_summary.py test.log START_MARKER END_MARKER PASSES
Takes the lines after the first line containing START_MARKER up to the first line containing END_MARKER after it,
groups them by kernel name (the name up to its first '(' or '<', and the full name), and prints per pass: launches,
milliseconds, share. Exits 1 if no kernel line was found (a profile that measured nothing is not a profile).
"""
import re
import sys
from collections import defaultdict

path, start, end, passes = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
units = {"ns": 1e-6, "µs": 1e-3, "us": 1e-3, "ms": 1.0, "s": 1e3}
line_re = re.compile(r"^\|\s*([0-9.]+)(ns|µs|us|ms|s)\s*\|\s*(.+?)\s*$")
inside, rows, unmeasured = False, [], 0
with open(path, encoding="utf-8", errors="replace") as f:
    for line in f:
        if not inside:
            inside = start in line
            continue
        if end in line:
            break
        if "| not measured" in line:
            unmeasured += 1
            continue
        m = line_re.match(line.strip())
        if m:
            rows.append((float(m.group(1)) * units[m.group(2)], m.group(3)))
# What a kernel computes, where the test printed it after the profile ("KERNEL-NAME <name> <description>")
described = {}
with open(path, encoding="utf-8", errors="replace") as f:
    for line in f:
        m = re.match(r"^KERNEL-NAME (\S+) (.+)$", line.strip())
        if m:
            described[m.group(1)] = m.group(2)
if not rows:
    print("no profiled kernels between the markers")
    sys.exit(1)
total = sum(ms for ms, _ in rows)
print(f"{len(rows) / passes:.0f} launches and {total / passes:.1f} ms of kernel time per pass "
      f"({passes} passes, {unmeasured} launches not measured)")
# Each distinct full name is a pipeline compiled at the first pass (its shape and comptime arguments are in the name)
print(f"{len({name for _, name in rows})} distinct kernels (full names), "
      f"{len({re.split(r'[(<{\s]', name, maxsplit=1)[0] for _, name in rows})} kernel families")
# The kind: the family without the hash cubecl appends to its name (CoopKernel_f30810b7 -> CoopKernel), every kernel
# counted; a hash changes with the kernel's id, so totals by kind are what compares across commits
kind = lambda n: re.sub(r"_[0-9a-f]{8}$", "", re.split(r"[(<{\s]", n, maxsplit=1)[0])
for label, key, top in (("by kind", kind, None), ("by kernel family", lambda n: re.split(r"[(<{\s]", n, maxsplit=1)[0], 40),
                        ("by full name", lambda n: n[:160], 40)):
    agg = defaultdict(lambda: [0, 0.0])
    for ms, name in rows:
        a = agg[key(name)]
        a[0] += 1
        a[1] += ms
    print(f"--- {label}" + (f" (top {top})" if top else " (all)"))
    for name, (n, ms) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:top]:
        what = f"  [{described[name]}]" if name in described else ""
        print(f"{ms / passes:9.2f} ms {100 * ms / total:5.1f} % {n / passes:7.1f} launches  {name}{what}")
