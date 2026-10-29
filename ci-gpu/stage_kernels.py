"""Which stage of a pass launches each kernel: cubecl's profile lines ("| <duration> | <kernel name>") between
STAGED-START and STAGED-END, each counted in the stage whose "STAGE-MARK <name>" comes next (the pass syncs and prints
the mark at the end of each stage).

Usage: python stage_kernels.py test.log
Prints per stage its kernel time and launches, then the kernels of each kind other than CoopKernel (copies,
fused elementwise kernels, ...) per stage, to find what makes them.
"""
import re
import sys
from collections import defaultdict

units = {"ns": 1e-6, "µs": 1e-3, "us": 1e-3, "ms": 1.0, "s": 1e3}
line_re = re.compile(r"^\|\s*([0-9.]+)(ns|µs|us|ms|s)\s*\|\s*(.+?)\s*$")
inside, pending, stages = False, [], []
with open(sys.argv[1], encoding="utf-8", errors="replace") as f:
    for line in f:
        line = line.strip()
        if not inside:
            inside = line.startswith("STAGED-START")
            continue
        if line.startswith("STAGED-END"):
            break
        if line.startswith("STAGE-MARK "):
            stages.append((line[len("STAGE-MARK "):], pending))
            pending = []
            continue
        m = line_re.match(line)
        if m:
            pending.append((float(m.group(1)) * units[m.group(2)], m.group(3)))
if pending:
    stages.append(("(after the last mark)", pending))
if not stages:
    print("no stage marks between STAGED-START and STAGED-END")
    sys.exit(1)
kind = lambda n: re.sub(r"_[0-9a-f]{8}$", "", re.split(r"[(<{\s]", n, maxsplit=1)[0])
print("--- per stage: ms, launches")
for name, ks in stages:
    print(f"{sum(ms for ms, _ in ks):9.2f} ms {len(ks):4} launches  {name}")
print("--- kernels other than CoopKernel, per stage (ms, launches, name)")
for name, ks in stages:
    agg = defaultdict(lambda: [0, 0.0])
    for ms, n in ks:
        if kind(n) != "CoopKernel":
            agg[n[:80]][0] += 1
            agg[n[:80]][1] += ms
    if agg:
        print(f"[{name}]")
        for n, (c, ms) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
            print(f"  {ms:8.2f} ms {c:3}  {n}")
