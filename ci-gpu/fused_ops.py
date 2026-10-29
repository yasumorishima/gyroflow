"""The operations in each of burn's fused elementwise kernels, with their time in the profile.

Usage: python fused_ops.py compile.log profile.log
compile.log: cubecl's compilation log at level full (each kernel's id, with the fused operations, then its source);
profile.log: profile_summary.py's output. Prints per ElemwiseFuse kernel its time and launches per pass (or "-" if it
is not among the profile's top names) and the operations it was fused from, in order, longest first. A kernel made of
a Cat alone is a concatenation copied in memory: what a reader of the two parts would not need.
"""
import re
import sys

compile_log, profile_log = sys.argv[1], sys.argv[2]
OPS = {"Cat", "Add", "Sub", "Mul", "Div", "Exp", "Erf", "Tanh", "Sigmoid", "Powf", "Sqrt", "Abs", "Recip", "Clamp",
       "Max", "Min", "Log", "Equal", "Lower", "LowerEqual", "Greater", "GreaterEqual", "ConditionalAssign", "Assign",
       "Gather", "Select", "Mask", "MaskFill", "Slice", "SliceAssign"}
source = open(compile_log, encoding="utf-8", errors="replace").read()
profile = open(profile_log, encoding="utf-8", errors="replace").read()
rows = []
for block in source.split("[START_KERNEL_COMPILATION]"):
    m = re.search(r"fn elemwise_fuse_([0-9a-f]{8})", block)
    if not m:
        continue
    h = m.group(1)
    p = re.search(r"([\d.]+) ms\s+[\d.]+ %\s+([\d.]+) launches\s+ElemwiseFuse_" + h, profile)
    info = block[:block.find("source:")]
    ops = [o for o in re.findall(r"^\s{8,}([A-Z][A-Za-z]+)(?: \{|\(|,)", info, re.M) if o in OPS]
    rows.append((float(p.group(1)) if p else -1.0, p.group(2) if p else "-", h, " ".join(ops)))
for ms, n, h, ops in sorted(rows, reverse=True):
    print(f"{ms if ms >= 0 else '-':>8} ms {n:>5} launches  ElemwiseFuse_{h}: {ops}")
