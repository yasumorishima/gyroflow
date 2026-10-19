"""Compare every flow that exists against the PyTorch reference; write summary.md.
A flow file that is missing is reported as missing (its step failed), never skipped silently."""
import os

import numpy as np

N = 2 * 320 * 576
ref = np.fromfile("torch_flow.f32", dtype="<f4")
assert ref.size == N, "torch reference has the wrong size"
rows = ["| flow | median dx | median dy | max abs diff vs torch | mean | p99 |", "|---|---|---|---|---|---|"]
status = {}
for name, f in [("torch (reference)", "torch_flow.f32"), ("onnxruntime", "ort_flow.f32"),
                ("burn 0.21, published .bpk", "old_published.f32"),
                ("burn 0.22, published .bpk", "new_published.f32"),
                ("burn 0.22, regenerated .bpk", "new_generated.f32")]:
    if not os.path.exists(f):
        rows.append(f"| {name} | missing | | | | |")
        status[f] = "missing"
        continue
    x = np.fromfile(f, dtype="<f4")
    if x.size != N or not np.isfinite(x).all():
        rows.append(f"| {name} | bad output (size {x.size}, finite {np.isfinite(x).all()}) | | | | |")
        status[f] = "bad"
        continue
    v = x.reshape(1, 2, 320, 576)
    d = np.abs(x - ref)
    rows.append(f"| {name} | {np.median(v[0, 0]):.4f} | {np.median(v[0, 1]):.4f} | {d.max():.6f} | "
                f"{d.mean():.2e} | {np.quantile(d, 0.99):.2e} |")
    status[f] = float(d.mean())
text = "\n".join(rows) + "\n\nTrue flow of the synthetic pair: dx = +4, dy = 0.\n"
print(text)
with open(os.environ.get("GITHUB_STEP_SUMMARY", "summary.md"), "a") as fh:
    fh.write(text)
