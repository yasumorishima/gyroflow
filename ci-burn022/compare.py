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
                ("burn 0.22, regenerated .bpk", "new_generated.f32"),
                ("burn 0.22 wgpu (lavapipe), published .bpk", "new_gpu.f32")]:
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
# Only what this run builds is required (REQUIRED, space-separated); the rest is reported if present.
required = os.environ.get("REQUIRED", "ort_flow.f32 new_published.f32 new_generated.f32").split()
bad = [f for f in required if not isinstance(status.get(f), float)]
if bad:
    raise SystemExit(f"no usable output from: {bad}")
ref_dx = float(np.median(ref.reshape(1, 2, 320, 576)[0, 0]))
for f in required:
    x = np.fromfile(f, dtype="<f4").reshape(1, 2, 320, 576)
    if status[f] >= 1e-3 or abs(float(np.median(x[0, 0])) - ref_dx) >= 0.05:
        raise SystemExit(f"{f} is not close to PyTorch: mean diff {status[f]:.2e}, median dx {np.median(x[0, 0]):.4f}")
print("all required outputs match PyTorch (mean diff < 1e-3, median dx within 0.05)")
