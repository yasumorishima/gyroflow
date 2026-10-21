"""Export the gmflow ONNX for GPU-only inference and check it against the unchanged model.

Same model and settings as #1143 (gmflow-scale2-regrefine6, mixdata, 320x576, AdrianEddy/unimatch
da140fa), with lowmem.py's local correlations so that no intermediate tensor is >= 128 MiB.
Usage: python export_gpu.py gmflow.pth unimatch_dir video.mp4
Writes gmflow-scale2-regrefine6-320x576-opset16-sim.onnx, and for each test pair k:
pair{k}_in0.f32, pair{k}_in1.f32 ([1,3,320,576], raw 0-255 grey in all 3 channels) and
pair{k}_torch.f32 ([1,2,320,576], the unchanged PyTorch model). Exits 1 if a gate fails.
"""
import copy
import hashlib
import sys

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import onnxsim
import torch

PTH, UNIMATCH, VIDEO = sys.argv[1], sys.argv[2], sys.argv[3]
H, W, SHIFT = 320, 576, 4
OUT = "gmflow-scale2-regrefine6-320x576-opset16-sim.onnx"
# Gates (px). That the patch changes no value is checked in float64, where float reordering is ~1e-12 and the
# check is sharp: the first version checked it in float32 with a mean limit of 1e-5 set before measuring the noise,
# and on a second runner float32 reordering alone was 1e-5 (onnxruntime vs PyTorch too). float32 results, the ones
# that ship, are held to the margin the unpatched export had in onnxruntime (max 5.7e-4, mean 1.9e-5).
EXACT64 = 1e-6
F32_MAX, F32_MEAN = 2e-3, 5e-5
ORT_MAX, ORT_MEAN = 2e-3, 5e-5

digest = hashlib.sha256(open(PTH, "rb").read()).hexdigest()
assert digest.startswith("4e7b215d"), f"checkpoint sha256 {digest} does not match the file name"
sys.path.insert(0, UNIMATCH)
sys.path.insert(0, ".")
from unimatch.unimatch import UniMatch  # noqa: E402
import lowmem  # noqa: E402

torch.manual_seed(0)
net = UniMatch(feature_channels=128, num_scales=2, upsample_factor=4, num_head=1, ffn_dim_expansion=4,
               num_transformer_layers=6, reg_refine=True, task="flow")
net.load_state_dict(torch.load(PTH, map_location="cpu", weights_only=False)["model"], strict=True)
net.eval()


class Flow(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, img0, img1):
        out = self.m(img0, img1, attn_type="swin", attn_splits_list=[2, 8], corr_radius_list=[-1, 4],
                     prop_radius_list=[-1, 1], num_reg_refine=6, task="flow")
        return out["flow_preds"][-1]


def as_input(grey):
    return torch.from_numpy(np.repeat(grey[None, None], 3, 1).astype(np.float32))


# Pair 0: textured grey, frame 1 = frame 0 moved 4 px right (true flow dx = +4, dy = 0).
rng = np.random.default_rng(0)
base = rng.random((H + 8, W + SHIFT + 8))
for _ in range(3):
    base = (base + np.roll(base, 1, 0) + np.roll(base, -1, 0) + np.roll(base, 1, 1) + np.roll(base, -1, 1)) / 5
base = base[4:4 + H, 4:4 + W + SHIFT]
base = np.round((base - base.min()) / (base.max() - base.min()) * 255.0)
pairs = [(as_input(base[:, SHIFT:SHIFT + W]), as_input(base[:, 0:W]))]

# Pairs 1-2: consecutive frames of a real video, grey, resized to the model size.
cap = cv2.VideoCapture(VIDEO)
frames = []
while len(frames) < 62:
    ok, f = cap.read()
    if not ok:
        break
    frames.append(np.round(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (W, H),
                                      interpolation=cv2.INTER_AREA)).astype(np.float64))
assert len(frames) >= 62, f"video gave {len(frames)} frames"
pairs += [(as_input(frames[0]), as_input(frames[1])), (as_input(frames[60]), as_input(frames[61]))]

wrap = Flow(net).eval()
with torch.no_grad():
    refs = [wrap(a, b).numpy() for a, b in pairs]
    wrap64 = Flow(copy.deepcopy(net).double()).eval()
    refs64 = [wrap64(a.double(), b.double()).numpy() for a, b in pairs]
for k, ((a, b), r) in enumerate(zip(pairs, refs)):
    a.numpy().tofile(f"pair{k}_in0.f32")
    b.numpy().tofile(f"pair{k}_in1.f32")
    r.astype(np.float32).tofile(f"pair{k}_torch.f32")
    print(f"pair {k}: torch median dx {np.median(r[0, 0]):.4f} dy {np.median(r[0, 1]):.4f}")

failed = []


def gate(tag, got, ref, max_lim, mean_lim):
    d = np.abs(got - ref)
    ok = d.max() <= max_lim and d.mean() <= mean_lim
    print(f"{tag}: max {d.max():.2e} mean {d.mean():.2e} p99 {np.quantile(d, 0.99):.2e} "
          f"(limits {max_lim:g} / {mean_lim:g}) {'PASS' if ok else 'FAIL'}")
    if not ok:
        failed.append(tag)


lowmem.install()
with torch.no_grad():
    for k, ((a, b), r, r64) in enumerate(zip(pairs, refs, refs64)):
        gate(f"pair {k} patched vs unchanged, float64", wrap64(a.double(), b.double()).numpy(), r64, EXACT64, EXACT64)
        gate(f"pair {k} patched vs unchanged, float32", wrap(a, b).numpy(), r, F32_MAX, F32_MEAN)

t0, t1 = pairs[0]
torch.onnx.export(wrap, (t0, t1), "raw.onnx", opset_version=16, input_names=["img0", "img1"],
                  output_names=["flow"], do_constant_folding=True, dynamo=False)
sim, ok = onnxsim.simplify(onnx.load("raw.onnx"))
assert ok, "onnxsim could not validate the simplified model"
onnx.save(sim, OUT)
print("onnx nodes", len(sim.graph.node), "initializers", len(sim.graph.initializer))
print("onnx sha256", hashlib.sha256(open(OUT, "rb").read()).hexdigest())

sess = ort.InferenceSession(OUT, providers=["CPUExecutionProvider"])
for k, ((a, b), r) in enumerate(zip(pairs, refs)):
    got = sess.run(None, {"img0": a.numpy(), "img1": b.numpy()})[0]
    gate(f"pair {k} onnxruntime (patched export) vs unchanged torch", got, r, ORT_MAX, ORT_MEAN)

if failed:
    print("FAILED:", failed)
    sys.exit(1)
