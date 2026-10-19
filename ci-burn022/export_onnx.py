"""Re-create the gmflow ONNX that #1143 builds from, plus reference outputs.

Model: autonomousvision/unimatch gmflow-scale2-regrefine6 (mixdata, trained at 320x576), code from
AdrianEddy/unimatch (export fixes). Same settings as unimatch's scripts/gmflow_demo.sh.
Writes, in the current directory:
  gmflow-scale2-regrefine6-320x576-opset16-sim.onnx
  in0.f32, in1.f32      [1, 3, 320, 576] little-endian f32, raw 0-255 grey in all 3 channels
  torch_flow.f32        [1, 2, 320, 576] PyTorch output
  ort_flow.f32          same, from onnxruntime on the simplified ONNX
"""
import hashlib
import sys

import numpy as np
import onnx
import onnxruntime as ort
import onnxsim
import torch

PTH, UNIMATCH = sys.argv[1], sys.argv[2]
H, W, SHIFT = 320, 576, 4
OUT = "gmflow-scale2-regrefine6-320x576-opset16-sim.onnx"

digest = hashlib.sha256(open(PTH, "rb").read()).hexdigest()
assert digest.startswith("4e7b215d"), f"checkpoint sha256 {digest} does not match the file name"
sys.path.insert(0, UNIMATCH)
from unimatch.unimatch import UniMatch  # noqa: E402

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


# Textured grey pair; frame 1 is frame 0 moved 4 px right, so the true flow is dx = +4, dy = 0.
rng = np.random.default_rng(0)
base = rng.random((H + 8, W + SHIFT + 8)).astype(np.float64)
for _ in range(3):  # smooth so the texture has structure at several scales
    base = (base + np.roll(base, 1, 0) + np.roll(base, -1, 0) + np.roll(base, 1, 1) + np.roll(base, -1, 1)) / 5
base = base[4:4 + H, 4:4 + W + SHIFT]
base = (base - base.min()) / (base.max() - base.min()) * 255.0
img0 = base[:, SHIFT:SHIFT + W]
img1 = base[:, 0:W]
t0 = torch.from_numpy(np.repeat(img0[None, None], 3, 1).astype(np.float32))
t1 = torch.from_numpy(np.repeat(img1[None, None], 3, 1).astype(np.float32))
t0.numpy().tofile("in0.f32")
t1.numpy().tofile("in1.f32")

wrap = Flow(net).eval()
with torch.no_grad():
    ref = wrap(t0, t1).numpy()
ref.astype(np.float32).tofile("torch_flow.f32")
print("torch flow", ref.shape, "median dx %.4f dy %.4f" % (np.median(ref[0, 0]), np.median(ref[0, 1])))

torch.onnx.export(wrap, (t0, t1), "raw.onnx", opset_version=16, input_names=["img0", "img1"],
                  output_names=["flow"], do_constant_folding=True, dynamo=False)
sim, ok = onnxsim.simplify(onnx.load("raw.onnx"))
assert ok, "onnxsim could not validate the simplified model"
onnx.save(sim, OUT)

sess = ort.InferenceSession(OUT, providers=["CPUExecutionProvider"])
got = sess.run(None, {"img0": t0.numpy(), "img1": t1.numpy()})[0]
got.astype(np.float32).tofile("ort_flow.f32")
d = np.abs(got - ref)
print("ort vs torch: max %.6f mean %.2e p99 %.2e" % (d.max(), d.mean(), np.quantile(d, 0.99)))
print("onnx nodes", len(sim.graph.node), "initializers", len(sim.graph.initializer))
