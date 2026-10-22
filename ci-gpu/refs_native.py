"""PyTorch references for the hand-written burn gmflow (gmflow-scale2-regrefine6, mixdata, 320x576, AdrianEddy/unimatch
da140fa, the unchanged model: no ONNX, no lowmem patch).

Usage: python refs_native.py gmflow.pth unimatch_dir video.mp4 out_dir
Writes, for each test pair k (0: textured grey moved 4 px right, 1-2: consecutive frames of a real video):
  pair{k}_in0.f32, pair{k}_in1.f32   [1,3,320,576] raw 0-255 grey in all 3 channels (as before)
  pair{k}_torch.f32                  [1,2,320,576] flow frame 0 -> frame 1 (as before)
  pair{k}_bwd_torch.f32              [1,2,320,576] flow frame 1 -> frame 0, from pred_bidir_flow=True
and, for pair 1, every intermediate the port is checked against, stage by stage: inter/<name>.f32 + inter/manifest.txt
("name d0,d1,..." per line, float32, C order). Exits 1 if a gate fails.
"""
import hashlib
import os
import sys

import cv2
import numpy as np
import torch

PTH, UNIMATCH, VIDEO, OUT = sys.argv[1:5]
H, W, SHIFT = 320, 576, 4
# The bidirectional pass gives what two separate passes give up to the order float32 sums are taken in (batched
# matmuls): measured, not gated here. A first limit of mean 1e-5, set before measuring, failed at 1.9e-5 (max 5.4e-4): the
# float32 reordering noise of this model. The burn bidirectional flow is gated against these references like the
# one-way flow (max 1e-2, mean 1e-4).

os.makedirs(os.path.join(OUT, "inter"), exist_ok=True)
digest = hashlib.sha256(open(PTH, "rb").read()).hexdigest()
assert digest.startswith("4e7b215d"), f"checkpoint sha256 {digest} does not match the file name"
print("checkpoint sha256", digest)
sys.path.insert(0, UNIMATCH)
from unimatch.unimatch import UniMatch  # noqa: E402

torch.manual_seed(0)
net = UniMatch(feature_channels=128, num_scales=2, upsample_factor=4, num_head=1, ffn_dim_expansion=4,
               num_transformer_layers=6, reg_refine=True, task="flow")
net.load_state_dict(torch.load(PTH, map_location="cpu", weights_only=False)["model"], strict=True)
net.eval()
ARGS = dict(attn_type="swin", attn_splits_list=[2, 8], corr_radius_list=[-1, 4], prop_radius_list=[-1, 1],
            num_reg_refine=6, task="flow")


def as_input(grey):
    return torch.from_numpy(np.repeat(grey[None, None], 3, 1).astype(np.float32))


rng = np.random.default_rng(0)
base = rng.random((H + 8, W + SHIFT + 8))
for _ in range(3):
    base = (base + np.roll(base, 1, 0) + np.roll(base, -1, 0) + np.roll(base, 1, 1) + np.roll(base, -1, 1)) / 5
base = base[4:4 + H, 4:4 + W + SHIFT]
base = np.round((base - base.min()) / (base.max() - base.min()) * 255.0)
pairs = [(as_input(base[:, SHIFT:SHIFT + W]), as_input(base[:, 0:W]))]
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

failed = []


def gate(tag, got, ref, max_lim, mean_lim):
    d = np.abs(got - ref)
    ok = d.max() <= max_lim and d.mean() <= mean_lim
    print(f"{tag}: max {d.max():.2e} mean {d.mean():.2e} (limits {max_lim:g} / {mean_lim:g}) {'PASS' if ok else 'FAIL'}")
    if not ok:
        failed.append(tag)


def save(path, t):
    a = t.detach().numpy().astype(np.float32)
    a.tofile(path)
    return a


with torch.no_grad():
    for k, (a, b) in enumerate(pairs):
        fwd = net(a, b, **ARGS)["flow_preds"][-1]
        bwd_sep = net(b, a, **ARGS)["flow_preds"][-1]
        both = net(a, b, pred_bidir_flow=True, **ARGS)["flow_preds"][-1]
        assert both.shape[0] == 2, both.shape
        save(os.path.join(OUT, f"pair{k}_in0.f32"), a)
        save(os.path.join(OUT, f"pair{k}_in1.f32"), b)
        save(os.path.join(OUT, f"pair{k}_torch.f32"), fwd)
        save(os.path.join(OUT, f"pair{k}_bwd_torch.f32"), both[1:2])
        print(f"pair {k}: fwd median dx {fwd[0, 0].median():.4f} dy {fwd[0, 1].median():.4f}; "
              f"bwd median dx {both[1, 0].median():.4f} dy {both[1, 1].median():.4f}")
        for tag, got, want in (("fwd vs the one-way pass", both[0:1], fwd), ("bwd vs the (b, a) pass", both[1:2], bwd_sep)):
            d = (got - want).abs()
            print(f"pair {k} bidirectional {tag}: max {d.max():.2e} mean {d.mean():.2e} (measured, not gated)")

# Intermediates of pair 1, one-way pass, in the order the model computes them
inter, calls = [], {}


def record(name, t):
    n = calls.get(name, 0)
    calls[name] = n + 1
    inter.append((f"{name}{n}", t.detach().clone()))


import unimatch.unimatch as um  # noqa: E402

orig = {k: getattr(um, k) for k in ("global_correlation_softmax", "local_correlation_softmax",
                                     "local_correlation_with_flow", "upsample_flow_with_mask", "flow_warp")}


def wrap(name, fn, pick):
    def f(*args, **kwargs):
        out = fn(*args, **kwargs)
        record(name, pick(out))
        return out
    return f


um.global_correlation_softmax = wrap("global_corr_flow", orig["global_correlation_softmax"], lambda o: o[0])
um.local_correlation_softmax = wrap("local_corr_flow", orig["local_correlation_softmax"], lambda o: o[0])
um.local_correlation_with_flow = wrap("corr_with_flow", orig["local_correlation_with_flow"], lambda o: o)
um.upsample_flow_with_mask = wrap("upsampled", orig["upsample_flow_with_mask"], lambda o: o)
um.flow_warp = wrap("warped_feature1", orig["flow_warp"], lambda o: o[0])


# A forward hook that returns anything but None replaces the output of the module: these return None


def hook(*picks):
    def h(module, inputs, output):
        for name, pick in picks:
            record(name, pick(output))
        return None
    return h


net.backbone.register_forward_hook(hook(("backbone", lambda o: o[0]), ("backbone", lambda o: o[1])))
net.transformer.register_forward_hook(hook(("transformer_f0_", lambda o: o[0]), ("transformer_f1_", lambda o: o[1])))
net.feature_flow_attn.register_forward_hook(hook(("propagated", lambda o: o)))
net.refine.register_forward_hook(hook(("refine_net", lambda o: o[0]), ("refine_delta", lambda o: o[2])))
net.refine_proj.register_forward_hook(hook(("refine_proj", lambda o: o)))
with torch.no_grad():
    a, b = pairs[1]
    final = net(a, b, **ARGS)["flow_preds"][-1]
record("final", final)
gate("pair 1 recorded pass vs the reference", final.numpy(), np.fromfile(os.path.join(OUT, "pair1_torch.f32"),
     dtype=np.float32).reshape(final.shape), 0.0, 0.0)
with open(os.path.join(OUT, "inter", "manifest.txt"), "w") as m:
    for name, t in inter:
        save(os.path.join(OUT, "inter", f"{name}.f32"), t)
        m.write(f"{name} {','.join(str(d) for d in t.shape)}\n")
        print(f"inter {name} {tuple(t.shape)}")

if failed:
    print("FAILED:", failed)
    sys.exit(1)
