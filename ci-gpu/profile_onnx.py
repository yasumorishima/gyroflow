"""List the big intermediate tensors of an ONNX graph, biggest first.

A GPU backend allocates each tensor as one buffer, and Vulkan only guarantees 128 MiB for one
storage buffer (maxStorageBufferRange), so every tensor above that is a device on which gmflow
cannot run. Usage: python profile_onnx.py model.onnx [threshold MiB, default 16]
Exits 1 if any tensor's shape cannot be inferred (a size we cannot see is not a size we checked).
"""
import sys

import numpy as np
import onnx

path = sys.argv[1]
limit_mib = float(sys.argv[2]) if len(sys.argv) > 2 else 16.0
model = onnx.shape_inference.infer_shapes(onnx.load(path), strict_mode=True)
g = model.graph

itemsize = {onnx.TensorProto.FLOAT: 4, onnx.TensorProto.FLOAT16: 2, onnx.TensorProto.DOUBLE: 8,
            onnx.TensorProto.INT64: 8, onnx.TensorProto.INT32: 4, onnx.TensorProto.BOOL: 1,
            onnx.TensorProto.UINT8: 1, onnx.TensorProto.INT8: 1}
info = {}
for v in list(g.value_info) + list(g.output) + list(g.input):
    t = v.type.tensor_type
    dims = [d.dim_value if d.HasField("dim_value") else None for d in t.shape.dim] if t.HasField("shape") else None
    info[v.name] = (t.elem_type, dims)

unknown, rows = [], []
for n in g.node:
    for o in n.output:
        if not o:
            continue
        et, dims = info.get(o, (None, None))
        if dims is None or any(d is None for d in dims) or et not in itemsize:
            unknown.append((n.op_type, n.name, o))
            continue
        nbytes = int(np.prod(dims, dtype=np.int64)) * itemsize[et] if dims else itemsize[et]
        rows.append((nbytes, n.op_type, n.name, dims))

rows.sort(key=lambda r: -r[0])
mib = lambda b: b / 2**20
print(f"{path}: {len(g.node)} nodes, {len(rows)} sized outputs, {len(unknown)} unsized")
print(f"largest: {mib(rows[0][0]):.1f} MiB  ({rows[0][1]} {rows[0][2]} {rows[0][3]})")
for cap in (64, 128, 256):
    print(f"outputs over {cap} MiB: {sum(1 for r in rows if mib(r[0]) > cap)}")
print(f"--- outputs over {limit_mib:g} MiB")
for nbytes, op, name, dims in rows:
    if mib(nbytes) <= limit_mib:
        break
    print(f"{mib(nbytes):8.1f} MiB  {op:12s} {dims}  {name}")
if unknown:
    print("--- outputs without a static shape")
    for u in unknown[:30]:
        print("   ", u)
    sys.exit(1)
# Optional third argument: fail if any output reaches this size (MiB)
if len(sys.argv) > 3 and mib(rows[0][0]) >= float(sys.argv[3]):
    print(f"FAIL: largest output {mib(rows[0][0]):.1f} MiB >= {sys.argv[3]} MiB")
    sys.exit(1)
