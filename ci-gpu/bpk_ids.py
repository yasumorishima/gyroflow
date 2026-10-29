"""SHA-256 of a burnpack (.bpk) file with its parameter ids set to zero.

burn gives every parameter a fresh random id each time burn-onnx writes the weights, so two builds of the same ONNX
differ only in those 8-byte ids in the header; loading matches tensors by name. With two files, checks that they
are the same weights apart from the ids.
"""
import hashlib
import struct
import sys

MARK = b"\x68param_id\x1b"  # CBOR: text key "param_id", then a uint64


def normalized(path):
    b = bytearray(open(path, "rb").read())
    assert b[:4] == b"NRUB", f"{path}: not a burnpack file"
    (hlen,) = struct.unpack_from("<I", b, 6)
    end = 10 + hlen
    k = b.find(b"\x67tensors\xb9", 10, end)
    assert k != -1, f"{path}: no tensors map in the header"
    (count,) = struct.unpack_from(">H", b, k + 9)
    n, i = 0, b.find(MARK, 10, end)
    while i != -1:
        j = i + len(MARK)
        b[j:j + 8] = bytes(8)
        n += 1
        i = b.find(MARK, j + 8, end)
    assert n == count, f"{path}: {n} parameter ids for {count} tensors"
    return hashlib.sha256(b).hexdigest(), count, len(b)


assert len(sys.argv) >= 3, "usage: bpk_ids.py FILE FILE..."
got = [normalized(p) for p in sys.argv[1:]]
for p, (h, count, size) in zip(sys.argv[1:], got):
    print(f"{p}: {size} bytes, {count} tensors, SHA-256 with ids zeroed {h}")
assert len({h for h, _, _ in got}) == 1, "different weights"
