"""The benzene model file: a header (magic HXMD and a JSON tensor table) followed by 4096-aligned blobs, plus the table
of fixed-size records next to it that the host runner reads (<model>.tab, benzene/host/model_file.h).

A layer blob is a byte string of 4096-aligned parts (weight streams, fp32 vectors) plus the part offsets, in the order
of a part enum of benzene/include/benzene/protocol.h (e.g. cosmos3_edge::llm_layer_part,
vision_transformer::block_part).
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

HEADER = 1 << 20  # reserved for the magic + JSON; blobs start here


def pack_parts(parts: list) -> tuple[bytes, np.ndarray]:
    blob, offs = bytearray(), []
    for prt in parts:
        b = prt if isinstance(prt, bytes) else np.ascontiguousarray(prt, np.float32).tobytes()
        offs.append(len(blob))
        blob += b
        blob += bytes((-len(blob)) % 4096)
    return bytes(blob), np.array(offs, np.int32)


class Writer:
    """Streams blobs to disk after a reserved header, then writes the JSON table into the header."""

    def __init__(self, path: Path):
        self.path, self.tensors = path, {}
        self.f = open(path, "wb")
        self.f.write(bytes(HEADER))

    def _add(self, data: bytes) -> int:
        off = self.f.tell()
        self.f.write(data)
        self.f.write(bytes((-self.f.tell()) % 4096))
        return off

    def blob(self, name: str, blob_offs: tuple[bytes, np.ndarray]):
        blob, offs = blob_offs
        self.tensors[name] = {"off": self._add(blob), "size": len(blob), "parts": [int(o) for o in offs]}

    def f32(self, name: str, a: np.ndarray):
        a = np.ascontiguousarray(a, np.float32)
        self.tensors[name] = {"off": self._add(a.tobytes()), "size": a.nbytes, "dtype": "f32", "shape": list(a.shape)}

    def raw(self, name: str, data: bytes, **meta):
        self.tensors[name] = {"off": self._add(data), "size": len(data), **meta}

    def finish(self, header: dict) -> int:
        size = self.f.tell()
        js = json.dumps({**header, "tensors": self.tensors}).encode()
        assert 16 + len(js) <= HEADER, "header table too large"
        self.f.seek(0)
        self.f.write(b"HXMD" + struct.pack("<III", 1, len(js), 0) + js)
        self.f.close()
        # fixed-size records for the C runtime: name[48], u64 off, u64 size, u32 kind (0 blob, 1 f32, 2 raw),
        # u32 nparts, u32 parts[24], u32 ndim, u32 shape[4], u32 aux (raw: columns) -> 192 bytes
        with open(self.path.with_suffix(".tab"), "wb") as t:
            t.write(b"HXTB" + struct.pack("<I", len(self.tensors)))
            for name, e in self.tensors.items():
                parts = e.get("parts", [])
                shape = e.get("shape", [])
                kind = 0 if "parts" in e else 1 if e.get("dtype") == "f32" else 2
                assert len(name) < 48 and len(parts) <= 24 and len(shape) <= 4
                t.write(name.encode().ljust(48, b"\0"))
                t.write(struct.pack("<QQII", e["off"], e["size"], kind, len(parts)))
                t.write(struct.pack("<24I", *(parts + [0] * (24 - len(parts)))))
                t.write(struct.pack("<I4II", len(shape), *(shape + [0] * (4 - len(shape))), int(e.get("cols", 0))))
        return size


def load_export(path) -> dict:
    """{tensor name: (bytes, meta)} of a model file."""
    d = open(path, "rb").read()
    n = struct.unpack("<I", d[8:12])[0]
    t = json.loads(d[16 : 16 + n])["tensors"]
    return {k: (d[v["off"] : v["off"] + v["size"]], v) for k, v in t.items()}
