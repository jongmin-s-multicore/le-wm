"""sha256 of the arrays of converted HDF5 files (pixels, action, ep_len, ep_offset), independent of the chunk layout.
Used to check that a copy converted on another machine has the same content.

usage: python -I h5_digest.py <file.h5> [...]   -> one JSON line per file
"""
import hashlib, json, sys

import h5py
import numpy as np


def digest(path):
    out = {"file": path}
    with h5py.File(path, "r") as f:
        for k in ("pixels", "action", "ep_len", "ep_offset"):
            d, h = f[k], hashlib.sha256()
            for i in range(0, d.shape[0], 2000):
                h.update(np.ascontiguousarray(d[i : i + 2000]).tobytes())
            out[k] = {"shape": list(d.shape), "dtype": str(d.dtype), "sha256": h.hexdigest()}
    return out


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(json.dumps(digest(p)), flush=True)
