"""ACWM-Phys Cloth Move (HF t1an/ACWM-Phys, mp4 + metadata.pt) to stable-worldmodel HDF5, one file per split.

The layout is the one of the released tworoom.h5: flat per-row columns plus ep_len / ep_offset.
  pixels     (N, 224, 224, 3) uint8. decord frames, torchvision Resize(224) (antialias) on [0, 1] floats, then
             round(255 x). These are the frames the DINO-WM loader gives (thread0 datasets/acwm_phys_dset.py).
  action     (N, 6) float32, dims [0, 1, 2, 4, 5, 6] of the metadata actions (dims 3 and 7 are 1.0 everywhere,
             a z-score would divide them by 0). Row t holds frame t and the action that takes frame t to t+1.
             An episode has 65 frames and 64 actions, so its last row is NaN, as in tworoom.h5.
  ep_idx, step_idx (N,), ep_len, ep_offset (episodes,)
<out>.json keeps the video name and group (ind, smaller, larger) of each episode in row order.

usage (envs/dino_wm python: decord, torchvision 0.18, h5py):
  python -I convert_clothmove.py <clothmove dir> <out dir> [--splits ind_train,ind_test,ood_test] [--workers 8]
                                 [--pix-chunk 100]
  -> <out dir>/clothmove_<split>.h5 and .json
--pix-chunk is the number of frames per HDF5 chunk of pixels: 100 as in tworoom.h5, or 1 for random access (the
LeWM loader reads 20 rows per sample, which is 1-2 chunks of 15 MB with 100).
"""
import argparse, json, os, time
from multiprocessing import Pool
from pathlib import Path

import decord
import h5py
import numpy as np
import torch
import torchvision.transforms.functional as TF

ACTION_DIMS = [0, 1, 2, 4, 5, 6]
IMG = 224


def frames(path, n):
    torch.set_num_threads(1)
    vr = decord.VideoReader(str(path), num_threads=1)
    assert len(vr) >= n, f"{path}: {len(vr)} frames, need {n}"
    x = torch.as_tensor(vr.get_batch(list(range(n))).asnumpy())  # (n, H, W, 3) uint8
    x = TF.resize(x.permute(0, 3, 1, 2).float() / 255.0, IMG, antialias=True)
    return (x * 255).round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()


def job(args):
    return frames(*args)


def group(name):
    return "smaller" if name.startswith("smaller") else "larger" if name.startswith("larger") else "ind"


def convert(src, out, split, workers, pix_chunk=100):
    meta = torch.load(src / split / "metadata.pt", map_location="cpu", weights_only=True)
    names = [Path(e["video_path"]).name for e in meta]
    acts = [e["actions"][:, ACTION_DIMS].float().numpy() for e in meta]
    n_obs = [min(int(e["length"]), len(a) + 1) for e, a in zip(meta, acts)]
    assert all(n == len(a) + 1 for n, a in zip(n_obs, acts)), "expected one more frame than actions"
    ep_len = np.array(n_obs, dtype=np.int32)
    ep_offset = np.concatenate([[0], np.cumsum(ep_len)[:-1]]).astype(np.int64)
    N = int(ep_len.sum())
    path = out / f"clothmove_{split}.h5"
    tmp = path.with_suffix(".h5.tmp")
    t0 = time.time()
    with h5py.File(tmp, "w") as f:
        pix = f.create_dataset("pixels", (N, IMG, IMG, 3), np.uint8, chunks=(pix_chunk, IMG, IMG, 3))
        act = np.full((N, len(ACTION_DIMS)), np.nan, np.float32)
        ep_idx = np.repeat(np.arange(len(meta), dtype=np.int32), ep_len)
        step_idx = np.concatenate([np.arange(n, dtype=np.int64) for n in ep_len])
        for e, a in enumerate(acts):
            act[ep_offset[e] : ep_offset[e] + len(a)] = a
        with Pool(workers) as pool:
            for e, x in enumerate(pool.imap(job, [(src / split / n, k) for n, k in zip(names, n_obs)], chunksize=4)):
                pix[ep_offset[e] : ep_offset[e] + ep_len[e]] = x
                if e % 200 == 0:
                    print(f"{split} {e}/{len(meta)} {time.time() - t0:.0f} s", flush=True)
        f.create_dataset("action", data=act, chunks=(1000, len(ACTION_DIMS)))
        f.create_dataset("ep_idx", data=ep_idx, chunks=(1000,))
        f.create_dataset("step_idx", data=step_idx, chunks=(1000,))
        f.create_dataset("ep_len", data=ep_len, chunks=(min(1024, len(ep_len)),))
        f.create_dataset("ep_offset", data=ep_offset, chunks=(min(1024, len(ep_len)),))
    os.replace(tmp, path)
    side = {"source": str(src / split), "split": split, "action_dims": ACTION_DIMS, "img_size": IMG, "pix_chunk": pix_chunk,
            "n_episodes": len(meta), "n_rows": N, "names": names, "groups": [group(n) for n in names]}
    path.with_suffix(".json").write_text(json.dumps(side, indent=1))
    print(f"{split}: {len(meta)} episodes, {N} rows -> {path} ({time.time() - t0:.0f} s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("out")
    ap.add_argument("--splits", default="ind_train,ind_test,ood_test")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--pix-chunk", type=int, default=100)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for s in a.splits.split(","):
        convert(Path(a.src), out, s, a.workers, a.pix_chunk)


if __name__ == "__main__":
    main()
