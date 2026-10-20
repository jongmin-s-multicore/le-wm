"""ACWM-Phys (HF t1an/ACWM-Phys, mp4 + metadata.pt) to stable-worldmodel HDF5, one file per environment and split.

The layout is the one of the released tworoom.h5: flat per-row columns plus ep_len / ep_offset.
  pixels  (N, 224, 224, 3) uint8: decord frames, torchvision resize on [0, 1] floats (antialias), round(255 x).
          These are the frames of the DINO-WM loaders of issue jongmin-s-multicore/thread0#6 (thread0 branch
          hanbin5/acwm-phys, datasets/acwm_phys_dset.py and acwm_pour_water_dset.py with conf/env/acwm_*.yaml).
  action  (N, A) float32. Row t holds frame t and the action that takes frame t to t+1. When an episode has one
          frame more than actions, its last row is NaN, as in tworoom.h5.
  ep_idx, step_idx (N,), ep_len, ep_offset (episodes,)

  env         folder                     frames            action                                         resize
  cloth_move  deformable/clothmove       65                dims 0,1,2,4,5,6 (3 and 7 are always 1.0); 64   224x224
  push_sand   particle/push_sand         64 (304x512)      pusher pose of the next frame: x, z, cos yaw,    224x224, not cropped
                                                           sin yaw of actions[t+1]; 63
  stack_cube  rigid_dynamics/stack_cube  53-58             all 7 (end-effector delta pose, gripper); T     224x224
  pour_water  particle/pour_water        300 (360x360)     dims 1, 3 (0 and 2 are always 0); 300;          224x224
                                                           ind_train episodes 60-79 left out (no pour)
<out>.json keeps per episode the video name, the group and extra labels (Stack Cube success and OoD direction,
Pour Water first-frame water pixels W0), and the action mean / std of all non-NaN rows (the z-score of train.py).

usage (any python with decord, torchvision, h5py):
  python -I convert_acwm.py <ACWM-Phys dir> <out dir> --env cloth_move [--splits ind_train,ind_test,ood_test]
                            [--workers 8] [--pix-chunk 100] [--max-episodes 0]
  -> <out dir>/<prefix>_<split>.h5 and .json   (prefix: clothmove, pushsand, stackcube, pourwater)
--pix-chunk is the number of frames per HDF5 chunk of pixels: 100 as in tworoom.h5, or 1 for random access (the
LeWM loader reads 20 rows per sample, which is 1-2 chunks of 15 MB with 100).
"""
import argparse, json, os, re, time
from multiprocessing import Pool
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import torchvision.transforms.functional as TF

IMG = 224
ENVS = {
    "cloth_move": dict(path="deformable/clothmove", prefix="clothmove", action="raw", dims=[0, 1, 2, 4, 5, 6], resize="short"),
    "push_sand": dict(path="particle/push_sand", prefix="pushsand", action="pose_next", dims=None, resize="full"),
    "stack_cube": dict(path="rigid_dynamics/stack_cube", prefix="stackcube", action="raw", dims=None, resize="short"),
    "pour_water": dict(path="particle/pour_water", prefix="pourwater", action="raw", dims=[1, 3], resize="short",
                       exclude={"ind_train": set(range(60, 80))}, frames_eq_actions=True),
}
W0_SPLIT = 4000  # Pour Water ood_test first-frame water pixels (360x360): low 1675-2795, high 5360-8366


def pose_next_actions(a):
    nxt = a[1:].float()
    yaw = 2 * torch.atan2(nxt[:, 4], nxt[:, 6])
    return torch.stack([nxt[:, 0], nxt[:, 2], torch.cos(yaw), torch.sin(yaw)], dim=1)


def water_px(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return int(((hsv[..., 0] >= 98) & (hsv[..., 0] <= 110) & (hsv[..., 1] > 70) & (hsv[..., 2] > 100)).sum())


def frames(path, n, resize, w0):
    import decord  # only for decoding; load_split (action statistics) works without it
    torch.set_num_threads(1)
    vr = decord.VideoReader(str(path), num_threads=1)
    assert len(vr) >= n, f"{path}: {len(vr)} frames, need {n}"
    raw = vr.get_batch(list(range(n))).asnumpy()  # (n, H, W, 3) uint8
    x = torch.as_tensor(raw).permute(0, 3, 1, 2).float() / 255.0
    if resize == "full":  # whole frame to 224x224 (DINO-WM resize_transform)
        x = TF.resize(x, [IMG, IMG], antialias=True)
    else:  # shorter side to 224, then the centre 224x224 (DINO-WM default_transform; the frames are square)
        x = TF.center_crop(TF.resize(x, IMG, antialias=True), IMG)
    u8 = (x * 255).round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()
    return u8, (water_px(raw[0]) if w0 else None)


def job(args):
    return frames(*args)


def episode_index(name):
    return int(re.search(r"(\d+)", name).group(1))


def load_split(src, env, split):
    spec = ENVS[env]
    meta = torch.load(src / spec["path"] / split / "metadata.pt", map_location="cpu", weights_only=True)
    excl = spec.get("exclude", {}).get(split, set())
    meta = [e for e in meta if episode_index(Path(e["video_path"]).name) not in excl]
    names = [Path(e["video_path"]).name for e in meta]  # Stack Cube stores absolute paths of the source machine
    if spec["action"] == "pose_next":
        acts = [pose_next_actions(e["actions"]).numpy() for e in meta]
    else:
        dims = spec["dims"] if spec["dims"] is not None else list(range(meta[0]["actions"].shape[1]))
        acts = [e["actions"][:, dims].float().numpy() for e in meta]
    if spec.get("frames_eq_actions"):
        n_obs = [len(a) for a in acts]
        assert all(int(e["length"]) == len(a) for e, a in zip(meta, acts))
    else:
        n_obs = [min(int(e["length"]), len(a) + 1) for e, a in zip(meta, acts)]
        assert all(n >= len(a) for n, a in zip(n_obs, acts))
    return meta, names, acts, n_obs


def convert(src, out, env, split, workers, pix_chunk=100, max_episodes=0):
    spec = ENVS[env]
    meta, names, acts, n_obs = load_split(src, env, split)
    if max_episodes:
        meta, names, acts, n_obs = meta[:max_episodes], names[:max_episodes], acts[:max_episodes], n_obs[:max_episodes]
    ep_len = np.array(n_obs, dtype=np.int32)
    ep_offset = np.concatenate([[0], np.cumsum(ep_len)[:-1]]).astype(np.int64)
    N, A = int(ep_len.sum()), acts[0].shape[1]
    path = out / f"{spec['prefix']}_{split}.h5"
    tmp = path.with_suffix(".h5.tmp")
    t0 = time.time()
    w0s = []
    with h5py.File(tmp, "w") as f:
        pix = f.create_dataset("pixels", (N, IMG, IMG, 3), np.uint8, chunks=(min(pix_chunk, N), IMG, IMG, 3))
        act = np.full((N, A), np.nan, np.float32)
        for e, a in enumerate(acts):
            k = min(len(a), ep_len[e])
            act[ep_offset[e] : ep_offset[e] + k] = a[:k]
        ep_idx = np.repeat(np.arange(len(meta), dtype=np.int32), ep_len)
        step_idx = np.concatenate([np.arange(n, dtype=np.int64) for n in ep_len])
        jobs = [(src / spec["path"] / split / n, k, spec["resize"], env == "pour_water") for n, k in zip(names, n_obs)]
        with Pool(workers) as pool:
            for e, (x, w0) in enumerate(pool.imap(job, jobs, chunksize=2)):
                pix[ep_offset[e] : ep_offset[e] + ep_len[e]] = x
                w0s.append(w0)
                if e % 200 == 0:
                    print(f"{env} {split} {e}/{len(meta)} {time.time() - t0:.0f} s", flush=True)
        for k, v in (("action", act), ("ep_idx", ep_idx), ("step_idx", step_idx), ("ep_len", ep_len), ("ep_offset", ep_offset)):
            f.create_dataset(k, data=v, chunks=(min(1000 if v.ndim > 1 or k in ("ep_idx", "step_idx") else 1024, len(v)),) + v.shape[1:])
    os.replace(tmp, path)
    ok = act[~np.isnan(act).any(1)]
    t = torch.from_numpy(ok)
    groups, extra = [], {}
    if env == "cloth_move":
        groups = ["smaller" if n.startswith("smaller") else "larger" if n.startswith("larger") else "ind" for n in names]
    elif env == "pour_water":
        extra["w0_360px"] = w0s
        groups = ["ind" if split != "ood_test" else ("low" if w < W0_SPLIT else "high") for w in w0s]
    else:
        groups = ["ind" if split != "ood_test" else "ood"] * len(names)
    if env == "stack_cube":
        extra["success"] = [bool(e["success"]) for e in meta]
        extra["ood_label"] = [str(e.get("ood_label", "")) for e in meta]
    side = {"env": env, "source": str(src / spec["path"] / split), "split": split, "action": spec["action"], "action_dims": spec["dims"],
            "resize": spec["resize"], "img_size": IMG, "pix_chunk": pix_chunk, "n_episodes": len(meta), "n_rows": N,
            "action_mean": t.mean(0).tolist(), "action_std": t.std(0).tolist(), "names": names, "groups": groups, "extra": extra}
    path.with_suffix(".json").write_text(json.dumps(side, indent=1))
    print(f"{env} {split}: {len(meta)} episodes, {N} rows -> {path} ({time.time() - t0:.0f} s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("out")
    ap.add_argument("--env", required=True, choices=list(ENVS))
    ap.add_argument("--splits", default="ind_train,ind_test,ood_test")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--pix-chunk", type=int, default=100)
    ap.add_argument("--max-episodes", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for s in a.splits.split(","):
        convert(Path(a.src), out, a.env, s, a.workers, a.pix_chunk, a.max_episodes)


if __name__ == "__main__":
    main()
