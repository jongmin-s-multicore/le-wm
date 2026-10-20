"""Compare converted ACWM-Phys HDF5 rows (convert_acwm.py) with the DINO-WM loaders of thread0 (branch
hanbin5/acwm-phys: datasets/acwm_phys_dset.py, acwm_pour_water_dset.py, conf/env/acwm_*.yaml) on random training
windows: the 4 frames at stride 5 and the 4 action blocks of 5 actions, normalized with the ind_train statistics.

usage (envs/dino_wm python): python -I check_acwm.py <thread0 dir> <ACWM-Phys dir> <h5 dir> --env push_sand
                             [--splits ind_test,ood_test] [--n 8]
"""
import argparse, json, sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_acwm import ENVS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("thread0"); ap.add_argument("src"); ap.add_argument("h5dir")
    ap.add_argument("--env", required=True, choices=list(ENVS))
    ap.add_argument("--splits", default="ind_test,ood_test")
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    sys.path.insert(0, a.thread0)
    from datasets.img_transforms import default_transform
    spec, root = ENVS[a.env], Path(a.src) / ENVS[a.env]["path"]
    if a.env == "pour_water":
        from datasets.acwm_pour_water_dset import ACWMPourWaterDataset as D
        kw = dict(exclude=list(range(60, 80)))
        tf = default_transform(224)
    else:
        from datasets.acwm_phys_dset import ACWMPhysDataset as D, resize_transform
        kw = dict(action_dims=spec["dims"], action_format=spec["action"])
        tf = resize_transform(224) if spec["resize"] == "full" else default_transform(224)
    train = D(str(root), "ind_train", normalize_action=True, **kw)
    mean, std = train.action_mean, train.action_std
    rng = np.random.RandomState(0)
    out = {}
    for split in a.splits.split(","):
        ds = D(str(root), split, transform=tf, normalize_action=True, action_stats=(mean, std), **kw)
        side = json.loads((Path(a.h5dir) / f"{spec['prefix']}_{split}.json").read_text())
        names = [Path(n).name for n in ds.video_names]
        n_ep = side["n_episodes"]
        assert side["names"] == names[:n_ep], "episode order differs"
        with h5py.File(Path(a.h5dir) / f"{spec['prefix']}_{split}.h5", "r") as f:
            off, ln = f["ep_offset"][:], f["ep_len"][:]
            worst_px, worst_act, nan_rows = 0, 0.0, 0
            for e in rng.choice(n_ep, min(a.n, n_ep), replace=False):
                start = int(rng.randint(0, ds.get_seq_length(int(e)) - 19))  # DINO-WM windows end at the last action
                obs, act, _ = ds.get_slice(int(e), start, start + 20, 5)
                img = ((obs["visual"].clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).numpy()
                px = f["pixels"][off[e] + start : off[e] + start + 16][::5]
                worst_px = max(worst_px, int(np.abs(px.astype(int) - img.astype(int)).max()))
                raw = torch.from_numpy(f["action"][off[e] + start : off[e] + start + 20])
                z = torch.nan_to_num((raw - mean) / std, 0.0).reshape(4, -1)
                k = min(4, act.shape[0])
                ref = torch.nan_to_num(act[:k].float(), 0.0)
                worst_act = max(worst_act, float((z[:k] - ref).abs().max()))
                nan_rows += int(np.isnan(f["action"][off[e] : off[e] + ln[e]]).any(1).sum())
        out[split] = {"episodes_checked": min(a.n, n_ep), "max_abs_pixel_diff": worst_px, "max_abs_action_diff": worst_act,
                      "nan_action_rows_in_checked_episodes": nan_rows, "rows_per_episode": sorted(set(int(x) for x in ln))}
        print(a.env, split, out[split], flush=True)
    print(json.dumps({a.env: out}))


if __name__ == "__main__":
    main()
