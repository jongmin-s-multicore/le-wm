"""Compare converted Cloth Move HDF5 rows with the DINO-WM loader of thread0 (branch hanbin5/acwm-phys,
datasets/acwm_phys_dset.py, conf/env/acwm_cloth_move.yaml) on a few episodes and frames.

usage (envs/dino_wm python): python -I check_convert.py <thread0 dir> <clothmove dir> <h5 dir> [--splits ind_test,ood_test]
"""
import argparse, json, sys
from pathlib import Path

import h5py
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("thread0"); ap.add_argument("src"); ap.add_argument("h5dir")
    ap.add_argument("--splits", default="ind_test,ood_test")
    ap.add_argument("--n", type=int, default=6)
    a = ap.parse_args()
    sys.path.insert(0, a.thread0)
    from datasets.acwm_phys_dset import ACWMPhysDataset
    from datasets.img_transforms import default_transform

    dims = [0, 1, 2, 4, 5, 6]
    train = ACWMPhysDataset(a.src, "ind_train", action_dims=dims, normalize_action=True)
    rng = np.random.RandomState(0)
    out = {}
    for split in a.splits.split(","):
        ds = ACWMPhysDataset(a.src, split, transform=default_transform(224), action_dims=dims, normalize_action=True,
                             action_stats=(train.action_mean, train.action_std))
        side = json.loads((Path(a.h5dir) / f"clothmove_{split}.json").read_text())
        assert side["names"] == ds.video_names
        with h5py.File(Path(a.h5dir) / f"clothmove_{split}.h5", "r") as f:
            off, ln = f["ep_offset"][:], f["ep_len"][:]
            assert (ln == np.array(ds.n_obs)).all()
            worst_px, worst_act, nan_ok = 0, 0.0, True
            for e in rng.choice(len(ds), a.n, replace=False):
                start = int(rng.randint(0, 45))
                obs, act, _ = ds.get_slice(int(e), start, start + 20, 5)  # what the DINO-WM training slicer returns
                img = ((obs["visual"].clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).numpy()
                rows = off[e] + np.arange(start, start + 20, 5)
                px = f["pixels"][rows[0] : rows[-1] + 1][::5]
                worst_px = max(worst_px, int(np.abs(px.astype(int) - img.astype(int)).max()))
                raw = f["action"][off[e] + start : off[e] + start + 20]
                z = (torch.from_numpy(raw) - train.action_mean) / train.action_std
                blocks = z.reshape(4, 5 * len(dims))  # (n f) d -> n (f d)
                worst_act = max(worst_act, float((blocks - act).abs().max()))
                last = f["action"][off[e] + ln[e] - 1]
                nan_ok &= bool(np.isnan(last).all()) and not np.isnan(f["action"][off[e] : off[e] + ln[e] - 1]).any()
        out[split] = {"episodes_checked": a.n, "max_abs_pixel_diff": worst_px, "max_abs_action_diff": worst_act,
                      "last_row_nan_only": nan_ok}
        print(split, out[split], flush=True)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
