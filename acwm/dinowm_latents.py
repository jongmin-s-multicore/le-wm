"""DINO-WM latents of the Cloth Move test episodes, for the same probe as the LeWM eval (eval_clothmove.py).

Model: our DINO-WM Cloth Move run of issue jongmin-s-multicore/thread0#6 (train_cloth_h1f5: DINOv2 ViT-S/14 frozen,
num_hist 1, frameskip 5). Rollouts are made as in thread0 repro/eval/eval_acwm_phys.py (M3): the first real frame
and all action blocks, frames 0, 5, ..., 60. The 25-step predictions start from the real frame 25 frames earlier.
The visual latent of a frame is 256 patches x 384; the probe uses its mean over the patches (384 numbers). The
rollout latents are also decoded and measured, to check that they give the cloth pixels of #6.

usage (envs/dino_wm python, cwd = thread0 checkout on branch hanbin5/acwm-phys):
  python <le-wm>/acwm/dinowm_latents.py <ckpt_base> <model_name> <h5 dir> <out.npz> [--device cpu] [--epoch 15]
"""
import argparse, json, os, sys, time
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, os.getcwd())
from plan import load_model  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cloth_measure import FS, GRID, T, cloth_px  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_base"); ap.add_argument("model_name"); ap.add_argument("h5dir"); ap.add_argument("out")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--max-eps", type=int, default=0, help="episodes per split, for a quick check")
    ap.add_argument("--epoch", type=int, default=15, help="checkpoint model_<epoch>.pth (#6 evaluated epoch 15); 0 = model_latest.pth")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    torch.set_num_threads(a.threads)
    from omegaconf import OmegaConf
    import hydra

    mp = Path(a.ckpt_base) / "outputs" / a.model_name
    cfg = OmegaConf.load(mp / "hydra.yaml")
    H, f = cfg.num_hist, cfg.frameskip
    assert f == FS
    dev = torch.device(a.device)
    ck = mp / "checkpoints" / (f"model_{a.epoch}.pth" if a.epoch else "model_latest.pth")
    model = load_model(ck, cfg, cfg.num_action_repeat, dev)
    model.eval()  # VWorldModel.eval() returns None
    ds = cfg.env.dataset
    from datasets.acwm_phys_dset import load_acwm_phys_split
    tf = hydra.utils.instantiate(ds.transform)
    kw = dict(n_rollout=ds.n_rollout, exclude=list(ds.exclude), normalize_action=cfg.normalize_action,
              action_dims=list(ds.action_dims), action_format=ds.action_format)
    out = {k: [] for k in ("names", "groups", "z_real", "z_roll", "z_25", "px", "px_dec_roll", "px_dec_real", "px_dec_25")}
    t0 = time.time()
    for split in ("ind_test", "ood_test"):
        base = load_acwm_phys_split(tf, ds.data_path, split, **kw)
        side = json.loads((Path(a.h5dir) / f"clothmove_{split}.json").read_text())
        assert side["names"] == base.video_names
        with h5py.File(Path(a.h5dir) / f"clothmove_{split}.h5", "r") as hf:
            off = hf["ep_offset"][:]
            for e in range(len(base) if not a.max_eps else min(a.max_eps, len(base))):
                vis = base.get_frames(e, range(T))[0]["visual"]  # (65, 3, 224, 224) in [-1, 1], the DINO-WM transform
                zeros = lambda n: torch.zeros(1, n, 1, device=dev)
                zr = model.encode_obs({"visual": vis[None].to(dev), "proprio": zeros(T)})["visual"][0]  # (65, 256, 384)
                act = base.actions[e]  # normalized, (64, 6)
                fo = vis[GRID][None][:, :H].to(dev)
                fa = act[: FS * (len(GRID) - 1)].reshape(1, len(GRID) - 1, -1).to(dev)
                zp, _ = model.rollout({"visual": fo, "proprio": zeros(H)}, fa)
                zroll = zp["visual"][0]  # (13, 256, 384), frame 0 is the encoded real frame
                dec = model.decode_obs({"visual": zroll[None], "proprio": zp["proprio"]})[0]["visual"][0]
                dec_u8 = ((dec.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
                dreal = model.decode_obs({"visual": zr[GRID][None], "proprio": zeros(len(GRID))})[0]["visual"][0]
                dreal_u8 = ((dreal.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
                z25 = torch.full((len(GRID), zr.shape[-1]), float("nan"))
                px25 = np.full(len(GRID), -1)
                for j in range(5, len(GRID)):
                    i = j - 5
                    o = vis[GRID[i] : GRID[i] + 1][None].to(dev)
                    aa = act[GRID[i] : GRID[i] + FS * 5].reshape(1, 5, -1).to(dev)
                    z5, _ = model.rollout({"visual": o, "proprio": zeros(1)}, aa)
                    z25[j] = z5["visual"][0, -1].mean(0).cpu()
                    d5 = model.decode_obs({"visual": z5["visual"][:, -1:], "proprio": z5["proprio"][:, -1:]})[0]["visual"][0, 0]
                    px25[j] = cloth_px(((d5.clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy())
                raw = hf["pixels"][off[e] : off[e] + T]
                out["names"].append(base.video_names[e]); out["groups"].append(side["groups"][e])
                out["z_real"].append(zr.mean(1).cpu().numpy()); out["z_roll"].append(zroll.mean(1).cpu().numpy())
                out["z_25"].append(z25.numpy()); out["px"].append(np.array([cloth_px(x) for x in raw]))
                out["px_dec_roll"].append(np.array([cloth_px(x) for x in dec_u8]))
                out["px_dec_real"].append(np.array([cloth_px(x) for x in dreal_u8]))
                out["px_dec_25"].append(px25)
                if e % 20 == 0:
                    print(f"{split} {e}/{len(base)} {time.time() - t0:.0f} s", flush=True)
    np.savez_compressed(a.out, **{k: np.array(v) for k, v in out.items()}, num_hist=H, model=a.model_name, ckpt=ck.name)
    print(f"done {time.time() - t0:.0f} s -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
