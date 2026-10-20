"""The cloth-pixel probe of eval_clothmove.py on DINO-WM latents (dinowm_latents.py output), with the same folds
(even / odd episode of every group, sorted by name), the same probes (ridge, MLP) and the same table.
Features: the mean over the 256 patches of the DINO-WM visual latent (384 numbers per frame).

usage: python acwm/probe_dinowm.py <dinowm_latents.npz> <out prefix>   -> <out prefix>.json
"""
import json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_clothmove import MLP, Ridge, DINO_WM_6, boot, r2  # noqa: E402
from cloth_measure import GRID, T  # noqa: E402

GROUPS = ["ind", "smaller", "larger"]


def main():
    z = np.load(sys.argv[1])
    out_prefix = sys.argv[2]
    names, groups = [str(n) for n in z["names"]], [str(g) for g in z["groups"]]
    fold = np.zeros(len(names), int)
    for g in GROUPS:
        idx = sorted([i for i in range(len(names)) if groups[i] == g], key=lambda i: names[i])
        for k, i in enumerate(idx):
            fold[i] = k % 2
    zr, zroll, z25, px = z["z_real"], z["z_roll"], z["z_25"], z["px"].astype(float)
    res = {"source": sys.argv[1], "features": "mean over patches of the DINO-WM visual latent", "dim": int(zr.shape[-1]),
           "decoded": {}, "probe": {}}
    G = {g: np.array([i for i in range(len(names)) if groups[i] == g]) for g in GROUPS}
    for g, m in G.items():  # decoded rollout / Enc->Dec / 25-step, to compare with #6
        res["decoded"][g] = {"n": len(m), "real_px_last": boot(px[m][:, GRID[-1]]), "dec_rollout_last": boot(z["px_dec_roll"][m][:, -1]),
                             "dec_real_last": boot(z["px_dec_real"][m][:, -1]), "dec_25step_last": boot(z["px_dec_25"][m][:, -1]),
                             "issue6": DINO_WM_6[g]}
    for pname, P in (("ridge", Ridge), ("mlp", MLP)):
        o = {k: np.full(v, np.nan) for k, v in (("real_all", px.shape), ("real", (len(names), len(GRID))), ("roll", (len(names), len(GRID))), ("p25", (len(names), len(GRID))))}
        info = []
        for f in (0, 1):
            fit = np.where(fold != f)[0]
            X = zr[fit].reshape(-1, zr.shape[-1])
            y = px[fit].reshape(-1)
            grp = np.repeat(np.arange(len(fit)), T)
            pr = P().fit(X, y, grp)
            info.append({"fold": f, "n_fit_frames": len(y), "alpha": getattr(pr, "alpha", None), "best_epoch": getattr(pr, "best_epoch", None)})
            for i in np.where(fold == f)[0]:
                o["real_all"][i] = pr.predict(zr[i])
                o["real"][i] = o["real_all"][i][GRID]
                o["roll"][i] = pr.predict(zroll[i])
                ok = np.isfinite(z25[i]).all(1)
                o["p25"][i, ok] = pr.predict(z25[i][ok])
        r = {"folds": info, "r2_real_all_frames": r2(px.ravel(), o["real_all"].ravel()), "mae_real_all_frames": float(np.abs(px - o["real_all"]).mean())}
        for g, m in G.items():
            r[g] = {"n": len(m), "r2_real_all_frames": r2(px[m].ravel(), o["real_all"][m].ravel()),
                    "mae_real_all_frames": float(np.abs(px[m] - o["real_all"][m]).mean()),
                    "last": {"real_px": boot(px[m][:, GRID[-1]]), "probe_real": boot(o["real"][m][:, -1]),
                             "probe_rollout": boot(o["roll"][m][:, -1]), "probe_25step": boot(o["p25"][m][:, -1])},
                    "curve": {"real_px": px[m][:, GRID].mean(0).tolist(), "probe_real": o["real"][m].mean(0).tolist(),
                              "probe_rollout": o["roll"][m].mean(0).tolist(), "probe_25step": np.nanmean(o["p25"][m], 0).tolist()}}
        res["probe"][pname] = r
        np.savez_compressed(f"{out_prefix}_{pname}.npz", names=np.array(names), groups=np.array(groups), **o)
    Path(out_prefix + ".json").write_text(json.dumps(res, indent=1))
    for g in GROUPS:
        d = res["decoded"][g]
        print(f"{g:8s} decoded last: real {d['real_px_last'][0]:.0f} rollout {d['dec_rollout_last'][0]:.0f} enc-dec {d['dec_real_last'][0]:.0f} "
              f"25-step {d['dec_25step_last'][0]:.0f} | #6 {d['issue6']}")
    for pname in ("ridge", "mlp"):
        r = res["probe"][pname]
        print(f"[{pname}] R2 real frames {r['r2_real_all_frames']:.3f}")
        for g in GROUPS:
            L = r[g]["last"]
            print(f"  {g:8s} last: real {L['real_px'][0]:.0f} probe(real) {L['probe_real'][0]:.0f} probe(rollout) {L['probe_rollout'][0]:.0f} probe(25-step) {L['probe_25step'][0]:.0f}")


if __name__ == "__main__":
    main()
