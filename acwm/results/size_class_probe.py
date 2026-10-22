"""Cloth Move: which cloth-size group (InD, smaller, larger) a 3-class probe reads from an embedding.

The probe is L2 logistic regression on standardized features, fitted on real-frame embeddings (all 65 frames) of half
of the episodes of every group and applied to the other half (the two folds of acwm/eval_acwm.py). It reads the last
grid frame (60) of the real frame, the 25-step prediction and the rollout from frame 0. A predictor that keeps the
size gives the episode's own group; a predictor that returns to the training distribution gives InD.

  LeWM     eval_cloth_move.npz (acwm/eval_acwm.py): emb, pred_roll, pred25
  DINO-WM  dinowm_cloth_latents.npz (acwm/dinowm_latents.py): mean over patches of the visual latent

usage: python size_class_probe.py [folder]  -> size_class_probe.json
"""
import json, sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

D = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent)
G = ["ind", "smaller", "larger"]


def folds(names, groups):
    f = np.zeros(len(names), int)
    for g in G:
        idx = sorted([i for i in range(len(names)) if groups[i] == g], key=lambda i: names[i])
        for k, i in enumerate(idx):
            f[i] = k % 2
    return f


def run(real_all, last_real, last_25, last_roll, names, groups):
    groups = np.array(groups)
    y_ep = np.array([G.index(g) for g in groups])
    f = folds(names, groups)
    pred = {k: np.zeros(len(names), int) for k in ("real", "25step", "rollout")}
    acc_all = []
    for fold in (0, 1):
        fit = np.where(f != fold)[0]
        X = np.concatenate([real_all[i] for i in fit])
        y = np.concatenate([[y_ep[i]] * len(real_all[i]) for i in fit])
        mu, sd = X.mean(0), X.std(0) + 1e-8
        m = LogisticRegression(C=1.0, max_iter=5000).fit((X - mu) / sd, y)
        te = np.where(f == fold)[0]
        Xt = np.concatenate([real_all[i] for i in te])
        yt = np.concatenate([[y_ep[i]] * len(real_all[i]) for i in te])
        acc_all.append(((m.predict((Xt - mu) / sd)) == yt).mean())
        for k, Z in (("real", last_real), ("25step", last_25), ("rollout", last_roll)):
            pred[k][te] = m.predict((Z[te] - mu) / sd)
    out = {"accuracy_real_all_frames": float(np.mean(acc_all))}
    for g in G:
        sel = groups == g
        out[g] = {"n": int(sel.sum())}
        for k in pred:
            c = np.bincount(pred[k][sel], minlength=3) / sel.sum()
            out[g][k] = {G[i]: float(c[i]) for i in range(3)}
    return out


def main():
    lw = np.load(D / "eval_cloth_move.npz", allow_pickle=True)
    names, groups = [str(n) for n in lw["names"]], [str(g) for g in lw["groups"]]
    f32 = lambda e: np.asarray(e, dtype=np.float32)  # eval_acwm.py saves per-episode arrays as objects
    emb = [f32(e) for e in lw["emb"]]
    roll = np.stack([f32(e)[-1] for e in lw["pred_roll"]])
    p25 = np.stack([f32(e)[-1] for e in lw["pred25"]])
    real = np.stack([e[60] for e in emb])
    res = {"lewm": run(emb, real, p25, roll, names, groups)}
    dw = np.load(D / "dinowm_cloth_latents.npz")
    dn, dg = [str(n) for n in dw["names"]], [str(g) for g in dw["groups"]]
    res["dinowm"] = run(list(dw["z_real"]), dw["z_real"][:, 60], dw["z_25"][:, -1], dw["z_roll"][:, -1], dn, dg)
    (D / "size_class_probe.json").write_text(json.dumps(res, indent=1))
    for model, r in res.items():
        print(f"{model}: 3-class accuracy on held-out real frames {r['accuracy_real_all_frames']:.3f}")
        for g in G:
            row = " | ".join(f"{k} " + " ".join(f"{c}:{100 * v:.0f}%" for c, v in r[g][k].items()) for k in ("real", "25step", "rollout"))
            print(f"  {g:8s} n={r[g]['n']:3d} | {row}")


if __name__ == "__main__":
    main()
