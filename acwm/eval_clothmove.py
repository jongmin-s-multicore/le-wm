"""LeWM on ACWM-Phys Cloth Move: embedding prediction error and a probe of the visible cloth amount.

There is no simulator, so there is no planning. Test sets: ind_test (100 episodes), ood_test smaller (50) and
larger (50). Frames are on the 5-frame grid 0, 5, ..., 60 (65 frames, 64 actions); action block k = actions 5k..5k+4.

(a) Embedding error, MSE over the 192 dims, against copy-last (the last real embedding the prediction starts from):
    1-step   real frames s, s+5, s+10 and their action blocks -> frame s+15, every s in 0..45 (the training windows)
    rollout  real frame 0 only, then the model's own predictions with all action blocks -> frames 5..60
    25-step  for every frame t in 25..60: real frames up to t-25 (the last 3 on the grid), then 5 predicted steps
(b) Probe from embedding to the visible cloth pixels of a 224x224 frame (orange front + pink back, the HSV mask of
    thread0 repro/eval/eval_acwm_phys_quant.py). Two folds: half of the episodes of every group (even / odd index) fit
    the probe on real embeddings of all 65 frames, the other half is measured; then the halves swap. The probe is
    applied to real embeddings (does the encoder keep the size?) and to predicted ones (does the predictor pull the
    size back to the training value?). Ridge regression (alpha by episode-grouped CV) and a small MLP.

usage (cwd = le-wm checkout):
  python acwm/eval_clothmove.py <run dir with config.json and weights_*.pt> <h5 dir> <out prefix> [--weights FILE]
  -> <out prefix>.json, <out prefix>.npz
"""
import argparse, json, sys, time
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FS, HIST, T = 5, 3, 65
GRID = list(range(0, T - 1, FS))  # 0..60, 13 frames
DINO_WM_6 = {  # issue jongmin-s-multicore/thread0#6, table 2, last frame (60), cloth pixels at 224 px
    "ind": {"real": 1819, "enc_dec": 1790, "step25": 1801, "rollout": 1888},
    "smaller": {"real": 404, "enc_dec": 388, "step25": 840, "rollout": 1129},
    "larger": {"real": 4475, "enc_dec": 4352, "step25": 2773, "rollout": 1836},
}


def hue(hsv, lo, hi, s, v=80):
    h = hsv[..., 0]
    m = (h >= lo) & (h <= hi) if lo <= hi else ((h >= lo) | (h <= hi))
    return m & (hsv[..., 1] > s) & (hsv[..., 2] > v)


def cloth_px(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return int((hue(hsv, 8, 16, 120) | hue(hsv, 158, 170, 60)).sum())


def load_model(run, weights, dev):
    import hydra
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(json.loads((run / "config.json").read_text()))
    model = hydra.utils.instantiate(cfg)
    sd = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=True)
    return model.to(dev).eval(), cfg


def action_stats(train_h5):
    """z-score of train.py get_column_normalizer: all rows of ind_train without NaN, torch mean / std"""
    with h5py.File(train_h5, "r") as f:
        a = torch.from_numpy(np.array(f["action"][:]))
    a = a[~torch.isnan(a).any(dim=1)]
    return a.mean(0, keepdim=True), a.std(0, keepdim=True)


def load_split(h5dir, split, mean, std):
    side = json.loads((h5dir / f"clothmove_{split}.json").read_text())
    eps = []
    with h5py.File(h5dir / f"clothmove_{split}.h5", "r") as f:
        off, ln = f["ep_offset"][:], f["ep_len"][:]
        for e in range(len(off)):
            assert ln[e] == T
            raw = f["pixels"][off[e] : off[e] + T]  # (T, 224, 224, 3) uint8
            px = np.array([cloth_px(x) for x in raw])
            act = ((torch.from_numpy(f["action"][off[e] : off[e] + T]) - mean) / std).float()
            act = torch.nan_to_num(act, 0.0)
            eps.append({"name": side["names"][e], "group": side["groups"][e], "raw": raw, "action": act, "px": px})
    return eps


_PREP = None


@torch.no_grad()
def encode(model, raw, dev):
    """raw (T, H, W, 3) uint8 -> (T, D), with the image transform of train.py (ImageNet normalisation, 224)"""
    global _PREP
    if _PREP is None:
        from utils import get_img_preprocessor
        _PREP = get_img_preprocessor(source="pixels", target="pixels", img_size=224)
    pix = _PREP({"pixels": torch.from_numpy(raw).permute(0, 3, 1, 2)})["pixels"]
    return model.encode({"pixels": pix.unsqueeze(0).to(dev)})["emb"][0].float().cpu()


@torch.no_grad()
def predict_last(model, ctx, acts, dev):
    """ctx (B, t, D) embeddings, acts (B, t, 30) action blocks -> prediction for the frame after the last one"""
    a = model.action_encoder(acts.to(dev))
    return model.predict(ctx.to(dev), a)[:, -1].float().cpu()


def blocks(act, starts):
    """action blocks (len(starts), 30): actions s..s+4 for every start s"""
    return torch.stack([act[s : s + FS].reshape(-1) for s in starts])


@torch.no_grad()
def rollout(model, ctx, acts, n, dev):
    """ctx (B, h, D) real embeddings with action blocks acts (B, h + n - 1, 30); n predicted steps, history HIST"""
    emb = ctx.to(dev)
    h = ctx.shape[1]
    preds = []
    for k in range(n):
        lo = max(0, emb.shape[1] - HIST)
        a = model.action_encoder(acts[:, lo : h + k].to(dev))
        p = model.predict(emb[:, lo:], a)[:, -1:]
        emb = torch.cat([emb, p], 1)
        preds.append(p[:, 0].float().cpu())
    return torch.stack(preds, 1)  # (B, n, D)


def mse(a, b):
    return ((a - b) ** 2).mean(-1)


def boot(x, n=2000, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return [None, None, None]
    b = x[np.random.RandomState(seed).randint(0, len(x), (n, len(x)))].mean(1)
    return [float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


class Ridge:
    def __init__(self, alphas=(1e-2, 1e-1, 1, 10, 100, 1000)):
        self.alphas = alphas

    @staticmethod
    def _fit(X, y, alpha):
        mu, sd = X.mean(0), X.std(0) + 1e-8
        Z = (X - mu) / sd
        A = Z.T @ Z + alpha * np.eye(Z.shape[1])
        w = np.linalg.solve(A, Z.T @ (y - y.mean()))
        return mu, sd, w, y.mean()

    @staticmethod
    def _pred(p, X):
        mu, sd, w, b = p
        return ((X - mu) / sd) @ w + b

    def fit(self, X, y, groups, k=5):
        ug = np.unique(groups)
        fold = {g: i % k for i, g in enumerate(np.random.RandomState(0).permutation(ug))}
        f = np.array([fold[g] for g in groups])
        err = []
        for al in self.alphas:
            e = 0.0
            for i in range(k):
                p = self._fit(X[f != i], y[f != i], al)
                e += ((self._pred(p, X[f == i]) - y[f == i]) ** 2).sum()
            err.append(e)
        self.alpha = self.alphas[int(np.argmin(err))]
        self.p = self._fit(X, y, self.alpha)
        return self

    def predict(self, X):
        return self._pred(self.p, X)


class MLP:
    def __init__(self, hidden=256, epochs=400, lr=1e-3, wd=1e-4, seed=0):
        self.h, self.epochs, self.lr, self.wd, self.seed = hidden, epochs, lr, wd, seed

    def fit(self, X, y, groups):
        torch.manual_seed(self.seed)
        ug = np.unique(groups)
        val_g = set(np.random.RandomState(self.seed).permutation(ug)[: max(1, len(ug) // 10)])
        v = np.array([g in val_g for g in groups])
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-8
        self.ys = float(y.std()) + 1e-8
        Xt = torch.tensor((X - self.mu) / self.sd, dtype=torch.float32)
        yt = torch.tensor((y - y.mean()) / self.ys, dtype=torch.float32)
        self.ym = float(y.mean())
        net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], self.h), torch.nn.ReLU(), torch.nn.Linear(self.h, self.h),
                                  torch.nn.ReLU(), torch.nn.Linear(self.h, 1))
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.wd)
        tr, va = torch.tensor(~v), torch.tensor(v)
        best, best_sd = 1e18, None
        for ep in range(self.epochs):
            perm = torch.nonzero(tr)[:, 0][torch.randperm(int(tr.sum()))]
            for i in range(0, len(perm), 256):
                j = perm[i : i + 256]
                loss = ((net(Xt[j])[:, 0] - yt[j]) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                vl = float(((net(Xt[va])[:, 0] - yt[va]) ** 2).mean())
            if vl < best:
                best, best_sd, self.best_epoch = vl, {k: t.clone() for k, t in net.state_dict().items()}, ep
        net.load_state_dict(best_sd)
        self.net = net.eval()
        return self

    def predict(self, X):
        with torch.no_grad():
            z = self.net(torch.tensor((X - self.mu) / self.sd, dtype=torch.float32))[:, 0].numpy()
        return z * self.ys + self.ym


def r2(y, p):
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("h5dir"); ap.add_argument("out")
    ap.add_argument("--weights", default=None, help="default: the weights_epoch_*.pt with the highest epoch")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    run, h5dir, dev = Path(a.run), Path(a.h5dir), torch.device(a.device)
    weights = Path(a.weights) if a.weights else max(run.glob("weights_epoch_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
    t0 = time.time()
    model, mcfg = load_model(run, weights, dev)
    mean, std = action_stats(h5dir / "clothmove_ind_train.h5")
    eps = load_split(h5dir, "ind_test", mean, std) + load_split(h5dir, "ood_test", mean, std)
    print(f"loaded {len(eps)} episodes, {time.time() - t0:.0f} s", flush=True)

    starts1 = list(range(0, T - 4 * FS + 1))  # 0..45
    for ep in eps:
        e = encode(model, ep.pop("raw"), dev)
        ep["emb"] = e
        act = ep["action"]
        # 1-step, every training window
        ctx = torch.stack([e[[s, s + FS, s + 2 * FS]] for s in starts1])
        ab = torch.stack([blocks(act, [s, s + FS, s + 2 * FS]) for s in starts1])
        p1 = predict_last(model, ctx, ab, dev)
        tgt = e[[s + 3 * FS for s in starts1]]
        ep["err1"] = mse(p1, tgt).numpy()
        ep["err1_copy"] = mse(e[[s + 2 * FS for s in starts1]], tgt).numpy()
        ep["pred1_grid"] = {s + 3 * FS: p1[i] for i, s in enumerate(starts1) if (s + 3 * FS) % FS == 0}
        # rollout from frame 0
        AB = blocks(act, GRID[:-1])  # (12, 30) blocks 0..11
        pr = rollout(model, e[[0]].unsqueeze(0), AB.unsqueeze(0), len(GRID) - 1, dev)[0]  # frames 5..60
        ep["pred_roll"] = torch.cat([e[[0]], pr])  # (13, D), frame 0 is real
        g = e[GRID]
        ep["err_roll"] = mse(pr, g[1:]).numpy()
        ep["err_roll_copy"] = mse(e[[0]].expand_as(g[1:]), g[1:]).numpy()
        # 25-step: target grid index j in 5..12 from real grid frames up to j-5
        p25, c25 = [], []
        for j in range(5, len(GRID)):
            i = j - 5
            lo = max(0, i - HIST + 1)
            ctxj = g[lo : i + 1].unsqueeze(0)
            acts = AB[lo:j].unsqueeze(0)
            p25.append(rollout(model, ctxj, acts, 5, dev)[0, -1])
            c25.append(g[i])
        ep["pred25"] = torch.stack(p25)  # frames 25..60
        ep["err25"] = mse(ep["pred25"], g[5:]).numpy()
        ep["err25_copy"] = mse(torch.stack(c25), g[5:]).numpy()
    print(f"predictions done, {time.time() - t0:.0f} s", flush=True)

    groups = ["ind", "smaller", "larger"]
    res = {"run": str(run), "weights": weights.name, "device": str(dev), "history": HIST, "frameskip": FS,
           "action_mean": mean[0].tolist(), "action_std": std[0].tolist(), "n_episodes": {}, "embedding_error": {},
           "probe": {}, "dino_wm_issue6": DINO_WM_6}
    for gname in groups:
        G = [ep for ep in eps if ep["group"] == gname]
        res["n_episodes"][gname] = len(G)
        d = {}
        for key in ("err1", "err_roll", "err25"):
            m = np.array([ep[key].mean() for ep in G])
            c = np.array([ep[key + "_copy"].mean() for ep in G])
            d[key.replace("err", "").strip("_") or "1"] = {"model": boot(m), "copy_last": boot(c), "ratio": float(m.mean() / c.mean())}
        d["roll_curve"] = {"model": np.mean([ep["err_roll"] for ep in G], 0).tolist(),
                           "copy_last": np.mean([ep["err_roll_copy"] for ep in G], 0).tolist()}
        d["roll_last"] = {"model": boot([ep["err_roll"][-1] for ep in G]), "copy_last": boot([ep["err_roll_copy"][-1] for ep in G])}
        res["embedding_error"][gname] = d
    # real embedding spread, to read the errors against
    allg = torch.cat([ep["emb"][GRID] for ep in eps])
    res["embedding_var_per_dim"] = float(allg.var(0).mean())

    # probes, two folds
    for gname in groups:
        G = [ep for ep in eps if ep["group"] == gname]
        for i, ep in enumerate(sorted(G, key=lambda x: x["name"])):
            ep["fold"] = i % 2
    probes = {"ridge": Ridge, "mlp": MLP}
    store = {}
    for pname, P in probes.items():
        out = {g: {"real": [], "roll": [], "p25": [], "p1": [], "gt": [], "real_all": [], "gt_all": []} for g in groups}
        info = []
        for fold in (0, 1):
            fit = [ep for ep in eps if ep["fold"] != fold]
            X = np.concatenate([ep["emb"].numpy() for ep in fit])
            y = np.concatenate([ep["px"] for ep in fit]).astype(float)
            grp = np.concatenate([[k] * T for k, _ in enumerate(fit)])
            pr = P().fit(X, y, grp)
            info.append({"fold": fold, "n_fit_frames": len(y), "alpha": getattr(pr, "alpha", None), "best_epoch": getattr(pr, "best_epoch", None)})
            for ep in eps:
                if ep["fold"] != fold:
                    continue
                o = out[ep["group"]]
                o["real_all"].append(pr.predict(ep["emb"].numpy()))
                o["gt_all"].append(ep["px"].astype(float))
                o["real"].append(pr.predict(ep["emb"][GRID].numpy()))
                o["gt"].append(ep["px"][GRID].astype(float))
                o["roll"].append(pr.predict(ep["pred_roll"].numpy()))
                o["p25"].append(np.concatenate([np.full(5, np.nan), pr.predict(ep["pred25"].numpy())]))
                o["p1"].append(np.array([np.nan] * 3 + [pr.predict(ep["pred1_grid"][t][None].numpy())[0] for t in GRID[3:]]))
        r = {"folds": info}
        yall = np.concatenate([np.concatenate(out[g]["gt_all"]) for g in groups])
        pall = np.concatenate([np.concatenate(out[g]["real_all"]) for g in groups])
        r["r2_real_all_frames"] = r2(yall, pall)
        r["mae_real_all_frames"] = float(np.abs(yall - pall).mean())
        for g in groups:
            o = {k: np.array(v) for k, v in out[g].items()}
            store[f"{pname}_{g}"] = o
            r[g] = {"n": len(o["gt"]),
                    "r2_real_all_frames": r2(o["gt_all"].ravel(), o["real_all"].ravel()),
                    "mae_real_all_frames": float(np.abs(o["gt_all"] - o["real_all"]).mean()),
                    "last": {"real_px": boot(o["gt"][:, -1]), "probe_real": boot(o["real"][:, -1]),
                             "probe_rollout": boot(o["roll"][:, -1]), "probe_25step": boot(o["p25"][:, -1]),
                             "probe_1step": boot(o["p1"][:, -1])},
                    "curve": {"real_px": o["gt"].mean(0).tolist(), "probe_real": o["real"].mean(0).tolist(),
                              "probe_rollout": o["roll"].mean(0).tolist(), "probe_25step": np.nanmean(o["p25"], 0).tolist(),
                              "probe_1step": np.nanmean(o["p1"], 0).tolist()}}
        res["probe"][pname] = r
    res["seconds"] = round(time.time() - t0, 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out + ".json").write_text(json.dumps(res, indent=1))
    np.savez_compressed(a.out + ".npz", names=np.array([ep["name"] for ep in eps]), groups=np.array([ep["group"] for ep in eps]),
                        emb=np.stack([ep["emb"].numpy() for ep in eps]), pred_roll=np.stack([ep["pred_roll"].numpy() for ep in eps]),
                        pred25=np.stack([ep["pred25"].numpy() for ep in eps]), px=np.stack([ep["px"] for ep in eps]),
                        err1=np.stack([ep["err1"] for ep in eps]), err1_copy=np.stack([ep["err1_copy"] for ep in eps]),
                        err_roll=np.stack([ep["err_roll"] for ep in eps]), err_roll_copy=np.stack([ep["err_roll_copy"] for ep in eps]),
                        **{f"probe_{k}_{kk}": v for k, o in store.items() for kk, v in o.items()})
    for pname in probes:
        r = res["probe"][pname]
        print(f"[{pname}] R2 real frames {r['r2_real_all_frames']:.3f}")
        for g in groups:
            L = r[g]["last"]
            print(f"  {g:8s} last frame: real {L['real_px'][0]:.0f}  probe(real) {L['probe_real'][0]:.0f}  "
                  f"probe(rollout) {L['probe_rollout'][0]:.0f}  probe(25-step) {L['probe_25step'][0]:.0f}  | DINO-WM rollout {DINO_WM_6[g]['rollout']}")
    for g in groups:
        d = res["embedding_error"][g]
        print(f"  {g:8s} emb MSE 1-step {d['1']['model'][0]:.4f} (copy {d['1']['copy_last'][0]:.4f})  "
              f"rollout {d['roll']['model'][0]:.4f} (copy {d['roll']['copy_last'][0]:.4f})")
    print(f"done {res['seconds']} s -> {a.out}.json", flush=True)


if __name__ == "__main__":
    main()
