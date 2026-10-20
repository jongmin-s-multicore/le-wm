"""LeWM on ACWM-Phys: embedding prediction error and a probe of the physical quantity of each environment
(measures.py). There is no simulator, so there is no planning. Test sets: ind_test and ood_test, split into the
groups of issue jongmin-s-multicore/thread0#6 (Cloth Move ind / smaller / larger, Push Sand ind / ood, Pour Water
ind / low / high, Stack Cube ind / ood with the failed episodes as the reported subsets).

Frames are on the 5-frame grid 0, 5, ..., up to k_full (measures.py); action block k = actions 5k..5k+4.
(a) Embedding error, MSE over the 192 dims, against copy-last (the last real embedding the prediction starts from):
    1-step   real frames s, s+5, s+10 and their action blocks -> frame s+15, for every training window s
    rollout  real frame 0 only, then the model's own predictions with all action blocks (as #6 for DINO-WM)
    25-step  every grid frame t >= 25: real grid frames up to t-25 (the last 3), then 5 predicted steps
(b) Probe from embedding to the quantity of a frame. Two folds: half of the episodes of every group (and, for
    Stack Cube, of success / fail) fit the probe on real embeddings of all their frames, the other half is
    measured; then the halves swap. Ridge regression (alpha by episode-grouped CV) and an MLP; Stack Cube uses
    logistic regression and an MLP classifier for "stacked". The probe is applied to real embeddings (does the
    encoder keep the quantity?) and to predicted ones (does the predictor keep it?).

usage (cwd = le-wm checkout):
  python acwm/eval_acwm.py <env> <run dir with weights_*.pt [config.json]> <h5 dir> <out prefix>
                           [--weights FILE] [--src ACWM-Phys dir] [--device cpu|cuda]
  -> <out prefix>.json, <out prefix>.npz
"""
import argparse, json, sys, time
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from measures import FS, KIND, MEASURE, NAME, k_full, summary_frames  # noqa: E402
from eval_clothmove import MLP, Ridge, boot, r2  # noqa: E402

HIST = 3
PREFIX = {"cloth_move": "clothmove", "push_sand": "pushsand", "stack_cube": "stackcube", "pour_water": "pourwater"}
GROUPS = {"cloth_move": ["ind", "smaller", "larger"], "push_sand": ["ind", "ood"], "pour_water": ["ind", "low", "high"],
          "stack_cube": ["ind", "ood"]}
# issue #6, table 2 (DINO-WM, InD training): last frame (Pour Water: mean of frames >= 250); Stack Cube: share of the
# failed episodes whose last frame shows the yellow cube on the green one
DINO_WM_6 = {
    "cloth_move": {"ind": dict(n=100, real=1819, enc_dec=1790, step25=1801, rollout=1888),
                   "smaller": dict(n=50, real=404, enc_dec=388, step25=840, rollout=1129),
                   "larger": dict(n=50, real=4475, enc_dec=4352, step25=2773, rollout=1836)},
    "push_sand": {"ind": dict(n=100, real=1792, enc_dec=1729, step25=1703, rollout=1665),
                  "ood": dict(n=100, real=3886, enc_dec=3215, step25=2405, rollout=1843)},
    "pour_water": {"ind": dict(n=50, real=1970, enc_dec=1962, step25=1984, rollout=2045),
                   "low": dict(n=26, real=1384, enc_dec=1370, step25=1593, rollout=2049),
                   "high": dict(n=24, real=2407, enc_dec=2326, step25=2162, rollout=2077)},
    "stack_cube": {"ind_fail": dict(n=17, real=0.12, enc_dec=0.18, step25=0.29, rollout=0.24),
                   "ood_fail": dict(n=71, real=0.08, enc_dec=0.06, step25=0.14, rollout=0.23)},
}


def model_config(run, action_dim):
    p = run / "config.json"
    if p.exists():
        return json.loads(p.read_text())
    # the le-wm model config resolved as train.py saves it (config.json is uploaded only at the end of a run)
    from omegaconf import OmegaConf
    root = Path(__file__).resolve().parents[1] / "config" / "train"
    cfg = OmegaConf.merge(OmegaConf.load(root / "lewm.yaml"), {"model": OmegaConf.load(root / "model" / "lewm.yaml")})
    cfg.model.action_encoder.input_dim = FS * action_dim
    return OmegaConf.to_container(OmegaConf.create({k: cfg[k] for k in ("img_size", "history_size", "embed_dim", "model")}), resolve=True)["model"]


def load_model(run, weights, action_dim, dev):
    import hydra
    from omegaconf import OmegaConf
    model = hydra.utils.instantiate(OmegaConf.create(model_config(run, action_dim)))
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    return model.to(dev).eval()


def train_action_stats(env, h5dir, src):
    """the z-score of train.py: mean / std of all non-NaN action rows of ind_train"""
    side = h5dir / f"{PREFIX[env]}_ind_train.json"
    if side.exists():
        d = json.loads(side.read_text())
        if "action_mean" in d:
            return torch.tensor([d["action_mean"]]), torch.tensor([d["action_std"]]), "ind_train sidecar"
    h5 = h5dir / f"{PREFIX[env]}_ind_train.h5"
    if h5.exists():
        with h5py.File(h5, "r") as f:
            a = torch.from_numpy(np.array(f["action"][:]))
        how = "ind_train h5"
    else:
        from convert_acwm import load_split
        _, _, acts, _ = load_split(Path(src), env, "ind_train")
        a = torch.from_numpy(np.concatenate(acts))
        how = "ind_train metadata"
    a = a[~torch.isnan(a).any(dim=1)]
    return a.mean(0, keepdim=True), a.std(0, keepdim=True), how


def load_split_eps(env, h5dir, split, mean, std):
    side = json.loads((h5dir / f"{PREFIX[env]}_{split}.json").read_text())
    eps = []
    with h5py.File(h5dir / f"{PREFIX[env]}_{split}.h5", "r") as f:
        off, ln = f["ep_offset"][:], f["ep_len"][:]
        for e in range(len(off)):
            raw = f["pixels"][off[e] : off[e] + ln[e]]
            a = torch.from_numpy(f["action"][off[e] : off[e] + ln[e]])
            n_act = int((~torch.isnan(a).any(1)).sum())
            ep = {"name": side["names"][e], "group": side["groups"][e], "split": split, "raw": raw, "n_rows": int(ln[e]),
                  "n_act": n_act, "q": np.array([MEASURE[env](x) for x in raw], float),
                  "action": torch.nan_to_num(((a - mean) / std).float(), 0.0)}
            if env == "stack_cube":
                ep["success"] = bool(side["extra"]["success"][e])
            eps.append(ep)
    return eps


_PREP = None


@torch.no_grad()
def encode(model, raw, dev, bs=256):
    global _PREP
    if _PREP is None:
        from utils import get_img_preprocessor
        _PREP = get_img_preprocessor(source="pixels", target="pixels", img_size=224)
    out = []
    for i in range(0, len(raw), bs):
        pix = _PREP({"pixels": torch.from_numpy(raw[i : i + bs]).permute(0, 3, 1, 2)})["pixels"]
        out.append(model.encode({"pixels": pix.unsqueeze(0).to(dev)})["emb"][0].float().cpu())
    return torch.cat(out)


def blocks(act, starts):
    return torch.stack([act[s : s + FS].reshape(-1) for s in starts])


@torch.no_grad()
def rollout(model, ctx, acts, n, dev):
    """ctx (B, h, D) real embeddings with action blocks acts (B, h + n - 1, A); n predicted steps, history HIST"""
    emb, h, preds = ctx.to(dev), ctx.shape[1], []
    for k in range(n):
        lo = max(0, emb.shape[1] - HIST)
        p = model.predict(emb[:, lo:], model.action_encoder(acts[:, lo : h + k].to(dev)))[:, -1:]
        emb = torch.cat([emb, p], 1)
        preds.append(p[:, 0].float().cpu())
    return torch.stack(preds, 1)


def mse(a, b):
    return ((a - b) ** 2).mean(-1)


class Logistic:
    """L2 logistic regression on standardized features, C by episode-grouped CV (sklearn)"""

    def __init__(self, Cs=(0.01, 0.1, 1, 10)):
        self.Cs = Cs

    def fit(self, X, y, groups, k=5):
        from sklearn.linear_model import LogisticRegression
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-8
        Z = (X - self.mu) / self.sd
        ug = np.unique(groups)
        fold = {g: i % k for i, g in enumerate(np.random.RandomState(0).permutation(ug))}
        f = np.array([fold[g] for g in groups])
        if y.min() == y.max():
            self.const = float(y[0]); return self
        self.const = None
        best = (-1, None)
        for C in self.Cs:
            acc = []
            for i in range(k):
                tr, va = f != i, f == i
                if len(np.unique(y[tr])) < 2:
                    continue
                m = LogisticRegression(C=C, max_iter=2000).fit(Z[tr], y[tr])
                acc.append(((m.predict_proba(Z[va])[:, 1] > 0.5) == y[va]).mean())
            if acc and np.mean(acc) > best[0]:
                best = (np.mean(acc), C)
        self.alpha = best[1]
        self.m = LogisticRegression(C=best[1], max_iter=2000).fit(Z, y)
        return self

    def predict(self, X):
        if self.const is not None:
            return np.full(len(X), self.const)
        return self.m.predict_proba((X - self.mu) / self.sd)[:, 1]


class MLPClass(MLP):
    """the MLP probe of eval_clothmove.py with a logistic output (probability of class 1)"""

    def fit(self, X, y, groups):
        torch.manual_seed(self.seed)
        ug = np.unique(groups)
        val_g = set(np.random.RandomState(self.seed).permutation(ug)[: max(1, len(ug) // 10)])
        v = np.array([g in val_g for g in groups])
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-8
        Xt = torch.tensor((X - self.mu) / self.sd, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)
        net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], self.h), torch.nn.ReLU(), torch.nn.Linear(self.h, self.h),
                                  torch.nn.ReLU(), torch.nn.Linear(self.h, 1))
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.wd)
        tr, va = torch.tensor(~v), torch.tensor(v)
        lossf = torch.nn.BCEWithLogitsLoss()
        best, best_sd = 1e18, None
        for ep in range(self.epochs):
            perm = torch.nonzero(tr)[:, 0][torch.randperm(int(tr.sum()))]
            for i in range(0, len(perm), 256):
                j = perm[i : i + 256]
                loss = lossf(net(Xt[j])[:, 0], yt[j])
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                vl = float(lossf(net(Xt[va])[:, 0], yt[va]))
            if vl < best:
                best, best_sd, self.best_epoch = vl, {k: t.clone() for k, t in net.state_dict().items()}, ep
        net.load_state_dict(best_sd)
        self.net = net.eval()
        return self

    def predict(self, X):
        with torch.no_grad():
            return torch.sigmoid(self.net(torch.tensor((X - self.mu) / self.sd, dtype=torch.float32))[:, 0]).numpy()


def report_sets(env, eps):
    """name -> episode list for the summary table"""
    if env == "stack_cube":
        return {"ind_fail": [e for e in eps if e["split"] == "ind_test" and not e["success"]],
                "ood_fail": [e for e in eps if e["split"] == "ood_test" and not e["success"]],
                "ind": [e for e in eps if e["split"] == "ind_test"], "ood": [e for e in eps if e["split"] == "ood_test"]}
    return {g: [e for e in eps if e["group"] == g] for g in GROUPS[env]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("env", choices=list(PREFIX)); ap.add_argument("run"); ap.add_argument("h5dir"); ap.add_argument("out")
    ap.add_argument("--weights", default=None, help="default: the weights_epoch_*.pt with the highest epoch")
    ap.add_argument("--src", default="/data/dinowm/downloads/ACWM-Phys")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    env, run, h5dir, dev = a.env, Path(a.run), Path(a.h5dir), torch.device(a.device)
    weights = Path(a.weights) if a.weights else max(run.glob("weights_epoch_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
    t0 = time.time()
    mean, std, stats_from = train_action_stats(env, h5dir, a.src)
    model = load_model(run, weights, mean.shape[1], dev)
    eps = load_split_eps(env, h5dir, "ind_test", mean, std) + load_split_eps(env, h5dir, "ood_test", mean, std)
    print(f"{env}: {len(eps)} episodes, action stats from {stats_from}, {time.time() - t0:.0f} s", flush=True)

    for ep in eps:
        e, act, n = encode(model, ep.pop("raw"), dev), ep["action"], ep["n_rows"]
        ep["emb"] = e
        kf = k_full(n, ep["n_act"])
        grid = list(range(0, FS * kf, FS))
        ep["grid"] = grid
        starts = list(range(0, n - 4 * FS + 1))
        ctx = torch.stack([e[[s, s + FS, s + 2 * FS]] for s in starts])
        ab = torch.stack([blocks(act, [s, s + FS, s + 2 * FS]) for s in starts])
        with torch.no_grad():
            p1 = model.predict(ctx.to(dev), model.action_encoder(ab.to(dev)))[:, -1].float().cpu()
        tgt = e[[s + 3 * FS for s in starts]]
        ep["err1"], ep["err1_copy"] = mse(p1, tgt).numpy(), mse(e[[s + 2 * FS for s in starts]], tgt).numpy()
        AB = blocks(act, grid[:-1])
        g = e[grid]
        pr = rollout(model, g[:1].unsqueeze(0), AB.unsqueeze(0), kf - 1, dev)[0]
        ep["pred_roll"] = torch.cat([g[:1], pr])
        ep["err_roll"], ep["err_roll_copy"] = mse(pr, g[1:]).numpy(), mse(g[:1].expand_as(g[1:]), g[1:]).numpy()
        p25 = torch.full_like(g, float("nan"))
        c25 = []
        for j in range(5, kf):
            i = j - 5
            lo = max(0, i - HIST + 1)
            p25[j] = rollout(model, g[lo : i + 1].unsqueeze(0), AB[lo:j].unsqueeze(0), 5, dev)[0, -1]
            c25.append(mse(g[i], g[j]).item())
        ep["pred25"] = p25
        ep["err25"] = mse(p25[5:], g[5:]).numpy()
        ep["err25_copy"] = np.array(c25)
    print(f"predictions done, {time.time() - t0:.0f} s", flush=True)

    res = {"env": env, "quantity": NAME[env], "run": str(run), "weights": weights.name, "device": str(dev), "history": HIST,
           "frameskip": FS, "action_stats_from": stats_from, "action_mean": mean[0].tolist(), "action_std": std[0].tolist(),
           "embedding_var_per_dim": float(torch.cat([ep["emb"][ep["grid"]] for ep in eps]).var(0).mean()),
           "embedding_error": {}, "probe": {}, "dino_wm_issue6": DINO_WM_6[env]}
    sets = report_sets(env, eps)
    for sname, G in sets.items():
        if not G:
            continue
        d = {"n": len(G)}
        for key in ("err1", "err_roll", "err25"):
            m = np.array([ep[key].mean() for ep in G if len(ep[key])])
            c = np.array([ep[key + "_copy"].mean() for ep in G if len(ep[key + "_copy"])])
            d[{"err1": "1step", "err_roll": "rollout", "err25": "25step"}[key]] = {"model": boot(m), "copy_last": boot(c), "ratio": float(m.mean() / c.mean())}
        d["rollout_last"] = {"model": boot([ep["err_roll"][-1] for ep in G]), "copy_last": boot([ep["err_roll_copy"][-1] for ep in G])}
        res["embedding_error"][sname] = d

    # folds: alternate within (group, success) strata, sorted by name
    strata = {}
    for ep in eps:
        strata.setdefault((ep["group"], ep.get("success")), []).append(ep)
    for S in strata.values():
        for i, ep in enumerate(sorted(S, key=lambda x: x["name"])):
            ep["fold"] = i % 2
    binary = KIND[env] == "binary"
    probes = {"logistic": Logistic, "mlp": MLPClass} if binary else {"ridge": Ridge, "mlp": MLP}
    store = {}
    for pname, P in probes.items():
        info = []
        for fold in (0, 1):
            fit = [ep for ep in eps if ep["fold"] != fold]
            X = np.concatenate([ep["emb"].numpy() for ep in fit])
            y = np.concatenate([ep["q"] for ep in fit])
            grp = np.concatenate([[k] * ep["n_rows"] for k, ep in enumerate(fit)])
            pr = P().fit(X, y, grp)
            info.append({"fold": fold, "n_fit_frames": int(len(y)), "alpha_or_C": getattr(pr, "alpha", None), "best_epoch": getattr(pr, "best_epoch", None)})
            for ep in eps:
                if ep["fold"] != fold:
                    continue
                ep[f"{pname}_all"] = pr.predict(ep["emb"].numpy())
                ep[f"{pname}_roll"] = pr.predict(ep["pred_roll"].numpy())
                p25 = np.full(len(ep["grid"]), np.nan)
                ok = torch.isfinite(ep["pred25"]).all(1).numpy()
                if ok.any():
                    p25[ok] = pr.predict(ep["pred25"][ok].numpy())
                ep[f"{pname}_25"] = p25
        y_all = np.concatenate([ep["q"] for ep in eps])
        p_all = np.concatenate([ep[f"{pname}_all"] for ep in eps])
        r = {"folds": info}
        if binary:
            r["accuracy_real_all_frames"] = float(((p_all > 0.5) == (y_all > 0.5)).mean())
            r["balanced_accuracy_real_all_frames"] = float(np.mean([((p_all > 0.5) == (y_all > 0.5))[y_all == c].mean() for c in (0, 1) if (y_all == c).any()]))
            r["positive_share_real_frames"] = float(y_all.mean())
        else:
            r["r2_real_all_frames"] = r2(y_all, p_all)
            r["mae_real_all_frames"] = float(np.abs(y_all - p_all).mean())
        for sname, G in sets.items():
            if not G:
                continue

            def val(ep, arr):  # the reported value of one episode: summary frames of the grid
                v = np.asarray(arr, float)[summary_frames(env, ep["grid"])]
                v = v[np.isfinite(v)]
                if not len(v):
                    return np.nan
                return float((v > 0.5).mean()) if binary else float(v.mean())

            row = {"n": len(G),
                   "real": boot([val(ep, ep["q"][ep["grid"]]) for ep in G]),
                   "probe_real": boot([val(ep, ep[f"{pname}_all"][ep["grid"]]) for ep in G]),
                   "probe_rollout": boot([val(ep, ep[f"{pname}_roll"]) for ep in G]),
                   "probe_25step": boot([val(ep, ep[f"{pname}_25"]) for ep in G])}
            yy = np.concatenate([ep["q"] for ep in G]); pp = np.concatenate([ep[f"{pname}_all"] for ep in G])
            if binary:
                row["accuracy_real_all_frames"] = float(((pp > 0.5) == (yy > 0.5)).mean())
            else:
                row["r2_real_all_frames"] = r2(yy, pp) if yy.std() > 0 else None
                row["mae_real_all_frames"] = float(np.abs(yy - pp).mean())
            L = min(len(ep["grid"]) for ep in G)
            row["curve"] = {"real": np.mean([ep["q"][ep["grid"]][:L] for ep in G], 0).tolist(),
                            "probe_real": np.mean([ep[f"{pname}_all"][ep["grid"]][:L] for ep in G], 0).tolist(),
                            "probe_rollout": np.mean([ep[f"{pname}_roll"][:L] for ep in G], 0).tolist(),
                            "probe_25step": np.nanmean([ep[f"{pname}_25"][:L] for ep in G], 0).tolist()}
            r[sname] = row
        res["probe"][pname] = r
        store[pname] = {k: [ep[f"{pname}_{k}"] for ep in eps] for k in ("all", "roll", "25")}
    res["seconds"] = round(time.time() - t0, 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out + ".json").write_text(json.dumps(res, indent=1))
    obj = lambda xs: np.array(xs, dtype=object)
    np.savez_compressed(a.out + ".npz", names=np.array([ep["name"] for ep in eps]), groups=np.array([ep["group"] for ep in eps]),
                        split=np.array([ep["split"] for ep in eps]), success=np.array([ep.get("success", True) for ep in eps]),
                        emb=obj([ep["emb"].numpy() for ep in eps]), pred_roll=obj([ep["pred_roll"].numpy() for ep in eps]),
                        pred25=obj([ep["pred25"].numpy() for ep in eps]), q=obj([ep["q"] for ep in eps]),
                        grid=obj([np.array(ep["grid"]) for ep in eps]),
                        **{f"probe_{p}_{k}": obj(v) for p, d in store.items() for k, v in d.items()})
    for pname in probes:
        r = res["probe"][pname]
        head = f"acc {r['accuracy_real_all_frames']:.3f}" if binary else f"R2 {r['r2_real_all_frames']:.3f}"
        print(f"[{pname}] real frames {head}")
        for sname in sets:
            if sname not in r:
                continue
            L, D = r[sname], DINO_WM_6[env].get(sname, {})
            fm = (lambda v: f"{v[0]:.2f}") if binary else (lambda v: f"{v[0]:.0f}")
            print(f"  {sname:9s} real {fm(L['real'])}  probe(real) {fm(L['probe_real'])}  probe(rollout) {fm(L['probe_rollout'])}  "
                  f"probe(25-step) {fm(L['probe_25step'])}  | DINO-WM #6 rollout {D.get('rollout', '-')}")
    for sname, d in res["embedding_error"].items():
        print(f"  {sname:9s} emb MSE 1-step {d['1step']['model'][0]:.4f} (copy {d['1step']['copy_last'][0]:.4f})  "
              f"rollout {d['rollout']['model'][0]:.4f} (copy {d['rollout']['copy_last'][0]:.4f})")
    print(f"done {res['seconds']} s -> {a.out}.json", flush=True)


if __name__ == "__main__":
    main()
