"""Speed and GPU memory of LeWM training on ACWM-Phys Cloth Move (ROOT/AGENTS.md 3.4).

  gpu    one batch kept on the GPU; forward + backward + AdamW step repeated, bf16 autocast and grad clip 1.0
         like train.py (no data loading). Prints step/s and the torch peak memory.
  data   the training DataLoader alone (train.py settings), batch/s.

The model, data and transforms are built from the Hydra config as in train.py.

usage (cwd = le-wm checkout, STABLEWM_HOME set):
  python acwm/speed.py gpu  --batch 128 [--steps 60] [--mem-cap-gib 0] [-o out.json] [overrides ...]
  python acwm/speed.py data --batch 128 [--batches 150] [--workers 6] [-o out.json] [overrides ...]
"""
import argparse, json, os, sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build(overrides, batch, workers=None):
    import hydra
    import stable_pretraining as spt
    import stable_worldmodel as swm
    from omegaconf import OmegaConf, open_dict
    from utils import get_column_normalizer, get_img_preprocessor

    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    with hydra.initialize(version_base=None, config_path="../config/train"):
        cfg = hydra.compose("lewm", overrides=["data=acwm_cloth_move", f"loader.batch_size={batch}"] + list(overrides))
    if workers is not None:
        cfg.loader.num_workers = workers
        if workers == 0:
            cfg.loader.persistent_workers, cfg.loader.prefetch_factor = False, None
    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    name = dataset_cfg.pop("name")
    dataset = swm.data.load_dataset(name, transform=None, cache_dir=os.environ.get("LOCAL_DATASET_DIR"), **dataset_cfg)
    transforms = [get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)]
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if not col.startswith("pixels"):
                transforms.append(get_column_normalizer(dataset, col, col))
        cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")
    dataset.transform = spt.data.transforms.Compose(*transforms)
    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, _ = spt.data.random_split(dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen)
    loader = torch.utils.data.DataLoader(train_set, **cfg.loader, shuffle=True, drop_last=True, generator=rnd_gen)
    return cfg, dataset, train_set, loader


def lewm_loss(model, sigreg, batch, cfg):
    """the loss of train.py lejepa_forward"""
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    out = model.encode(batch)
    emb, act_emb = out["emb"], out["act_emb"]
    pred = model.predict(emb[:, : cfg.history_size], act_emb[:, : cfg.history_size])
    pred_loss = (pred - emb[:, cfg.num_preds :]).pow(2).mean()
    return pred_loss + cfg.loss.sigreg.weight * sigreg(emb.transpose(0, 1))


def run_gpu(a, overrides):
    import hydra
    from module import SIGReg

    cfg, dataset, train_set, loader = build(overrides, a.batch, workers=0)
    dev = torch.device("cuda")
    if a.mem_cap_gib:
        total = torch.cuda.get_device_properties(dev).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, a.mem_cap_gib * 2**30 / total), dev)
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(len(train_set), generator=g)[: a.batch].tolist()
    items = [train_set[i] for i in idx]
    batch = {k: torch.stack([it[k] for it in items]).to(dev) for k in ("pixels", "action")}
    torch.manual_seed(cfg.seed)
    model = hydra.utils.instantiate(cfg.model).to(dev)
    sigreg = SIGReg(**cfg.loss.sigreg.kwargs).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)
    model.train()
    torch.cuda.reset_peak_memory_stats(dev)

    def step():
        b = {k: v.clone() for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = lewm_loss(model, sigreg, b, cfg)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.trainer.gradient_clip_val)
        opt.step()
        return loss

    for _ in range(10):
        loss = step()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(a.steps):
        loss = step()
    torch.cuda.synchronize()
    dt = time.time() - t0
    res = {"mode": "gpu", "batch": a.batch, "steps": a.steps, "step_per_s": a.steps / dt, "loss": float(loss),
           "torch_peak_alloc_GiB": torch.cuda.max_memory_allocated(dev) / 2**30,
           "torch_peak_reserved_GiB": torch.cuda.max_memory_reserved(dev) / 2**30,
           "gpu": torch.cuda.get_device_name(dev), "torch": torch.__version__,
           "params_M": sum(p.numel() for p in model.parameters()) / 1e6}
    return res


def run_data(a, overrides):
    cfg, dataset, train_set, loader = build(overrides, a.batch, workers=a.workers)
    it = iter(loader)
    for _ in range(5):
        next(it)
    t0 = time.time()
    n = 0
    for _ in range(a.batches):
        b = next(it)
        n += 1
    dt = time.time() - t0
    return {"mode": "data", "batch": a.batch, "batches": n, "batch_per_s": n / dt, "num_workers": cfg.loader.num_workers,
            "samples": len(dataset), "train_samples": len(train_set), "pixels": list(b["pixels"].shape),
            "action": list(b["action"].shape), "cpus": os.cpu_count()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["gpu", "data"])
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--batches", type=int, default=150)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--mem-cap-gib", type=float, default=0)
    ap.add_argument("-o", "--out", default=None)
    a, overrides = ap.parse_known_args()
    res = run_gpu(a, overrides) if a.mode == "gpu" else run_data(a, overrides)
    res["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(json.dumps(res), flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
