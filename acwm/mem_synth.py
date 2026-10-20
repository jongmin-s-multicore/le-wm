"""Peak GPU memory of one LeWM training step for several batch sizes, with a synthetic batch of the Cloth Move shape
(4 frames of 3x224x224, action blocks of 5 x 6 = 30). Model = config/train/model/lewm.yaml (ViT-tiny patch 14,
predictor 6 layers, projectors 2048), bf16 autocast, AdamW, grad clip 1.0, as in train.py. Memory depends only on
the shapes, so random values are enough. A straight line through the small batches gives the value at batch 128.

usage: python acwm/mem_synth.py --batches 8,16,32 [--cap-gib 6] [-o out.json]
"""
import argparse, json, sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jepa import JEPA  # noqa: E402
from module import ARPredictor, Embedder, MLP, SIGReg  # noqa: E402


def vit_tiny():
    try:
        import stable_pretraining as spt
        return spt.backbone.utils.vit_hf("tiny", patch_size=14, image_size=224, pretrained=False, use_mask_token=False), "stable_pretraining.vit_hf"
    except ImportError:
        from transformers import ViTConfig, ViTModel
        cfg = ViTConfig(hidden_size=192, num_hidden_layers=12, num_attention_heads=3, intermediate_size=768,
                        image_size=224, patch_size=14)
        m = ViTModel(cfg, add_pooling_layer=False, use_mask_token=False)
        m.config.interpolate_pos_encoding = True
        return m, "transformers.ViTModel (same config as vit_hf tiny)"


def build():
    enc, how = vit_tiny()
    mlp = lambda: MLP(input_dim=192, output_dim=192, hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d)
    model = JEPA(encoder=enc,
                 predictor=ARPredictor(num_frames=3, input_dim=192, hidden_dim=192, output_dim=192, depth=6, heads=16,
                                       mlp_dim=2048, dim_head=64, dropout=0.1, emb_dropout=0.0),
                 action_encoder=Embedder(input_dim=30, emb_dim=192), projector=mlp(), pred_proj=mlp())
    return model, how


def measure(B, dev, steps=4):
    torch.manual_seed(0)
    model, how = build()
    model = model.to(dev).train()
    sigreg = SIGReg(knots=17, num_proj=1024).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-3)
    pix = torch.randn(B, 4, 3, 224, 224, device=dev)
    act = torch.randn(B, 4, 30, device=dev)
    torch.cuda.synchronize(dev)
    torch.cuda.reset_peak_memory_stats(dev)
    t0 = time.time()
    for _ in range(steps):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.encode({"pixels": pix, "action": act})
            emb, ae = out["emb"], out["act_emb"]
            pred = model.predict(emb[:, :3], ae[:, :3])
            loss = (pred - emb[:, 1:]).pow(2).mean() + 0.09 * sigreg(emb.transpose(0, 1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    torch.cuda.synchronize(dev)
    r = {"batch": B, "peak_alloc_GiB": torch.cuda.max_memory_allocated(dev) / 2**30,
         "peak_reserved_GiB": torch.cuda.max_memory_reserved(dev) / 2**30, "s_per_step": (time.time() - t0) / steps,
         "params_M": sum(p.numel() for p in model.parameters()) / 1e6, "encoder": how,
         "attn": getattr(model.encoder.config, "_attn_implementation", None)}
    del model, opt, pix, act, out, emb, ae, pred, loss
    torch.cuda.empty_cache()
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="8,16,32")
    ap.add_argument("--cap-gib", type=float, default=0)
    ap.add_argument("-o", "--out", default=None)
    a = ap.parse_args()
    dev = torch.device("cuda", 0)
    if a.cap_gib:
        torch.cuda.set_per_process_memory_fraction(a.cap_gib * 2**30 / torch.cuda.get_device_properties(dev).total_memory, dev)
    rows = [measure(int(b), dev) for b in a.batches.split(",")]
    for r in rows:
        print(json.dumps(r), flush=True)
    import numpy as np
    b = np.array([r["batch"] for r in rows], float)
    res = {"rows": rows, "gpu": torch.cuda.get_device_name(dev), "torch": torch.__version__}
    for k in ("peak_alloc_GiB", "peak_reserved_GiB"):
        y = np.array([r[k] for r in rows])
        slope, icpt = np.polyfit(b, y, 1)
        res[k + "_fit"] = {"GiB_per_sample": float(slope), "intercept_GiB": float(icpt), "at_128": float(icpt + 128 * slope)}
    try:
        import transformers
        res["transformers"] = transformers.__version__
    except ImportError:
        pass
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}), flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
