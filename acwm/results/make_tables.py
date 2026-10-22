"""Tables of the LeWM ACWM-Phys runs (markdown) from the eval JSONs in this folder.

  eval_<env>.json            acwm/eval_acwm.py (LeWM)
  dinowm_cloth_probe.json    acwm/probe_dinowm.py (the same probe on DINO-WM latents, Cloth Move)

usage: python3 make_tables.py [folder]  -> prints the tables, writes tables.md
"""
import json, sys
from pathlib import Path

D = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent)
ENVS = ["cloth_move", "push_sand", "stack_cube", "pour_water"]
TITLE = {"cloth_move": "Cloth Move", "push_sand": "Push Sand", "stack_cube": "Stack Cube", "pour_water": "Pour Water"}
ROWS = {"cloth_move": [("ind", "InD"), ("smaller", "OoD, 0.26x area"), ("larger", "OoD, 2.5x area")],
        "push_sand": [("ind", "InD"), ("ood", "OoD, 2x sand")],
        "pour_water": [("ind", "InD"), ("low", "OoD, less water"), ("high", "OoD, more water")],
        "stack_cube": [("ind_fail", "InD, failed"), ("ood_fail", "OoD, failed")]}


def f(v, binary):
    if v is None or v[0] is None:
        return "-"
    return f"{100 * v[0]:.0f}%" if binary else f"{v[0]:,.0f}"


def pct(x, ref, binary):
    if binary or x is None or ref in (None, 0):
        return ""
    return f" ({100 * x / ref:.0f}%)"


def main():
    out = []
    dprobe = json.loads((D / "dinowm_cloth_probe.json").read_text()) if (D / "dinowm_cloth_probe.json").exists() else None
    for env in ENVS:
        p = D / f"eval_{env}.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        binary = env == "stack_cube"
        pr = r["probe"]["logistic" if binary else "ridge"]
        six = r["dino_wm_issue6"]
        q = r["quantity"]
        what = "last frame" if env not in ("pour_water", "stack_cube") else ("mean of frames 250-295" if env == "pour_water" else "share of failed episodes stacked in the last frame")
        out.append(f"### {TITLE[env]}: {q}, {what}\n")
        hdr = "| test | n | real | LeWM probe, real frames | LeWM probe, 25-step | LeWM probe, rollout | DINO-WM Enc->Dec (#6) | DINO-WM 25-step (#6) | DINO-WM rollout (#6) |"
        if env == "cloth_move" and dprobe:
            hdr += " DINO-WM probe, real | DINO-WM probe, rollout |"
        out.append(hdr)
        out.append("|" + "---|" * (hdr.count("|") - 1))
        for key, label in ROWS[env]:
            if key not in pr:
                continue
            L, S = pr[key], six.get(key, {})
            real = L["real"][0]
            cells = [label, str(L["n"]), f(L["real"], binary),
                     f(L["probe_real"], binary) + pct(L["probe_real"][0], real, binary),
                     f(L["probe_25step"], binary) + pct(L["probe_25step"][0], real, binary),
                     f(L["probe_rollout"], binary) + pct(L["probe_rollout"][0], real, binary)]
            for k in ("enc_dec", "step25", "rollout"):
                v = S.get(k)
                cells.append("-" if v is None else (f"{100 * v:.0f}%" if binary else f"{v:,}" + pct(v, S.get("real"), binary)))
            if env == "cloth_move" and dprobe:
                DL = dprobe["probe"]["ridge"][key]["last"]
                cells += [f(DL["probe_real"], False) + pct(DL["probe_real"][0], real, False), f(DL["probe_rollout"], False) + pct(DL["probe_rollout"][0], real, False)]
            out.append("| " + " | ".join(cells) + " |")
        acc = f"accuracy {pr['accuracy_real_all_frames']:.3f}" if binary else f"R2 {pr['r2_real_all_frames']:.3f}, MAE {pr['mae_real_all_frames']:.0f} px"
        out.append(f"\nProbe on held-out real frames (all frames, all groups): {acc}.\n")
        out.append("| test | n | 1-step MSE | copy-last | ratio | rollout MSE | copy-last | ratio | rollout last step | copy-last |")
        out.append("|---|---|---|---|---|---|---|---|---|---|")
        for key, label in ROWS[env] if env != "stack_cube" else [("ind", "InD"), ("ood", "OoD")]:
            e = r["embedding_error"].get(key)
            if not e:
                continue
            out.append(f"| {label} | {e['n']} | {e['1step']['model'][0]:.3f} | {e['1step']['copy_last'][0]:.3f} | {e['1step']['ratio']:.2f} | "
                       f"{e['rollout']['model'][0]:.3f} | {e['rollout']['copy_last'][0]:.3f} | {e['rollout']['ratio']:.2f} | "
                       f"{e['rollout_last']['model'][0]:.3f} | {e['rollout_last']['copy_last'][0]:.3f} |")
        out.append(f"\nEmbedding variance per dimension (real grid frames): {r['embedding_var_per_dim']:.3f}. Weights: {r['weights']}.\n")
    text = "\n".join(out)
    print(text)
    (D / "tables.md").write_text(text + "\n")


if __name__ == "__main__":
    main()
