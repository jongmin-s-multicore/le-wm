"""Cloth Move: visible cloth pixels over the episode, real frames vs LeWM (probe) vs DINO-WM (decoder), per test group.

  eval_cloth_move.json        LeWM probe curves (acwm/eval_acwm.py, ridge probe)
  dinowm_cloth_latents.npz    DINO-WM decoded rollout and Enc->Dec pixels per frame (acwm/dinowm_latents.py)

usage: python plot_curves.py [folder]   -> cloth_curves.png
"""
import json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True,
                     "axes.unicode_minus": False, "font.size": 11})

D = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent)
BLUE, ORANGE, INK, MUTED, SURFACE = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#fcfcfb"
GROUPS = [("ind", "InD"), ("smaller", "OoD, area 0.26x"), ("larger", "OoD, area 2.5x")]


def main():
    ev = json.loads((D / "eval_cloth_move.json").read_text())
    pr = ev["probe"]["ridge"]
    dw = np.load(D / "dinowm_cloth_latents.npz")
    groups = np.array([str(g) for g in dw["groups"]])
    frames = np.arange(0, 61, 5)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), facecolor=SURFACE)
    for ax, (g, title) in zip(axes, GROUPS):
        c = pr[g]["curve"]
        m = groups == g
        ax.set_facecolor(SURFACE)
        ax.plot(frames, c["real"], color=INK, lw=2, label="real")
        ax.plot(frames, c["probe_real"], color=BLUE, lw=2, ls=(0, (4, 3)), label="LeWM probe, real")
        ax.plot(frames[1:], c["probe_rollout"][1:], color=BLUE, lw=2, label="LeWM probe, rollout")
        ax.plot(frames, dw["px_dec_real"][m].mean(0), color=ORANGE, lw=2, ls=(0, (4, 3)), label="DINO-WM decoder, real")
        ax.plot(frames[1:], dw["px_dec_roll"][m][:, 1:].mean(0), color=ORANGE, lw=2, label="DINO-WM decoder, rollout")
        ax.set_title(f"{title} (n={int(m.sum())})", fontsize=12.5, color=INK, loc="left", pad=12)
        ax.set_xlabel("frame", color=MUTED, labelpad=8)
        ax.set_xticks(frames[::2])
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color="#e4e3df", lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#bdbcb6")
        ax.tick_params(colors=MUTED)
    axes[0].set_ylabel("visible cloth pixels (224 x 224 frame)", color=MUTED, labelpad=10)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 0.0), columnspacing=2.5, handlelength=3)
    fig.tight_layout(rect=(0.01, 0.09, 0.99, 0.98), w_pad=3)
    out = D / "cloth_curves.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main()
