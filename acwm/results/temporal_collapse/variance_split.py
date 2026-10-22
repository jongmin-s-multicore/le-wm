"""Split LeWM embedding variance into within-episode (temporal) and between-episode parts, per ACWM-Phys environment.
usage (cwd = ROOT): le-wm/.venv/bin/python -I results/lewm_acwm/temporal_collapse/variance_split.py"""
import numpy as np
for env in ('cloth_move', 'push_sand', 'pour_water', 'stack_cube'):
    d = np.load(f'results/lewm_acwm/eval_{env}.npz', allow_pickle=True)
    E = [np.asarray(e, dtype=np.float64) for e in d['emb']]
    allz = np.concatenate(E)
    tot = allz.var(0).mean()
    within = np.mean([e.var(0).mean() for e in E])
    between = np.stack([e.mean(0) for e in E]).var(0).mean()
    print(f'{env:11s} total var/dim {tot:.3f}  within-episode {within:.4f}  between-episode {between:.3f}  within/total {within/tot:.3f}'
          f'  |z_last - z_first| {np.mean([np.linalg.norm(e[-1] - e[0]) for e in E]):.2f}  mean |z| {np.linalg.norm(allz, axis=1).mean():.1f}')
