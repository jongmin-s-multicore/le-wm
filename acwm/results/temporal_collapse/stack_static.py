"""LeWM Stack Cube: is the (nearly constant) episode embedding a function of the static layout?
Per episode: green / yellow cube centroids in the first and last frame (colour masks, 256 px), which cube moved,
then ridge probes (5-fold CV, episodes) from the episode-mean embedding to each position."""
import subprocess, numpy as np, sys
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_predict, KFold
d = np.load('results/lewm_acwm/eval_stack_cube.npz', allow_pickle=True)
root = 'data/acwm_phys/rigid_dynamics/stack_cube'
def frames(path):
    raw = subprocess.run(['ffmpeg', '-loglevel', 'error', '-i', path, '-vf', 'scale=256:256:flags=area', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, 256, 256, 3).astype(int)
def cents(f):
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    yel = (r > 150) & (g > 150) & (b < 90)
    grn = (g > 70) & (r < 90) & (b < 90) & (g > r + 30)
    ys, xs = np.mgrid[0:256, 0:256]
    out = []
    for m in (grn, yel):
        n = m.sum()
        out += [(m * xs).sum() / max(n, 1), (m * ys).sum() / max(n, 1), n]
    return out
rows, Z, Zc = [], [], []
for i, (name, split) in enumerate(zip(d['names'], d['split'])):
    v = frames(f'{root}/{split}/{name}')
    c0, c1 = cents(v[0]), cents(v[-1])
    rows.append(c0 + c1)
    e = np.asarray(d['emb'][i], dtype=np.float64)
    Z.append(e.mean(0)); Zc.append(e)
rows = np.array(rows); Z = np.array(Z)
gx0, gy0, gn0, yx0, yy0, yn0, gx1, gy1, gn1, yx1, yy1, yn1 = rows.T
gmove = np.hypot(gx1 - gx0, gy1 - gy0); ymove = np.hypot(yx1 - yx0, yy1 - yy0)
moved_is_yellow = ymove > gmove
print(f'episodes {len(Z)}; yellow moved in {moved_is_yellow.mean()*100:.0f} %; median move: moved cube {np.median(np.maximum(gmove,ymove)):.1f} px, other {np.median(np.minimum(gmove,ymove)):.1f} px (256 px frame)')
stat_x = np.where(moved_is_yellow, gx0, yx0); stat_y = np.where(moved_is_yellow, gy0, yy0)
mov_x0 = np.where(moved_is_yellow, yx0, gx0); mov_y0 = np.where(moved_is_yellow, yy0, gy0)
mov_x1 = np.where(moved_is_yellow, yx1, gx1); mov_y1 = np.where(moved_is_yellow, yy1, gy1)
kf = KFold(5, shuffle=True, random_state=0)
def r2(target, X=Z):
    p = cross_val_predict(RidgeCV(alphas=np.logspace(-2, 4, 25)), X, target, cv=kf)
    return 1 - ((p - target) ** 2).sum() / ((target - target.mean()) ** 2).sum()
for lab, t in [('stationary cube x', stat_x), ('stationary cube y', stat_y), ('moved cube start x', mov_x0), ('moved cube start y', mov_y0),
               ('moved cube end x', mov_x1), ('moved cube end y', mov_y1), ('which cube moved (yellow=1)', moved_is_yellow.astype(float))]:
    print(f'  episode-mean embedding -> {lab:28s} R2 = {r2(t):.2f}')
# frame-level: does the embedding at the first vs last frame differ in what it says about the moved cube?
Z0 = np.array([e[0] for e in Zc]); Z1 = np.array([e[-1] for e in Zc])
print(f'  first-frame embedding -> moved cube start x R2 = {r2(mov_x0, Z0):.2f};  last-frame embedding -> moved cube end x R2 = {r2(mov_x1, Z1):.2f}')
print(f'  |z_last - z_first| mean {np.linalg.norm(Z1 - Z0, axis=1).mean():.2f} vs distance between two episodes mean {np.mean([np.linalg.norm(Z[i]-Z[j]) for i in range(0,300,7) for j in range(1,300,11) if i!=j]):.2f}')

# nonlinear: nearest neighbour in embedding space vs stationary-cube distance
from scipy.stats import spearmanr
P = np.stack([stat_x, stat_y], 1)
D_z = np.linalg.norm(Z[:, None] - Z[None], axis=-1); np.fill_diagonal(D_z, np.inf)
nn = D_z.argmin(1)
rng = np.random.default_rng(0); rnd = rng.permutation(len(Z))
print(f'  stationary-cube distance to embedding nearest neighbour: median {np.median(np.linalg.norm(P - P[nn], axis=1)):.1f} px vs random episode {np.median(np.linalg.norm(P - P[rnd], axis=1)):.1f} px')
iu = np.triu_indices(len(Z), 1)
D_p = np.linalg.norm(P[:, None] - P[None], axis=-1)
print(f'  Spearman(embedding distance, stationary-cube distance) over pairs = {spearmanr(D_z[iu], D_p[iu])[0]:.3f}')
# static image regions: per-episode first-frame pixel statistics far from cubes and robot (top-left room corner, bottom table band)
bg = []
for i, (name, split) in enumerate(zip(d['names'], d['split'])):
    v = frames(f'{root}/{split}/{name}')
    f0 = v[0]
    bg.append(np.concatenate([f0[:40, :40].reshape(-1, 3).mean(0), f0[230:, :].reshape(-1, 3).mean(0), f0[:40, 216:].reshape(-1, 3).mean(0)]))
bg = np.array(bg)
print('  first-frame corner/table colour spread across episodes (std per channel):', np.round(bg.std(0), 2))
for lab, t in [('room corner mean R', bg[:, 0]), ('table band mean R', bg[:, 3])]:
    if t.std() > 1e-6:
        print(f'  episode-mean embedding -> {lab:20s} R2 = {r2(t):.2f}')
