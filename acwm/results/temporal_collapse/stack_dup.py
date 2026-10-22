import subprocess, numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
d = np.load('results/lewm_acwm/eval_stack_cube.npz', allow_pickle=True)
root = 'data/acwm_phys/rigid_dynamics/stack_cube'
def first(path):
    raw = subprocess.run(['ffmpeg', '-loglevel', 'error', '-i', path, '-vf', 'select=eq(n\\,0),scale=64:64:flags=area', '-vframes', '1', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(64, 64, 3).astype(np.float32)
F = np.stack([first(f'{root}/{s}/{n}') for n, s in zip(d['names'], d['split'])])
Z = np.stack([np.asarray(e, dtype=np.float64).mean(0) for e in d['emb']])
n = len(F)
Df = cdist(F.reshape(n, -1), F.reshape(n, -1), 'cityblock') / F[0].size
Dt = cdist(F[:, :22].reshape(n, -1), F[:, :22].reshape(n, -1), 'cityblock') / F[0, :22].size  # room background, top third
Dz = cdist(Z, Z)
for M in (Df, Dt, Dz):
    np.fill_diagonal(M, np.inf)
nn = Dz.argmin(1); iu = np.triu_indices(n, 1)
twins = np.triu(Df < 1, 1)
print('pairs with near-identical first frame (MAD<1 grey level):', int(twins.sum()), '| episodes whose embedding-NN is such a twin:', int((Df[np.arange(n), nn] < 1).sum()), 'of', n)
print('first-frame MAD to embedding-NN: median', round(float(np.median(Df[np.arange(n), nn])), 2), '| random pair median', round(float(np.median(Df[iu])), 2))
print('room-background MAD to embedding-NN: median', round(float(np.median(Dt[np.arange(n), nn])), 2), '| random pair median', round(float(np.median(Dt[iu])), 2))
print('Spearman(embedding distance, whole first-frame distance) =', round(spearmanr(Dz[iu], Df[iu])[0], 3), '| room background only =', round(spearmanr(Dz[iu], Dt[iu])[0], 3))
if twins.any():
    print('twin pair splits:', dict(zip(*np.unique([d['split'][i] + '/' + d['split'][j] for i, j in zip(*np.where(twins))], return_counts=True))))
    print('embedding distance within twin pairs: median', round(float(np.median(Dz[twins])), 2), '| all pairs median', round(float(np.median(Dz[iu])), 2))
