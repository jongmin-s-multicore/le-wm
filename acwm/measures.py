"""The physical quantity of each ACWM-Phys environment, measured on a 224x224 RGB frame with the colour masks of
issue jongmin-s-multicore/thread0#6 (thread0 repro/eval/eval_acwm_phys_quant.py and eval_pour_water.py).

  cloth_move  visible cloth pixels (orange front + pink back)                         last grid frame
  push_sand   sand pixels (light blue)                                                last grid frame
  pour_water  water pixels in the floor band of the destination box                   mean of grid frames >= 250
  stack_cube  stacked: the yellow cube centre right above the green one (image rule)  last grid frame, failed episodes

Grid frames are 0, 5, 10, ... up to the last frame that an action block reaches (k_full of thread0
repro/eval/eval_acwm_phys.py): min(actions // 5, (frames - 1) // 5) + 1 frames.
"""
import cv2
import numpy as np

from cloth_measure import cloth_px, hue

FS = 5
MIN_PX = 6
STACK = dict(dx=11.0, dy=(-35.0, -6.5))  # yellow centre relative to green at 224 px
BAND_360 = (140, 225, 310, 300)  # Pour Water box floor band x0, y0, x1, y1 at 360 px
BAND = tuple(int(round(v * 224 / 360.0)) for v in BAND_360)
FILL_FROM = 250
KIND = {"cloth_move": "count", "push_sand": "count", "pour_water": "count", "stack_cube": "binary"}
NAME = {"cloth_move": "cloth pixels", "push_sand": "sand pixels", "pour_water": "water pixels in the box band",
        "stack_cube": "stacked (yellow on green)"}


def blob(m):
    n = int(m.sum())
    if n < MIN_PX:
        return n, np.nan, np.nan
    ys, xs = np.nonzero(m)
    return n, float(xs.mean()), float(ys.mean())


def sand_px(rgb):
    return int(hue(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV), 90, 102, 100).sum())


def water_band_px(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    m = (hsv[..., 0] >= 98) & (hsv[..., 0] <= 110) & (hsv[..., 1] > 70) & (hsv[..., 2] > 100)
    x0, y0, x1, y1 = BAND
    return int(m[y0:y1, x0:x1].sum())


def stacked(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    ny, yx, yy = blob(hue(hsv, 26, 33, 180, 130))
    ng, gx, gy = blob(hue(hsv, 55, 66, 150, 60))
    if not (ny >= 15 and ng >= 15):
        return 0
    dx, dy = yx - gx, yy - gy
    return int(abs(dx) < STACK["dx"] and STACK["dy"][0] < dy < STACK["dy"][1])


MEASURE = {"cloth_move": cloth_px, "push_sand": sand_px, "pour_water": water_band_px, "stack_cube": stacked}


def k_full(n_rows, n_actions):
    return min(n_actions // FS, (n_rows - 1) // FS) + 1


def summary_frames(env, grid):
    """indices into the grid frames that make the reported value"""
    if env == "pour_water":
        return [i for i, t in enumerate(grid) if t >= FILL_FROM]
    return [len(grid) - 1]
