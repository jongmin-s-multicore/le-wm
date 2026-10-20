"""Visible cloth pixels of a 224x224 RGB frame: orange front + pink back, the HSV mask of thread0
repro/eval/eval_acwm_phys_quant.py (Cloth Move). Grid constants of the Cloth Move episodes."""
import cv2

FS, T = 5, 65  # frameskip, frames per episode (64 actions)
GRID = list(range(0, T - 1, FS))  # frames 0, 5, ..., 60


def hue(hsv, lo, hi, s, v=80):
    h = hsv[..., 0]
    m = (h >= lo) & (h <= hi) if lo <= hi else ((h >= lo) | (h <= hi))
    return m & (hsv[..., 1] > s) & (hsv[..., 2] > v)


def cloth_px(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return int((hue(hsv, 8, 16, 120) | hue(hsv, 158, 170, 60)).sum())
