import numpy as np


def mirror_x(v: np.ndarray):
    res = v.copy()
    res[::3] = -res[::3]
    return res


def mirror_y(v: np.ndarray):
    res = v.copy()
    res[1::3] = -res[1::3]
    return res


def augment(v):
    # returns the four augmentations
    # (original, mirror_x, mirror_y, 180 deg rotation)
    return [v.copy(), mirror_x(v), mirror_y(v), mirror_x(mirror_y(v))]
