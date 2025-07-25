import numpy as np


def chw_to_hwc(img):
    return np.transpose(img, axes=[1, 2, 0]).copy()


def crop(img, size):
    H, W = size
    return img[:, :, :H, :W]
