import random
import math
import numpy as np
from torchvision import transforms
import torch.nn.functional as F


def get_transform():
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ]
    )


def augment(imgs=[], size=256):
    H, W, C = imgs[0].shape

    Hs = random.randint(0, H - size)
    Ws = random.randint(0, W - size)

    # random crop
    for i in range(len(imgs)):
        assert imgs[i].shape[0] == H and imgs[i].shape[1] == W
        imgs[i] = imgs[i][Hs : (Hs + size), Ws : (Ws + size), :]

    # horizontal flip
    if random.randint(0, 1) == 1:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1)

    return imgs


def padding(imgs, times):
    for i in range(len(imgs)):
        C, H, W = imgs[i].shape
        h_new = math.ceil(H / times) * times
        w_new = math.ceil(W / times) * times

        padding_h = h_new - H
        padding_w = w_new - W

        imgs[i] = F.pad(imgs[i], (0, padding_w, 0, padding_h), mode="reflect")

    return imgs
