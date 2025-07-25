import random
import yaml
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def read_img(filename, maxsize=0):
    img = cv2.imread(filename)
    if maxsize > 0:
        M = max(img.shape)
        if M > maxsize:
            ratio = float(maxsize) / float(M)
            img = cv2.resize(img, (0, 0), fx=ratio, fy=ratio)
    return img[:, :, ::-1]


def write_img(filename, img):
    if img.ndim == 3:
        img = img[:, :, ::-1]
    img = np.round((img.copy() * 255.0)).astype("uint8")
    cv2.imwrite(filename, img)


def load_yaml(filename):
    with open(filename, "r") as file:
        data = yaml.safe_load(file)
    return data


def write_yaml(filename, data):
    with open(filename, "w") as file:
        yaml.dump(data, file, default_flow_style=False, allow_unicode=True)


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)


def save_dcp_weights(filename, weight):
    """
    weight: (C, H, W)
    """
    C, H, W = weight.shape
    fig, axs = plt.subplots(1, C, figsize=(4.5 * C * W // H, 5), constrained_layout=True)
    for i in range(C):
        im = axs[i].imshow(weight[i, :, :], cmap="jet", vmin=0, vmax=1)
        axs[i].axis("off")
        axs[i].set_title(f"Channel {i}")
    fig.suptitle("DCP weights", fontsize=20)
    fig.colorbar(im, ax=axs, orientation="horizontal", fraction=0.05, aspect=50)
    plt.savefig(filename)
    plt.close()
