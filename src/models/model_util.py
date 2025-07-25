import torch
import torch.nn.functional as F


def multi_windowed_dcp(image, window_size):
    """
    image: (b, c, h, w)
    window_size: list of window size (int / odd)
    """
    B, C, H, W = image.shape
    layer = len(window_size)

    dcps = torch.zeros((B, layer, H, W), dtype=image.dtype, device=image.device)

    rgb_min = torch.amin(image, dim=1, keepdim=True)
    for i, size in enumerate(window_size):
        if size == 1:
            dcps[:, [i], :, :] = rgb_min
        else:
            dcps[:, [i], :, :] = -F.max_pool2d(-rgb_min, kernel_size=size, stride=1, padding=size // 2)

    return dcps
