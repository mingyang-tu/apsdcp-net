import torch
import os
import sys
import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from models.model_util import multi_windowed_dcp

image = torch.rand((1, 3, 512, 512))
window_size = [1, 3, 7, 15, 31, 63, 127]

outputs = multi_windowed_dcp(image.cuda(), window_size)

image_np = image.squeeze().permute(1, 2, 0).numpy()
rgb_min = np.min(image_np, axis=2)
for i, w in enumerate(window_size):
    gt = cv2.erode(rgb_min, np.ones((w, w)))
    output = outputs[0, i].detach().cpu().numpy()
    assert np.allclose(gt, output, atol=1e-6)
    print(f"Window size: {w} - OK")
