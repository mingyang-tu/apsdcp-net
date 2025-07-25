import os
import cv2
from torch.utils.data import Dataset

from .data_util import get_transform, padding
from utils.common import read_img


class SingleImageDataset(Dataset):
    def __init__(self, data_dir, max_size=0, padding_mul=8):
        self.max_size = max_size
        self.padding_mul = padding_mul

        self.transform = get_transform()

        self.root_dir = data_dir
        self.img_names = sorted(os.listdir(self.root_dir))
        self.img_num = len(self.img_names)

    def __len__(self):
        return self.img_num

    def __getitem__(self, idx):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        img_name = self.img_names[idx]
        source_img = read_img(os.path.join(self.root_dir, img_name), self.max_size)

        H, W, C = source_img.shape

        source_img = self.transform(source_img)

        [source_img] = padding([source_img], self.padding_mul)

        return {"source": source_img, "filename": img_name, "original_size": (H, W)}
