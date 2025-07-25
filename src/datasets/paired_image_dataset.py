import os
import cv2
from torch.utils.data import Dataset

from .data_util import get_transform, augment, padding
from utils.common import read_img


class RESIZE6KDataset(Dataset):
    def __init__(self, data_dir, mode, max_size=0, crop_size=256, padding_mul=8):
        self.mode = mode
        self.max_size = max_size
        self.crop_size = crop_size
        self.padding_mul = padding_mul

        self.transform = get_transform()

        if mode == "train":
            self.root_dir = os.path.join(data_dir, "train")
            self.img_names = sorted(os.listdir(os.path.join(self.root_dir, "GT")))
        elif mode == "valid":
            self.root_dir = os.path.join(data_dir, "test")
            self.img_names = sorted(os.listdir(os.path.join(self.root_dir, "GT")))
            # choose 1/10 of the images for validation
            self.img_names = [self.img_names[i] for i in range(len(self.img_names)) if i % 10 == 0]
        else:
            raise Exception("ERROR: unsupported mode")

        self.img_num = len(self.img_names)

    def __len__(self):
        return self.img_num

    def __getitem__(self, idx):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        img_name = self.img_names[idx]
        source_img = read_img(os.path.join(self.root_dir, "hazy", img_name), self.max_size)
        target_img = read_img(os.path.join(self.root_dir, "GT", img_name), self.max_size)

        H, W, C = source_img.shape

        if self.mode == "train":
            [source_img, target_img] = augment([source_img, target_img], self.crop_size)

        source_img = self.transform(source_img)
        target_img = self.transform(target_img)

        if self.mode != "train":
            [source_img, target_img] = padding([source_img, target_img], self.padding_mul)

        return {"source": source_img, "target": target_img, "filename": img_name, "original_size": (H, W)}


class PairedDataset(Dataset):
    def __init__(self, data_dir, sub_dir, mode, max_size=0, crop_size=256, padding_mul=8):
        self.mode = mode
        self.max_size = max_size
        self.crop_size = crop_size
        self.padding_mul = padding_mul

        self.transform = get_transform()

        self.root_dir = os.path.join(data_dir, sub_dir)
        self.img_names = sorted(os.listdir(os.path.join(self.root_dir, "GT")))
        self.img_num = len(self.img_names)

    def __len__(self):
        return self.img_num

    def __getitem__(self, idx):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        img_name = self.img_names[idx]
        source_img = read_img(os.path.join(self.root_dir, "hazy", img_name), self.max_size)
        target_img = read_img(os.path.join(self.root_dir, "GT", img_name), self.max_size)

        H, W, C = source_img.shape

        if self.mode == "train":
            [source_img, target_img] = augment([source_img, target_img], self.crop_size)

        source_img = self.transform(source_img)
        target_img = self.transform(target_img)

        if self.mode != "train":
            [source_img, target_img] = padding([source_img, target_img], self.padding_mul)

        return {"source": source_img, "target": target_img, "filename": img_name, "original_size": (H, W)}


class ITSDataset(Dataset):
    def __init__(self, data_dir, mode, max_size=0, crop_size=256, padding_mul=8):
        self.mode = mode
        self.max_size = max_size
        self.crop_size = crop_size
        self.padding_mul = padding_mul

        self.transform = get_transform()

        self.root_dir = data_dir
        self.img_names = sorted(os.listdir(os.path.join(self.root_dir, "clear")))

        if mode == "train":
            self.img_names = [self.img_names[i] for i in range(len(self.img_names)) if i % 30 != 0]
        elif mode == "valid":
            self.img_names = [self.img_names[i] for i in range(len(self.img_names)) if i % 30 == 0]
        else:
            raise Exception("ERROR: unsupported mode")

        self.img_num = len(self.img_names)

    def __len__(self):
        return self.img_num

    def __getitem__(self, idx):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        img_name = self.img_names[idx]
        source_img = read_img(os.path.join(self.root_dir, "haze", img_name), self.max_size)
        target_img = read_img(os.path.join(self.root_dir, "clear", img_name), self.max_size)

        H, W, C = source_img.shape

        if self.mode == "train":
            [source_img, target_img] = augment([source_img, target_img], self.crop_size)

        source_img = self.transform(source_img)
        target_img = self.transform(target_img)

        if self.mode != "train":
            [source_img, target_img] = padding([source_img, target_img], self.padding_mul)

        return {"source": source_img, "target": target_img, "filename": img_name, "original_size": (H, W)}
