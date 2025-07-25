from torch.utils.data import DataLoader

from .paired_image_dataset import RESIZE6KDataset
from .single_image_dataset import SingleImageDataset


def build_dataloader(mode, config):
    if config["name"] == "RESIDE-6K":
        if mode == "train":
            dataset = RESIZE6KDataset(
                config["data_dir"],
                "train",
                max_size=config["max_size"],
                crop_size=config["crop_size"],
                padding_mul=config["padding_mul"],
            )
            loader = DataLoader(
                dataset,
                batch_size=config["batch_size"],
                shuffle=True,
                num_workers=config["num_workers"],
                pin_memory=True,
            )
        elif mode == "valid":
            dataset = RESIZE6KDataset(
                config["data_dir"],
                "valid",
                max_size=config["max_size"],
                padding_mul=config["padding_mul"],
            )
            loader = DataLoader(
                dataset,
                batch_size=1,
                num_workers=config["num_workers"],
                pin_memory=True,
            )
        elif mode == "test":
            dataset = SingleImageDataset(
                config["data_dir"],
                max_size=config["max_size"],
                padding_mul=config["padding_mul"],
            )
            loader = DataLoader(
                dataset,
                batch_size=1,
                num_workers=config["num_workers"],
                pin_memory=True,
            )
        else:
            raise Exception("ERROR: unsupported mode")

    else:
        raise Exception("ERROR: unsupported dataset")

    return loader
