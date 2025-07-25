import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from datasets import RESIZE6KDataset

if __name__ == "__main__":
    dataset = RESIZE6KDataset(
        "/home/tmy/Desktop/RESIDE-6K",
        "train",
    )
    print(len(dataset))
