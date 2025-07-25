import os
import argparse
import torch
from tqdm import tqdm

from datasets import build_dataloader
from models import APSDCP_Refine
from utils.common import write_img, load_yaml, write_yaml
from utils.image_process import crop, chw_to_hwc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, help="configuration file")
    args = parser.parse_args()

    config = load_yaml(args.config)
    model = APSDCP_Refine()

    saved_model_dir = config["model_path"]
    if os.path.exists(saved_model_dir):
        model.load_state_dict(torch.load(saved_model_dir, weights_only=True, map_location=torch.device("cpu")))
        print(f">> Model path: {saved_model_dir}")
    else:
        print(">> No existing trained model!")
        exit(0)

    model = torch.nn.DataParallel(model)

    test_loader = build_dataloader("test", config["dataset"])

    result_dir = os.path.join(config["save_dir"], config["exp_name"])
    os.makedirs(os.path.join(result_dir, "J"), exist_ok=True)
    write_yaml(os.path.join(result_dir, "config.yml"), config)

    model.cuda()
    model.eval()
    for batch in tqdm(test_loader):
        input = batch["source"].cuda()
        filename = batch["filename"][0]

        with torch.no_grad():
            J = model(input)

        J = crop(J, batch["original_size"]).clamp(0, 1)

        out_J = chw_to_hwc(J.detach().cpu().squeeze(0).numpy())
        write_img(os.path.join(result_dir, "J", filename), out_J)

    print(f">> Results are saved in {result_dir}")


if __name__ == "__main__":
    main()
