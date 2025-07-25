import os
import argparse
import math
import torch
import torch.nn as nn
from torch import autocast, GradScaler
from tqdm import tqdm
from datetime import datetime
from tensorboardX import SummaryWriter

from datasets import build_dataloader
from models import APSDCP
from metrics import calculate_psnr_pt
from utils.common import AverageMeter, set_seed, load_yaml, write_yaml, save_dcp_weights
from utils.image_process import crop
from utils.loss import CharbonnierLoss


def cosine_with_warmup(warmup_steps, total_steps, min_ratio):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            cosine_decay = 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
            return cosine_decay * (1 - min_ratio) + min_ratio

    return lr_lambda


class Trainer:
    def __init__(self, config):
        self.config = config

        self.model = APSDCP()
        self.model = nn.DataParallel(self.model).cuda()

        self.criterion = CharbonnierLoss()

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config["lr"])

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            cosine_with_warmup(config["warmup_iter"], config["max_iter"], 5e-3),
        )
        self.scaler = GradScaler()

        self.train_loader = build_dataloader("train", config["dataset"])
        self.val_loader = build_dataloader("valid", config["dataset"])

        self.metric_best = 0

        # loggers
        exp_name = datetime.now().strftime("%Y_%m%d_%H%M%S") + ("_" if len(config["exp_name"]) > 0 else "") + config["exp_name"]

        self.log_dir = os.path.join(config["save_dir"], "log", exp_name)
        self.model_dir = os.path.join(config["save_dir"], "model", exp_name)
        self.weights_dir = os.path.join(self.log_dir, "weights")

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.weights_dir, exist_ok=True)

        write_yaml(os.path.join(self.log_dir, "config.yml"), config)

        self.writer = SummaryWriter(log_dir=self.log_dir)

    def update_train_log(self, step, loss, lr):
        self.writer.add_scalar("train/loss", loss, step)
        self.writer.add_scalar("train/lr", lr, step)

    def update_valid_log(self, step, psnr, J, J_refine, t, t_refine):
        self.writer.add_scalar("valid/psnr", psnr, step)
        self.writer.add_image("valid/J", J, step)
        self.writer.add_image("valid/J_refine", J_refine, step)
        self.writer.add_image("valid/t", t, step)
        self.writer.add_image("valid/t_refine", t_refine, step)

    def save_model(self, model_name):
        save_path = os.path.join(self.model_dir, model_name)
        torch.save(self.model.module.state_dict(), save_path)
        print(f"Save model to {save_path}.")

    def _valid(self, step):
        PSNR = AverageMeter()
        torch.cuda.empty_cache()

        self.model.eval()
        for batch in self.val_loader:
            source_img = batch["source"].cuda()
            target_img = batch["target"].cuda()

            with torch.no_grad(), torch.autocast(device_type="cuda"):
                J, J_refine, t, t_refine, weights = self.model(source_img)

            J_refine = crop(J_refine, batch["original_size"]).clamp(0, 1)
            target_img = crop(target_img, batch["original_size"])

            psnr = calculate_psnr_pt(J_refine, target_img)
            PSNR.update(psnr, source_img.size(0))

        J = crop(J, batch["original_size"]).clamp(0, 1).float()
        J_refine = crop(J_refine, batch["original_size"]).clamp(0, 1).float()
        t = crop(t, batch["original_size"]).clamp(0, 1).float()
        t_refine = crop(t_refine, batch["original_size"]).clamp(0, 1).float()

        avg_psnr = PSNR.avg
        self.update_valid_log(
            step,
            avg_psnr,
            J.detach().cpu().squeeze(0),
            J_refine.detach().cpu().squeeze(0),
            t.detach().cpu().squeeze(0),
            t_refine.detach().cpu().squeeze(0),
        )
        print(f">> Validation : step {step}, PSNR = {avg_psnr}")

        weights = crop(weights, batch["original_size"])
        out_weights = weights.detach().cpu().squeeze(0).numpy()
        save_dcp_weights(os.path.join(self.weights_dir, f"step_{step}.png"), out_weights)

        if avg_psnr > self.metric_best:
            self.save_model(f"model_best_{step}.pth")
            self.metric_best = avg_psnr

    def start_training(self):
        print("========== Start Training ==========")

        curr_step = 0
        pbar = tqdm(range(self.config["max_iter"]))
        while True:
            self.model.train()
            for batch in self.train_loader:
                source_img = batch["source"].cuda()
                target_img = batch["target"].cuda()

                with autocast(device_type="cuda"):
                    J, J_refine, t, t_refine, weights = self.model(source_img)
                    loss = self.criterion(J_refine, target_img)
                    if curr_step < self.config["fix_pss_iter"]:
                        loss += self.criterion(J, target_img)

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                self.update_train_log(curr_step, loss.item(), self.optimizer.param_groups[0]["lr"])
                self.scheduler.step()

                if curr_step % self.config["eval_freq"] == 0:
                    self._valid(curr_step)
                    self.model.train()
                    torch.cuda.empty_cache()

                curr_step += 1
                pbar.update(1)
                if curr_step == self.config["fix_pss_iter"]:
                    self.model.module.fix_pssnet()
                if curr_step > self.config["max_iter"]:
                    break
            if curr_step > self.config["max_iter"]:
                break

        self.save_model("model_final.pth")


def main():
    set_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, help="configuration file")
    args = parser.parse_args()

    config = load_yaml(args.config)

    trainer = Trainer(config)
    trainer.start_training()


if __name__ == "__main__":
    main()
