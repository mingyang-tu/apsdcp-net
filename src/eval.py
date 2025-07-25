import os
import sys
import cv2

from utils.common import AverageMeter
from metrics import calculate_psnr, calculate_ssim_256, calculate_ssim


if __name__ == "__main__":
    results_root = sys.argv[1]
    dehazed_dir = os.path.join(results_root, "J")
    gt_dir = sys.argv[2]

    write_file = os.path.join(results_root, "psnr_ssim.txt")

    filenames = sorted(os.listdir(dehazed_dir))

    PSNR = AverageMeter()
    SSIM = AverageMeter()
    with open(write_file, "w") as f:
        for filename in filenames:
            dehazed_file = os.path.join(dehazed_dir, filename)
            gt_file = os.path.join(gt_dir, filename)

            dehazed_img = cv2.imread(dehazed_file)
            gt_img = cv2.imread(gt_file)

            if dehazed_img is None or gt_img is None:
                raise Exception("Image not found")

            # resize to the same size
            if dehazed_img.shape[0] != gt_img.shape[0] or dehazed_img.shape[1] != gt_img.shape[1]:
                print(f"Resizing {filename}...")
                gt_img = cv2.resize(gt_img, (dehazed_img.shape[1], dehazed_img.shape[0]))

            psnr = calculate_psnr(dehazed_img, gt_img)
            ssim = calculate_ssim_256(dehazed_img, gt_img)

            PSNR.update(psnr)
            SSIM.update(ssim)

            msg = f"[{filename}] PSNR: {psnr:.3f}, SSIM: {ssim:.5f}"
            print(msg)
            f.write(msg + "\n")

        msg = f"\nAverage PSNR: {PSNR.avg:.3f}, Average SSIM: {SSIM.avg:.5f}"
        print(msg)
        f.write(msg + "\n")
