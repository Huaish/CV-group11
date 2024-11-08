import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import argparse
from main import sift, matcher, show_matches, read_image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-img", type=str, default="data/hill1.JPG")
    parser.add_argument("--right-img", type=str, default="data/hill2.JPG")
    parser.add_argument("--output", type=str, default="exp")
    args = parser.parse_args()
    return args


#  ================== 1. Interest points detection & feature ==================


def show_keypoints(img_rgb, kp, color=(0, 0, 255), output_path=None):
    img = img_rgb.copy()
    for k in kp:
        x, y = k.pt
        cv2.circle(img, (int(x), int(y)), 1, color, 1)

    if output_path:
        cv2.imwrite(output_path, img)
    return img


def diff_sift_method(path, name, output_dir="output"):

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    methods = ["harris", "sift", "mser"]

    for ax, method in zip(axs, methods):
        img, _, gray = read_image(path)
        kp, des = sift(gray, method=method)
        img_kp = show_keypoints(img, kp)
        ax.imshow(cv2.cvtColor(img_kp, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{method} keypoints")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}_keypoints_comparison.jpg"))


def tune_thres_Harris(path, name, output_dir="output"):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    thresholds = [0.05, 0.01, 0.005]

    for threshold in thresholds:
        img, _, gray = read_image(path)
        kp, des = sift(gray, method="harris", threshold=threshold)
        img_kp = show_keypoints(img, kp)
        ax = axs[thresholds.index(threshold)]
        ax.imshow(cv2.cvtColor(img_kp, cv2.COLOR_BGR2RGB))
        ax.set_title(f"threshold={threshold}")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}_harris_threshold_comparison.jpg"))


def tune_blocksize_Harris(path, name, output_dir="output"):
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    blocksize = (2, 3, 4, 5, 7, 9)
    ksize = (3, 5, 5, 9, 11, 13)
    for i, j in zip(blocksize, ksize):
        img, _, gray = read_image(path)
        dst = cv2.cornerHarris(gray, i, j, 0.04)
        img[dst > 0.01 * dst.max()] = [0, 0, 255]
        ax = axs[blocksize.index(i) // 3, blocksize.index(i) % 3]
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(f"blocksize={i}")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}_harris_blocksize_comparison.jpg"))


#  ================== 2. Feature matching by SIFT features  ==================


def tune_thres_matcher(
    path_l, path_r, name, output_dir="output", thresholds=[0.5, 0.15, 0.75]
):
    method = ["sift", "mser"]

    for i, m in enumerate(method):
        for j, threshold in enumerate(thresholds):
            img_l, gray_l, _ = read_image(path_l)
            img_r, gray_r, _ = read_image(path_r)
            kp1, des1 = sift(gray_l, method=m)
            kp2, des2 = sift(gray_r, method=m)
            matches = matcher(kp1, kp2, des1, des2, threshold=threshold, method="FLANN")

            img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB)
            img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)
            img_concat = np.concatenate((img_l, img_r), axis=1)
            show_matches(
                img_concat,
                matches,
                os.path.join(output_dir, f"{name}_{m}_threshold_{threshold}.jpg"),
            )


if __name__ == "__main__":
    args = parse_args()
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    path = args.left_img
    name = os.path.basename(path).split(".")[0]

    # 1. Interest points detection & feature
    diff_sift_method(path, name, output_dir=output_dir)
    tune_thres_Harris(path, name, output_dir=output_dir)
    tune_blocksize_Harris(path, name, output_dir=output_dir)

    # 2. Feature matching by SIFT features
    path_l = args.left_img
    path_r = args.right_img

    tune_thres_matcher(path_l, path_r, name, output_dir=output_dir)
    # thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # tune_thres_matcher(
    #     path_l, path_r, name, output_dir=output_dir, thresholds=thresholds
    # )
