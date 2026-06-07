"""Generate report figures: RMSE-per-pair bar chart and a 2x2 panorama montage."""

from __future__ import annotations

import argparse
import json
import os
from typing import List

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODELS = ["translation", "similarity", "affine", "homography"]


def rmse_chart(summary_path: str, output_path: str) -> None:
    with open(summary_path) as fp:
        data = json.load(fp)
    by_model = {d["model"]: d for d in data}
    ref_idx = next(iter(by_model.values()))["ref_idx"]
    n = len(next(iter(by_model.values()))["rmse_per_pair"])
    pair_labels = []
    for i in range(n):
        if i == ref_idx:
            continue
        if i < ref_idx:
            pair_labels.append(f"({i},{i+1})")
        else:
            pair_labels.append(f"({i-1},{i})")

    x = np.arange(len(pair_labels))
    width = 0.2
    fig, ax = plt.subplots(figsize=(8, 4))
    for k, model in enumerate(MODELS):
        if model not in by_model:
            continue
        vals = [v for i, v in enumerate(by_model[model]["rmse_per_pair"]) if i != ref_idx]
        ax.bar(x + (k - 1.5) * width, vals, width, label=model)
    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels)
    ax.set_xlabel("adjacent pair (i,j) used to estimate transform")
    ax.set_ylabel("inlier reprojection RMSE [px]")
    ax.set_yscale("log")
    ax.set_title("Per-pair reprojection error by motion model")
    ax.legend()
    ax.grid(True, axis="y", which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def crop_zero(img: np.ndarray) -> np.ndarray:
    g = img.sum(axis=2) if img.ndim == 3 else img
    rows = np.any(g > 0, axis=1); cols = np.any(g > 0, axis=0)
    if not rows.any() or not cols.any():
        return img
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    return img[r0:r1 + 1, c0:c1 + 1]


def montage_4(results_dir: str, output_path: str) -> None:
    imgs: List[np.ndarray] = []
    titles: List[str] = []
    for m in MODELS:
        p = os.path.join(results_dir, f"pano_{m}_crop.jpg")
        if not os.path.exists(p):
            continue
        im = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        imgs.append(crop_zero(im))
        titles.append(m)
    n = len(imgs)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.6 * n))
    if n == 1:
        axes = [axes]
    for ax, im, t in zip(axes, imgs, titles):
        ax.imshow(im)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(t, fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def input_thumbnail(input_dir: str, output_path: str, max_dim: int = 800) -> None:
    files = sorted(f for f in os.listdir(input_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    imgs = []
    for f in files:
        im = cv2.imread(os.path.join(input_dir, f))
        if im is None:
            continue
        h, w = im.shape[:2]
        s = max_dim / max(h, w)
        if s < 1.0:
            im = cv2.resize(im, (int(w * s), int(h * s)))
        imgs.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    n = len(imgs)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 2.5))
    if n == 1:
        axes = [axes]
    for ax, im, name in zip(axes, imgs, files):
        ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(name, fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    rmse_chart(os.path.join(args.results, "summary.json"),
               os.path.join(args.results, "rmse_chart.png"))
    montage_4(args.results, os.path.join(args.results, "panorama_montage.png"))
    input_thumbnail(args.input, os.path.join(args.results, "inputs.png"))
    print("Figures written under", args.results)
