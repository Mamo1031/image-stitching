"""Driver: run feature matching, track building, and 4-model stitching on a
series of overlapping photos.

Usage:
    python run_stitching.py --input ../images/sample --output ../results
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stitching import (
    detect_features,
    match_pair,
    draw_matches,
    build_tracks,
    stitch_series,
    load_images,
    ESTIMATORS,
)


def matches_pairwise(features: List, n: int, ratio: float = 0.75
                     ) -> Dict[Tuple[int, int], List]:
    pm: Dict[Tuple[int, int], List] = {}
    for i in range(n - 1):
        pm[(i, i + 1)] = match_pair(features[i], features[i + 1], ratio=ratio)
    return pm


def save_match_visualisations(images, features, pairwise, output_dir, names):
    os.makedirs(output_dir, exist_ok=True)
    for (i, j), ms in pairwise.items():
        vis = draw_matches(images[i], features[i], images[j], features[j], ms,
                           max_draw=80)
        out_path = os.path.join(output_dir, f"matches_{i:02d}_{j:02d}.jpg")
        cv2.imwrite(out_path, vis)


def visualise_tracks(images, tracks, output_path, max_tracks: int = 30):
    """Show how the same feature propagates across the series."""
    n = len(images)
    if n == 0 or not tracks:
        return
    # Sort tracks by length (descending), keep longest.
    long_tracks = sorted(tracks, key=len, reverse=True)[:max_tracks]
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3))
    if n == 1:
        axes = [axes]
    rng = np.random.default_rng(0)
    colors = rng.random((len(long_tracks), 3))
    for ax_idx, ax in enumerate(axes):
        ax.imshow(cv2.cvtColor(images[ax_idx], cv2.COLOR_BGR2RGB))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"image {ax_idx}")
        for t_idx, t in enumerate(long_tracks):
            if ax_idx in t:
                x, y = t[ax_idx]
                ax.plot(x, y, "o", markersize=4,
                        markerfacecolor=colors[t_idx],
                        markeredgecolor="white", markeredgewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def crop_zero_borders(img: np.ndarray) -> np.ndarray:
    """Trim all-zero rows/cols around the stitched panorama."""
    gray = img.sum(axis=2) if img.ndim == 3 else img
    rows = np.any(gray > 0, axis=1)
    cols = np.any(gray > 0, axis=0)
    if not rows.any() or not cols.any():
        return img
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    return img[r0:r1 + 1, c0:c1 + 1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="folder of overlapping images")
    parser.add_argument("--output", required=True, help="output folder")
    parser.add_argument("--detector", default="sift", choices=["sift", "akaze", "orb"])
    parser.add_argument("--ratio", type=float, default=0.75)
    parser.add_argument("--max-dim", type=int, default=1200)
    parser.add_argument("--ref-idx", type=int, default=None,
                        help="reference image (default: middle)")
    parser.add_argument("--models", nargs="+",
                        default=["translation", "similarity", "affine", "homography"])
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"[1] Loading images from {args.input} ...")
    images, names = load_images(args.input, max_dim=args.max_dim)
    if len(images) < 2:
        raise SystemExit("need at least 2 images")
    print(f"    {len(images)} images: {names}")
    print(f"    sizes: {[img.shape[:2] for img in images]}")

    print(f"[2] Detecting features ({args.detector}) ...")
    features = [detect_features(img, method=args.detector) for img in images]
    for i, f in enumerate(features):
        print(f"    image {i}: {len(f.keypoints)} keypoints")

    print(f"[3] Matching adjacent pairs (Lowe ratio={args.ratio}, mutual best) ...")
    pairwise = matches_pairwise(features, len(images), ratio=args.ratio)
    for (i, j), ms in pairwise.items():
        print(f"    pair ({i},{j}): {len(ms)} good matches")
    save_match_visualisations(images, features, pairwise,
                              os.path.join(args.output, "matches"), names)

    print("[4] Building multi-image tracks ...")
    tracks = build_tracks(features, pairwise)
    n_per_len = {}
    for t in tracks:
        n_per_len[len(t)] = n_per_len.get(len(t), 0) + 1
    print(f"    {len(tracks)} tracks, length histogram (len:count) = "
          + ", ".join(f"{k}:{v}" for k, v in sorted(n_per_len.items())))
    visualise_tracks(images, tracks,
                     os.path.join(args.output, "tracks.png"))

    # Save tracks index file: a unique index per track listing {image: (x,y)}
    tracks_json = []
    for i, t in enumerate(tracks):
        tracks_json.append({
            "index_i": i,
            "length": len(t),
            "observations": {int(k): list(v) for k, v in t.items()},
        })
    with open(os.path.join(args.output, "tracks.json"), "w") as fp:
        json.dump({"n_tracks": len(tracks), "tracks": tracks_json[:200]}, fp, indent=2)

    # ------------------------------------------------------------------
    # Stitch under each motion model.
    # ------------------------------------------------------------------
    summary = []
    for model in args.models:
        print(f"[5] Stitching with {model} ...")
        res = stitch_series(images, features, pairwise,
                            model=model, ref_idx=args.ref_idx)
        # Save panorama (full + cropped)
        cv2.imwrite(os.path.join(args.output, f"pano_{model}.jpg"), res.panorama)
        cv2.imwrite(os.path.join(args.output, f"pano_{model}_crop.jpg"),
                    crop_zero_borders(res.panorama))
        print(f"    ref idx={res.ref_idx},  "
              f"adjacent RMSEs = "
              + ", ".join(f"{e:.2f}" for e in res.rmse_per_pair if e > 0))
        for i, M in enumerate(res.transforms):
            np.savetxt(os.path.join(args.output, f"M_{model}_{i:02d}.txt"), M, fmt="%.6f")
        summary.append({
            "model": model,
            "ref_idx": res.ref_idx,
            "rmse_per_pair": res.rmse_per_pair,
            "n_matches_per_pair": res.n_matches_per_pair,
            "n_tracks": res.n_tracks,
            "transforms": [M.tolist() for M in res.transforms],
        })

    with open(os.path.join(args.output, "summary.json"), "w") as fp:
        json.dump(summary, fp, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
