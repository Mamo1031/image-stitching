"""Image stitching with translation / similarity / affine / homography motion models.

The translation, similarity, and affine estimators solve the normal equations
A p = b explicitly, where A = sum_i J(x_i)^T J(x_i) and b = sum_i J(x_i)^T dx_i.
This is the standard linear-least-squares formulation for 2D motion estimation.

Homography uses cv2.findHomography (RANSAC).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_images(directory: str, max_dim: int = 1200) -> Tuple[List[np.ndarray], List[str]]:
    """Load JPEG/PNG images sorted by filename, optionally downscaling.

    Args:
        directory: folder containing the image series.
        max_dim:   if positive, resize so that max(h, w) <= max_dim.

    Returns:
        (images, names)
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    files = sorted(
        f for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in exts
    )
    images: List[np.ndarray] = []
    for f in files:
        img = cv2.imread(os.path.join(directory, f))
        if img is None:
            continue
        if max_dim > 0:
            h, w = img.shape[:2]
            s = max_dim / max(h, w)
            if s < 1.0:
                img = cv2.resize(img, (int(round(w * s)), int(round(h * s))),
                                 interpolation=cv2.INTER_AREA)
        images.append(img)
    return images, files


# ---------------------------------------------------------------------------
# Feature detection & pairwise matching
# ---------------------------------------------------------------------------

@dataclass
class ImageFeatures:
    keypoints: List[cv2.KeyPoint]
    descriptors: np.ndarray  # (N, D)
    points: np.ndarray       # (N, 2) float32, the (x, y) pixel coords


def detect_features(image: np.ndarray, method: str = "sift",
                    n_features: int = 4000) -> ImageFeatures:
    """Detect keypoints + descriptors with SIFT (default) or AKAZE/ORB."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    method = method.lower()
    if method == "sift":
        detector = cv2.SIFT_create(nfeatures=n_features)
    elif method == "akaze":
        detector = cv2.AKAZE_create()
    elif method == "orb":
        detector = cv2.ORB_create(nfeatures=n_features)
    else:
        raise ValueError(f"unknown detector: {method}")
    kp, desc = detector.detectAndCompute(gray, None)
    pts = np.array([k.pt for k in kp], dtype=np.float32) if kp else np.zeros((0, 2), np.float32)
    return ImageFeatures(keypoints=kp, descriptors=desc, points=pts)


def match_pair(fa: ImageFeatures, fb: ImageFeatures,
               ratio: float = 0.75, cross_check: bool = True) -> List[cv2.DMatch]:
    """Mutual-best + Lowe ratio test."""
    if fa.descriptors is None or fb.descriptors is None:
        return []
    if fa.descriptors.dtype == np.uint8:
        norm = cv2.NORM_HAMMING
    else:
        norm = cv2.NORM_L2
    bf = cv2.BFMatcher(norm)
    knn_ab = bf.knnMatch(fa.descriptors, fb.descriptors, k=2)
    good: List[cv2.DMatch] = []
    for pair in knn_ab:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    if not cross_check:
        return good
    # Mutual best check
    knn_ba = bf.knnMatch(fb.descriptors, fa.descriptors, k=1)
    best_ba = {p[0].queryIdx: p[0].trainIdx for p in knn_ba if p}
    return [m for m in good if best_ba.get(m.trainIdx) == m.queryIdx]


def draw_matches(img_a: np.ndarray, fa: ImageFeatures,
                 img_b: np.ndarray, fb: ImageFeatures,
                 matches: Sequence[cv2.DMatch], max_draw: int = 80) -> np.ndarray:
    matches_sorted = sorted(matches, key=lambda m: m.distance)[:max_draw]
    return cv2.drawMatches(img_a, fa.keypoints, img_b, fb.keypoints,
                           matches_sorted, None,
                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


# ---------------------------------------------------------------------------
# Multi-image tracks (a track = same physical point in >= 2 images)
# ---------------------------------------------------------------------------

class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def build_tracks(features: List[ImageFeatures],
                 pairwise_matches: Dict[Tuple[int, int], List[cv2.DMatch]]
                 ) -> List[Dict[int, Tuple[float, float]]]:
    """Chain pairwise matches into multi-image tracks via union-find.

    Each keypoint in image `img_idx` is identified by a global node id.
    A pair-match links two nodes; tracks = connected components with >= 2 nodes.

    Returns:
        list of {image_index: (x, y)}, one dict per track, indexed by unique i.
    """
    sizes = [len(f.keypoints) for f in features]
    offsets = np.cumsum([0] + sizes)

    def node_id(img: int, kp: int) -> int:
        return offsets[img] + kp

    uf = UnionFind(offsets[-1])
    for (i, j), matches in pairwise_matches.items():
        for m in matches:
            uf.union(node_id(i, m.queryIdx), node_id(j, m.trainIdx))

    root_to_track: Dict[int, Dict[int, Tuple[float, float]]] = {}
    for img_idx, feat in enumerate(features):
        for kp_idx, pt in enumerate(feat.points):
            r = uf.find(node_id(img_idx, kp_idx))
            track = root_to_track.setdefault(r, {})
            # Keep first occurrence per image (or could average).
            if img_idx not in track:
                track[img_idx] = (float(pt[0]), float(pt[1]))

    return [t for t in root_to_track.values() if len(t) >= 2]


# ---------------------------------------------------------------------------
# Linear least-squares estimators (translation / similarity / affine)
#
# Each estimator solves A p = b where
#   A = sum_i J(x_i)^T J(x_i)
#   b = sum_i J(x_i)^T (x'_i - x_i)
# Jacobians are re-parameterised so M = I when p = 0.
# ---------------------------------------------------------------------------

def estimate_translation_ls(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """p = (tx, ty).  J = [[1,0],[0,1]]  -> A = N*I, b = sum delta.

    Returns 3x3 homogeneous matrix.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = src.shape[0]
    delta = dst - src
    A = n * np.eye(2)
    b = delta.sum(axis=0)
    p = np.linalg.solve(A, b)
    tx, ty = p
    M = np.eye(3)
    M[0, 2] = tx
    M[1, 2] = ty
    return M


def estimate_similarity_ls(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """p = (tx, ty, a, b),  M = [[1+a, -b, tx],[b, 1+a, ty],[0,0,1]].

    Δx = a*x - b*y + tx
    Δy = b*x + a*y + ty
    J  = [[1,0, x,-y],[0,1, y, x]]
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    x = src[:, 0]
    y = src[:, 1]
    dx = dst[:, 0] - x
    dy = dst[:, 1] - y
    N = src.shape[0]

    Sx, Sy = x.sum(), y.sum()
    Sxx_yy = (x * x + y * y).sum()
    SdX, SdY = dx.sum(), dy.sum()
    SxdX_ydY = (x * dx + y * dy).sum()
    SxdY_mydX = (x * dy - y * dx).sum()

    A = np.array([
        [N, 0, Sx, -Sy],
        [0, N, Sy,  Sx],
        [Sx, Sy, Sxx_yy, 0.0],
        [-Sy, Sx, 0.0, Sxx_yy],
    ])
    b = np.array([SdX, SdY, SxdX_ydY, SxdY_mydX])
    p = np.linalg.solve(A, b)
    tx, ty, a, bb = p
    M = np.array([
        [1.0 + a, -bb, tx],
        [bb, 1.0 + a, ty],
        [0.0, 0.0, 1.0],
    ])
    return M


def estimate_affine_ls(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """p = (tx, ty, a00, a01, a10, a11),  M = [[1+a00, a01, tx],[a10, 1+a11, ty],[0,0,1]].

    Δx = tx + a00*x + a01*y
    Δy = ty + a10*x + a11*y
    J  = [[1,0,x,y,0,0],[0,1,0,0,x,y]]
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    x = src[:, 0]
    y = src[:, 1]
    dx = dst[:, 0] - x
    dy = dst[:, 1] - y
    N = src.shape[0]

    Sx = x.sum(); Sy = y.sum()
    Sxx = (x * x).sum(); Syy = (y * y).sum(); Sxy = (x * y).sum()
    A = np.array([
        [N, 0, Sx,  Sy,  0,   0  ],
        [0, N, 0,   0,   Sx,  Sy ],
        [Sx, 0, Sxx, Sxy, 0,   0  ],
        [Sy, 0, Sxy, Syy, 0,   0  ],
        [0, Sx, 0,   0,   Sxx, Sxy],
        [0, Sy, 0,   0,   Sxy, Syy],
    ])
    b = np.array([
        dx.sum(),
        dy.sum(),
        (x * dx).sum(),
        (y * dx).sum(),
        (x * dy).sum(),
        (y * dy).sum(),
    ])
    p = np.linalg.solve(A, b)
    tx, ty, a00, a01, a10, a11 = p
    M = np.array([
        [1.0 + a00, a01, tx],
        [a10, 1.0 + a11, ty],
        [0.0, 0.0, 1.0],
    ])
    return M


def estimate_homography(src: np.ndarray, dst: np.ndarray,
                        ransac_thresh: float = 3.0) -> np.ndarray:
    """Robust homography via OpenCV's RANSAC findHomography.

    Falls back to the exact 4-point getPerspectiveTransform when given
    exactly 4 correspondences.
    """
    src = np.asarray(src, dtype=np.float32)
    dst = np.asarray(dst, dtype=np.float32)
    if src.shape[0] == 4:
        return cv2.getPerspectiveTransform(src, dst).astype(np.float64)
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thresh)
    return H.astype(np.float64) if H is not None else np.eye(3)


ESTIMATORS = {
    "translation": estimate_translation_ls,
    "similarity":  estimate_similarity_ls,
    "affine":      estimate_affine_ls,
    "homography":  estimate_homography,
}


# ---------------------------------------------------------------------------
# Reprojection error
# ---------------------------------------------------------------------------

def apply_transform(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)
    ones = np.ones((pts.shape[0], 1))
    hom = np.hstack([pts, ones])
    out = hom @ M.T
    return out[:, :2] / out[:, 2:3]


def reprojection_rmse(M: np.ndarray, src: np.ndarray, dst: np.ndarray) -> float:
    pred = apply_transform(M, src)
    return float(np.sqrt(np.mean(np.sum((pred - dst) ** 2, axis=1))))


# ---------------------------------------------------------------------------
# Canvas, warp, blend
# ---------------------------------------------------------------------------

def compute_canvas(images: List[np.ndarray],
                   transforms: List[np.ndarray]) -> Tuple[Tuple[int, int], np.ndarray]:
    """Return (canvas_size_wh, origin_shift_matrix)."""
    all_corners = []
    for img, M in zip(images, transforms):
        h, w = img.shape[:2]
        c = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
        all_corners.append(apply_transform(M, c))
    pts = np.vstack(all_corners)
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    W = int(np.ceil(xmax - xmin))
    H = int(np.ceil(ymax - ymin))
    shift = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]], dtype=np.float64)
    return (W, H), shift


def warp_and_average(images: List[np.ndarray],
                     transforms: List[np.ndarray]) -> np.ndarray:
    """Warp each image to the common canvas, then average where they overlap."""
    (W, H), shift = compute_canvas(images, transforms)
    canvas = np.zeros((H, W, 3), dtype=np.float64)
    weights = np.zeros((H, W), dtype=np.float64)
    for img, M in zip(images, transforms):
        Mw = shift @ M
        warped = cv2.warpPerspective(img, Mw, (W, H), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(0, 0, 0))
        mask = cv2.warpPerspective(
            np.ones(img.shape[:2], dtype=np.float32), Mw, (W, H),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        canvas += warped.astype(np.float64) * mask[..., None]
        weights += mask
    norm = np.maximum(weights, 1e-6)
    canvas /= norm[..., None]
    canvas[weights < 1e-6] = 0
    return np.clip(canvas, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Full pipeline: detect -> match -> tracks -> per-image transform -> stitch
# ---------------------------------------------------------------------------

@dataclass
class StitchResult:
    panorama: np.ndarray
    transforms: List[np.ndarray]      # image i -> reference frame
    canvas_shift: np.ndarray          # applied to put origin at (0, 0)
    rmse_per_pair: List[float]        # adjacent-pair reprojection RMSE
    n_matches_per_pair: List[int]
    n_tracks: int
    ref_idx: int
    model: str


def ransac_inlier_mask(src: np.ndarray, dst: np.ndarray,
                       thresh: float = 3.0) -> np.ndarray:
    """Use cv2.findHomography(RANSAC) only to identify outliers in the matches.

    The mask is used to filter outliers before fitting *all* models, so
    translation/similarity/affine/homography are compared on the same data.
    """
    src = np.asarray(src, dtype=np.float32)
    dst = np.asarray(dst, dtype=np.float32)
    if src.shape[0] < 4:
        return np.ones(src.shape[0], dtype=bool)
    _H, mask = cv2.findHomography(src, dst, cv2.RANSAC, thresh)
    if mask is None:
        return np.ones(src.shape[0], dtype=bool)
    return mask.ravel().astype(bool)


def stitch_series(images: List[np.ndarray],
                  features: List[ImageFeatures],
                  pairwise_matches: Dict[Tuple[int, int], List[cv2.DMatch]],
                  model: str = "homography",
                  ref_idx: int = None,
                  ransac_thresh: float = 3.0) -> StitchResult:
    """Estimate per-image transforms and warp them onto one canvas.

    Strategy: estimate transform between adjacent pairs (i, i+1) such that the
    pair "moves toward" the reference image. Chain those transforms outward
    from the reference. Outliers are removed once via RANSAC homography and
    the same inlier set is used to fit every model.
    """
    if model not in ESTIMATORS:
        raise ValueError(f"unknown model: {model}")
    estimator = ESTIMATORS[model]
    N = len(images)
    if ref_idx is None:
        ref_idx = N // 2

    transforms: List[np.ndarray] = [np.eye(3) for _ in range(N)]
    rmse_per_pair: List[float] = [0.0] * N
    n_matches_per_pair: List[int] = [0] * N

    def src_dst(i: int, j: int) -> Tuple[np.ndarray, np.ndarray]:
        """Matched RANSAC-inlier points from image i (src) -> image j (dst)."""
        key = (i, j) if (i, j) in pairwise_matches else (j, i)
        ms = pairwise_matches[key]
        swap = key != (i, j)
        s_pts: List[Tuple[float, float]] = []
        d_pts: List[Tuple[float, float]] = []
        for m in ms:
            if swap:
                s_pts.append(tuple(features[i].points[m.trainIdx]))
                d_pts.append(tuple(features[j].points[m.queryIdx]))
            else:
                s_pts.append(tuple(features[i].points[m.queryIdx]))
                d_pts.append(tuple(features[j].points[m.trainIdx]))
        s = np.array(s_pts, dtype=np.float64)
        d = np.array(d_pts, dtype=np.float64)
        mask = ransac_inlier_mask(s, d, thresh=ransac_thresh)
        return s[mask], d[mask]

    # Left of reference: map i -> i+1, then chain to ref through transforms[i+1].
    for i in range(ref_idx - 1, -1, -1):
        s, d = src_dst(i, i + 1)
        M_i_to_iplus1 = estimator(s, d)
        transforms[i] = transforms[i + 1] @ M_i_to_iplus1
        rmse_per_pair[i] = reprojection_rmse(M_i_to_iplus1, s, d)
        n_matches_per_pair[i] = len(s)

    # Right of reference: map i -> i-1.
    for i in range(ref_idx + 1, N):
        s, d = src_dst(i, i - 1)
        M_i_to_iminus1 = estimator(s, d)
        transforms[i] = transforms[i - 1] @ M_i_to_iminus1
        rmse_per_pair[i] = reprojection_rmse(M_i_to_iminus1, s, d)
        n_matches_per_pair[i] = len(s)

    panorama = warp_and_average(images, transforms)
    (_W, _H), shift = compute_canvas(images, transforms)

    tracks = build_tracks(features, pairwise_matches)
    return StitchResult(
        panorama=panorama,
        transforms=transforms,
        canvas_shift=shift,
        rmse_per_pair=rmse_per_pair,
        n_matches_per_pair=n_matches_per_pair,
        n_tracks=len(tracks),
        ref_idx=ref_idx,
        model=model,
    )
