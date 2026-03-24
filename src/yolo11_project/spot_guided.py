from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.segmentation import slic


@dataclass
class SpotGuidedConfig:
    slic_segments: int = 200
    slic_compactness: float = 12.0
    glcm_distances: tuple[int, ...] = (1, 2)
    glcm_angles: tuple[float, ...] = (0.0, np.pi / 4, np.pi / 2)
    entropy_threshold_quantile: float = 0.75
    blend_alpha: float = 0.45


def _normalize_map(score_map: np.ndarray) -> np.ndarray:
    score_min = float(score_map.min())
    score_max = float(score_map.max())
    if score_max - score_min < 1e-8:
        return np.zeros_like(score_map, dtype=np.float32)
    return ((score_map - score_min) / (score_max - score_min)).astype(np.float32)


def _glcm_texture_score(gray_patch: np.ndarray, distances: Iterable[int], angles: Iterable[float]) -> float:
    if gray_patch.size == 0:
        return 0.0

    # 限制灰度级到 32 级，降低 GLCM 计算成本。
    quantized = (gray_patch.astype(np.float32) / 8.0).clip(0, 31).astype(np.uint8)
    glcm = graycomatrix(
        quantized,
        distances=list(distances),
        angles=list(angles),
        levels=32,
        symmetric=True,
        normed=True,
    )
    contrast = float(graycoprops(glcm, "contrast").mean())
    homogeneity = float(graycoprops(glcm, "homogeneity").mean())
    energy = float(graycoprops(glcm, "energy").mean())
    return contrast + (1.0 - homogeneity) + (1.0 - energy)


def _segment_entropy(gray_image: np.ndarray, labels: np.ndarray, segment_id: int) -> float:
    region = gray_image[labels == segment_id]
    if region.size == 0:
        return 0.0
    hist = np.bincount(region, minlength=256).astype(np.float64)
    hist = hist / (hist.sum() + 1e-12)
    hist = hist[hist > 0]
    return float(-(hist * np.log2(hist + 1e-12)).sum())


def build_candidate_map(image_bgr: np.ndarray, cfg: SpotGuidedConfig) -> np.ndarray:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    segments = slic(
        rgb,
        n_segments=cfg.slic_segments,
        compactness=cfg.slic_compactness,
        start_label=0,
    )

    score_map = np.zeros(gray.shape, dtype=np.float32)
    for segment_id in np.unique(segments):
        mask = segments == segment_id
        y_idx, x_idx = np.where(mask)
        if y_idx.size == 0:
            continue

        y_min, y_max = int(y_idx.min()), int(y_idx.max()) + 1
        x_min, x_max = int(x_idx.min()), int(x_idx.max()) + 1
        patch = gray[y_min:y_max, x_min:x_max]

        entropy = _segment_entropy(gray, segments, int(segment_id))
        texture = _glcm_texture_score(patch, cfg.glcm_distances, cfg.glcm_angles)
        score_map[mask] = entropy * 0.6 + texture * 0.4

    norm_scores = _normalize_map(score_map)
    threshold = float(np.quantile(norm_scores, cfg.entropy_threshold_quantile))
    candidate = (norm_scores >= threshold).astype(np.float32)
    return cv2.GaussianBlur(candidate, (7, 7), 0)


def apply_spot_guided_attention(image_bgr: np.ndarray, cfg: SpotGuidedConfig) -> tuple[np.ndarray, np.ndarray]:
    candidate_map = build_candidate_map(image_bgr, cfg)
    attention = np.repeat(candidate_map[:, :, None], repeats=3, axis=2)

    sharpened = cv2.detailEnhance(image_bgr, sigma_s=10, sigma_r=0.2)
    guided = image_bgr.astype(np.float32) * (1.0 - cfg.blend_alpha * attention) + sharpened.astype(np.float32) * (
        cfg.blend_alpha * attention
    )
    return guided.clip(0, 255).astype(np.uint8), candidate_map