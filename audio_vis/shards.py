from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:  # optional
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - optional dependency
    YOLO = None
try:  # optional
    from huggingface_hub import hf_hub_download
except ImportError:  # pragma: no cover - optional dependency
    hf_hub_download = None
try:  # optional
    from skimage.segmentation import slic
except ImportError:  # pragma: no cover - optional dependency
    slic = None


@dataclass
class Shard:
    rgba: np.ndarray
    bbox: Tuple[int, int, int, int]
    area: int
    centroid: Tuple[float, float]
    mean_color: Tuple[int, int, int]


class SegmentationEngine:
    def __init__(
        self,
        backend: str = "kmeans",
        model_path: Optional[str] = None,
        hf_model_id: Optional[str] = None,
        hf_filename: str = "yolov8n-seg.pt",
        hf_cache_dir: Optional[str] = None,
    ) -> None:
        self.backend = backend
        self.model = None
        if backend == "yolo" and YOLO is not None:
            resolved_path = model_path
            if resolved_path is None and hf_model_id:
                if hf_hub_download is None:
                    raise RuntimeError("huggingface_hub is required for yolo_hf_model_id.")
                resolved_path = hf_hub_download(
                    repo_id=hf_model_id,
                    filename=hf_filename,
                    cache_dir=hf_cache_dir,
                )
            if resolved_path:
                self.model = YOLO(resolved_path)

    def segment(
        self,
        image: np.ndarray,
        clusters: int,
        min_area: int,
        max_shards: int,
        slic_compactness: float = 10.0,
        slic_sigma: float = 1.0,
    ) -> List[Shard]:
        if self.backend == "slic":
            if slic is None:
                raise RuntimeError("scikit-image is required for segmentation_backend='slic'.")
            shards = self._segment_slic(image, clusters, min_area, slic_compactness, slic_sigma)
        elif self.backend == "yolo" and self.model is not None:
            shards = self._segment_yolo(image, min_area)
        else:
            shards = self._segment_kmeans(image, clusters, min_area)
        shards = sorted(shards, key=lambda s: s.area, reverse=True)[:max_shards]
        return shards

    def _segment_yolo(self, image: np.ndarray, min_area: int) -> List[Shard]:
        results = self.model.predict(image, verbose=False)
        if not results:
            return []
        result = results[0]
        if result.masks is None:
            return []
        masks = result.masks.data.cpu().numpy()
        shards: List[Shard] = []
        for mask in masks:
            mask_u8 = (mask > 0.5).astype(np.uint8)
            shard = _mask_to_shard(image, mask_u8, min_area)
            if shard is not None:
                shards.append(shard)
        return shards

    def _segment_kmeans(self, image: np.ndarray, clusters: int, min_area: int) -> List[Shard]:
        h, w = image.shape[:2]
        scale = 1.0
        max_dim = 480
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
        if scale < 1.0:
            small = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            small = image

        data = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
        k = max(2, min(clusters, data.shape[0]))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, _ = cv2.kmeans(data, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        labels = labels.reshape(small.shape[:2])
        if scale < 1.0:
            labels = cv2.resize(labels.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            labels = labels.astype(np.uint8)

        shards: List[Shard] = []
        for cluster_id in range(k):
            cluster_mask = (labels == cluster_id).astype(np.uint8)
            if cluster_mask.sum() < min_area:
                continue
            num_labels, component_map = cv2.connectedComponents(cluster_mask)
            for comp_id in range(1, num_labels):
                comp_mask = (component_map == comp_id)
                if comp_mask.sum() < min_area:
                    continue
                shard = _mask_to_shard(image, comp_mask.astype(np.uint8), min_area)
                if shard is not None:
                    shards.append(shard)
        return shards

    def _segment_slic(
        self,
        image: np.ndarray,
        segments: int,
        min_area: int,
        compactness: float,
        sigma: float,
    ) -> List[Shard]:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb_float = rgb.astype(np.float32) / 255.0
        segments = max(2, segments)
        labels = slic(
            rgb_float,
            n_segments=segments,
            compactness=compactness,
            sigma=sigma,
            start_label=0,
            convert2lab=True,
            enforce_connectivity=True,
        )

        shards: List[Shard] = []
        for seg_id in np.unique(labels):
            seg_mask = (labels == seg_id)
            if seg_mask.sum() < min_area:
                continue
            shard = _mask_to_shard(image, seg_mask.astype(np.uint8), min_area)
            if shard is not None:
                shards.append(shard)
        return shards


def _mask_to_shard(image: np.ndarray, mask: np.ndarray, min_area: int) -> Optional[Shard]:
    if mask.sum() < min_area:
        return None
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        return None
    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    mask_crop = mask[y0:y1, x0:x1].astype(np.uint8)
    image_crop = image[y0:y1, x0:x1]

    alpha = (mask_crop * 255).astype(np.uint8)
    rgba = cv2.cvtColor(image_crop, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha

    area = int(mask.sum())
    centroid = (float(xs.mean()), float(ys.mean()))

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = rgb[mask > 0]
    if pixels.size == 0:
        mean_color = (128, 128, 128)
    else:
        mean_vals = pixels.mean(axis=0)
        mean_color = (int(mean_vals[0]), int(mean_vals[1]), int(mean_vals[2]))

    return Shard(
        rgba=rgba,
        bbox=(int(x0), int(y0), int(x1), int(y1)),
        area=area,
        centroid=centroid,
        mean_color=mean_color,
    )


def merge_shards(
    image: np.ndarray,
    shards: List[Shard],
    include_remainder: bool = False,
) -> Optional[Shard]:
    h, w = image.shape[:2]
    union_mask = np.zeros((h, w), dtype=np.uint8)
    for shard in shards:
        x0, y0, x1, y1 = shard.bbox
        alpha = shard.rgba[:, :, 3]
        union_mask[y0:y1, x0:x1] |= (alpha > 0).astype(np.uint8)

    if include_remainder:
        union_mask[:, :] = 1

    if union_mask.sum() == 0:
        return None
    return _mask_to_shard(image, union_mask, 0)
