from __future__ import annotations

from typing import Iterable, List, Tuple

import cv2
import numpy as np

from .shards import Shard


def sort_shards(shards: List[Shard], mode: str, center: Tuple[float, float]) -> List[Shard]:
    if mode == "growth":
        return sorted(shards, key=lambda s: s.area)
    if mode == "sweep":
        return sorted(shards, key=lambda s: s.centroid[0])
    if mode == "explosion":
        return sorted(shards, key=lambda s: _distance(s.centroid, center))
    return shards


def partition_shards(shards: List[Shard], buckets: int) -> List[List[Shard]]:
    if buckets <= 0:
        return [shards]
    if not shards:
        return [[] for _ in range(buckets)]
    splits = np.array_split(np.arange(len(shards)), buckets)
    return [[shards[i] for i in split] for split in splits]


def render_shards(
    canvas: np.ndarray,
    shards: Iterable[Shard],
    rng: np.random.Generator,
    max_offset: int,
) -> None:
    for shard in shards:
        dx = int(rng.integers(-max_offset, max_offset + 1)) if max_offset > 0 else 0
        dy = int(rng.integers(-max_offset, max_offset + 1)) if max_offset > 0 else 0
        _blit_rgba(canvas, shard, dx, dy)


def apply_ghosting(current: np.ndarray, previous: np.ndarray, decay: float) -> np.ndarray:
    output = current.astype(np.float32) + previous.astype(np.float32) * decay
    return np.clip(output, 0, 255).astype(np.uint8)


def _blit_rgba(canvas: np.ndarray, shard: Shard, dx: int, dy: int) -> None:
    x0, y0, x1, y1 = shard.bbox
    x0 += dx
    x1 += dx
    y0 += dy
    y1 += dy

    h, w = canvas.shape[:2]
    if x1 <= 0 or y1 <= 0 or x0 >= w or y0 >= h:
        return

    src = shard.rgba
    src_h, src_w = src.shape[:2]

    x0c = max(0, x0)
    y0c = max(0, y0)
    x1c = min(w, x1)
    y1c = min(h, y1)

    sx0 = x0c - x0
    sy0 = y0c - y0
    sx1 = sx0 + (x1c - x0c)
    sy1 = sy0 + (y1c - y0c)

    if sx1 <= sx0 or sy1 <= sy0:
        return

    src_crop = src[sy0:sy1, sx0:sx1]
    alpha = src_crop[:, :, 3:4].astype(np.float32) / 255.0
    fg = src_crop[:, :, :3].astype(np.float32)
    bg = canvas[y0c:y1c, x0c:x1c].astype(np.float32)

    blended = fg * alpha + bg * (1.0 - alpha)
    canvas[y0c:y1c, x0c:x1c] = blended.astype(np.uint8)


def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
