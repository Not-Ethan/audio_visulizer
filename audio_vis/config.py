from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class RenderConfig:
    fps: int = 30
    width: int = 1920
    height: int = 1080
    sample_every_sec: float = 2.0
    segmentation_backend: str = "yolo"
    segmentation_clusters: int = 6
    yolo_model_path: Optional[str] = None
    yolo_hf_model_id: Optional[str] = None
    yolo_hf_filename: str = "yolov8n-seg.pt"
    yolo_hf_cache_dir: Optional[str] = None
    segmentation_slic_compactness: float = 10.0
    segmentation_slic_sigma: float = 1.0
    min_shard_area: int = 1200
    max_shards: int = 80
    ghost_decay: float = 0.9
    glitch_max_offset: int = 6
    accumulative: bool = True
    output_bitrate: str = "16M"
