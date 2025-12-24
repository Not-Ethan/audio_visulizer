from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def load_images_from_dir(images_dir: str, size: Tuple[int, int]) -> List[np.ndarray]:
    path = Path(images_dir)
    images: List[np.ndarray] = []
    for file_path in sorted(path.iterdir()):
        if file_path.suffix.lower() not in IMAGE_EXTS:
            continue
        image = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        images.append(image)
    if not images:
        raise RuntimeError(f"No images found in {images_dir}")
    return images


def load_frames_from_video(video_path: str, size: Tuple[int, int], sample_every_sec: float) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    interval = max(1, int(round(sample_every_sec * fps)))

    frames: List[np.ndarray] = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            frames.append(frame)
        idx += 1
    cap.release()

    if not frames:
        raise RuntimeError(f"No frames sampled from {video_path}")
    return frames
