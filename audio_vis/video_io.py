from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np


def _resample_frames(frames: List[np.ndarray], src_fps: float, target_fps: int) -> List[np.ndarray]:
    if not frames:
        return []
    if abs(src_fps - target_fps) < 0.01:
        return frames
    duration = len(frames) / src_fps
    target_count = int(round(duration * target_fps))
    src_times = np.linspace(0.0, duration, num=len(frames), endpoint=False)
    tgt_times = np.linspace(0.0, duration, num=target_count, endpoint=False)
    indices = np.searchsorted(src_times, tgt_times, side="right") - 1
    indices = np.clip(indices, 0, len(frames) - 1)
    return [frames[i] for i in indices]


def _resize_frames(frames: List[np.ndarray], size: Tuple[int, int]) -> List[np.ndarray]:
    width, height = size
    resized = []
    for frame in frames:
        resized.append(cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA))
    return resized


def _boomerang_sequence(frames: List[np.ndarray]) -> List[np.ndarray]:
    if len(frames) < 2:
        return frames
    return frames + frames[-2:0:-1]


def load_video_frames(video_path: str, target_fps: int, size: Tuple[int, int]) -> Tuple[List[np.ndarray], float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    frames = _resample_frames(frames, src_fps, target_fps)
    frames = _resize_frames(frames, size)
    duration = len(frames) / target_fps if target_fps else 0.0
    return frames, duration


def extend_frames_to_duration(frames: List[np.ndarray], target_frames: int) -> List[np.ndarray]:
    if not frames:
        return []
    if len(frames) >= target_frames:
        return frames[:target_frames]

    extended: List[np.ndarray] = []
    sequence = _boomerang_sequence(frames)
    while len(extended) < target_frames:
        extended.extend(sequence)
    return extended[:target_frames]


def create_video_writer(output_path: str, fps: int, size: Tuple[int, int]) -> cv2.VideoWriter:
    width, height = size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Unable to create video writer: {output_path}")
    return writer
