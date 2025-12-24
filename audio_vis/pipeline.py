from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from tqdm import tqdm

from .assets import load_frames_from_video, load_images_from_dir
from .audio_features import AudioMap, BeatInfo, MeasureInfo, build_audio_map
from .compositor import apply_ghosting, partition_shards, render_shards, sort_shards
from .config import RenderConfig
from .shards import SegmentationEngine, Shard, merge_shards
from .video_io import create_video_writer


@dataclass
class RenderMetadata:
    bpm: float
    total_measures: int
    total_beats: int


def _mux_audio(video_path: Path, audio_path: Path, output_path: Path, bitrate: str) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:v",
        bitrate,
        "-shortest",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _write_metadata(output_path: Path, metadata: RenderMetadata) -> None:
    metadata_path = output_path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(asdict(metadata), indent=2))


def _select_sort_mode(measure_index: int) -> str:
    modes = ["growth", "sweep", "explosion"]
    return modes[measure_index % len(modes)]


def _measure_for_beat(measures: List[MeasureInfo], beat_index: int) -> MeasureInfo:
    for measure in measures:
        if measure.beats[0].index <= beat_index <= measure.beats[-1].index:
            return measure
    return measures[-1]


def run_pipeline(
    audio_path: str,
    output_path: str,
    config: Optional[RenderConfig] = None,
    images_dir: Optional[str] = None,
    video_path: Optional[str] = None,
) -> Path:
    config = config or RenderConfig()
    if not images_dir and not video_path:
        raise RuntimeError("Provide either images_dir or video_path.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    audio_map = build_audio_map(audio_path)
    frame_count = int(audio_map.duration * config.fps)

    size = (config.width, config.height)
    if images_dir:
        assets = load_images_from_dir(images_dir, size)
    else:
        assets = load_frames_from_video(video_path, size, config.sample_every_sec)

    segmenter = SegmentationEngine(
        config.segmentation_backend,
        config.yolo_model_path,
        config.yolo_hf_model_id,
        config.yolo_hf_filename,
        config.yolo_hf_cache_dir,
    )
    shard_sets: List[List[Shard]] = []
    for asset in assets:
        shards = segmenter.segment(
            asset,
            clusters=config.segmentation_clusters,
            min_area=config.min_shard_area,
            max_shards=config.max_shards,
            slic_compactness=config.segmentation_slic_compactness,
            slic_sigma=config.segmentation_slic_sigma,
        )
        shard_sets.append(shards)

    temp_video_path = output_path.with_suffix(".silent.mp4")
    writer = create_video_writer(str(temp_video_path), config.fps, size)

    prev_frame = np.zeros((config.height, config.width, 3), dtype=np.uint8)
    rng = np.random.default_rng()

    current_measure_index = -1
    buckets: List[Optional[Shard]] = []
    current_asset = assets[0]

    for frame_idx in tqdm(range(frame_count), desc="Rendering", unit="frame"):
        timestamp = frame_idx / config.fps
        beat_index = _current_beat_index(audio_map.beats, timestamp)
        beat = audio_map.beats[beat_index]
        measure = _measure_for_beat(audio_map.measures, beat_index)

        if measure.index != current_measure_index:
            current_measure_index = measure.index
            asset_index = measure.index % len(shard_sets)
            shards = shard_sets[asset_index]
            current_asset = assets[asset_index]
            center = (config.width / 2.0, config.height / 2.0)
            sort_mode = _select_sort_mode(measure.index)
            sorted_shards = sort_shards(shards, sort_mode, center)
            shard_buckets = partition_shards(sorted_shards, len(measure.beats))
            include_remainder = config.segmentation_backend == "yolo"
            buckets = []
            for idx, bucket in enumerate(shard_buckets):
                buckets.append(
                    merge_shards(
                        current_asset,
                        bucket,
                        include_remainder=include_remainder and idx == len(shard_buckets) - 1,
                    )
                )

        bucket_index = beat.index - measure.beats[0].index
        shard = buckets[bucket_index] if bucket_index < len(buckets) else None
        active_shards = [shard] if shard is not None else []

        canvas = np.zeros((config.height, config.width, 3), dtype=np.uint8)
        max_offset = int(config.glitch_max_offset * beat.intensity)
        render_shards(canvas, active_shards, rng, max_offset)
        frame = apply_ghosting(canvas, prev_frame, config.ghost_decay)
        prev_frame = frame

        writer.write(frame)

    writer.release()
    _mux_audio(temp_video_path, Path(audio_path), output_path, config.output_bitrate)
    temp_video_path.unlink(missing_ok=True)

    metadata = RenderMetadata(
        bpm=audio_map.bpm,
        total_measures=len(audio_map.measures),
        total_beats=len(audio_map.beats),
    )
    _write_metadata(output_path, metadata)
    return output_path


def _current_beat_index(beats: List[BeatInfo], timestamp: float) -> int:
    if not beats:
        return 0
    times = [beat.start for beat in beats]
    idx = int(np.searchsorted(times, timestamp, side="right") - 1)
    return int(np.clip(idx, 0, len(beats) - 1))
