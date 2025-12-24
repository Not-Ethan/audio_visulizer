from __future__ import annotations

from dataclasses import dataclass
from typing import List

import librosa
import numpy as np


@dataclass
class BeatInfo:
    index: int
    start: float
    end: float
    intensity: float


@dataclass
class MeasureInfo:
    index: int
    start: float
    end: float
    beats: List[BeatInfo]


@dataclass
class AudioMap:
    bpm: float
    duration: float
    beats: List[BeatInfo]
    measures: List[MeasureInfo]


def _normalize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    min_val = float(values.min())
    max_val = float(values.max())
    if max_val - min_val < 1e-9:
        return np.zeros_like(values)
    return (values - min_val) / (max_val - min_val)


def _build_intensity(
    rms: np.ndarray,
    centroid: np.ndarray,
    times: np.ndarray,
    beat_times: np.ndarray,
    duration: float,
) -> List[float]:
    rms_n = _normalize(rms)
    centroid_n = _normalize(centroid)

    beat_intensity = []
    for idx, beat_time in enumerate(beat_times):
        end_time = beat_times[idx + 1] if idx + 1 < len(beat_times) else duration
        mask = (times >= beat_time) & (times < end_time)
        if not mask.any():
            intensity = 0.0
        else:
            rms_val = float(rms_n[mask].mean())
            hf_val = float(centroid_n[mask].mean())
            intensity = 0.7 * rms_val + 0.3 * hf_val
        beat_intensity.append(float(intensity))
    return beat_intensity


def build_audio_map(audio_path: str, hop_length: int = 512) -> AudioMap:
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    duration = float(librosa.get_duration(y=y, sr=sr))

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    bpm, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop_length)

    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
    if beat_times.size == 0:
        beat_times = np.arange(0.0, duration, 0.5)

    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    feature_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    intensities = _build_intensity(rms, centroid, feature_times, beat_times, duration)

    beats: List[BeatInfo] = []
    for idx, beat_time in enumerate(beat_times):
        end_time = beat_times[idx + 1] if idx + 1 < len(beat_times) else duration
        beats.append(BeatInfo(index=idx, start=float(beat_time), end=float(end_time), intensity=intensities[idx]))

    measures: List[MeasureInfo] = []
    for measure_idx in range(0, len(beats), 4):
        beat_slice = beats[measure_idx : measure_idx + 4]
        if not beat_slice:
            continue
        start = beat_slice[0].start
        end = beat_slice[-1].end
        measures.append(MeasureInfo(index=len(measures), start=start, end=end, beats=beat_slice))

    return AudioMap(bpm=float(bpm), duration=duration, beats=beats, measures=measures)
