# Rhythmic Mosaic Video Generator

A modular Python pipeline that transforms static images or video clips into audio-reactive motion graphics by rhythmically revealing semantic shards on a black void background. The output is a high-fidelity MP4 with the original audio muxed in.

## Highlights
- **Beat/measure time-map** with per-beat intensity scoring
- **Semantic shard generation** via segmentation (YOLOv8-seg optional) with K-means fallback
- **Rhythmic mosaic compositor** with sorting modes, bucket partitioning, and ghosting trails
- **Batch-ready** directory watcher and metadata output

## Requirements
- Python 3.10+
- `ffmpeg` in PATH

Install dependencies:
```bash
pip install -r requirements.txt
```

Optional:
- **YOLOv8 segmentation**: install `ultralytics` and set `RenderConfig.segmentation_backend="yolo"` with a local model path or Hugging Face model ID.
- **SLIC superpixels**: install `scikit-image` and set `RenderConfig.segmentation_backend="slic"` for organic, contiguous chunks.

## Project Structure
```
audio_visulizer/
  audio_vis/
    assets.py            # Image/video asset loading
    audio_features.py    # Beat/measure map + intensity
    compositor.py        # Shard sorting, rendering, ghosting
    config.py            # Render configuration
    pipeline.py          # Orchestration + muxing + metadata
    shards.py            # Shard extraction + segmentation engine
    video_io.py          # Video writer
    watcher.py           # Directory watcher workflow
    cli.py               # CLI entrypoint
  requirements.txt
  README.md
```

## Usage
### Render a single video
```bash
python -m audio_vis render --audio path/to/song.mp3 --video path/to/footage.mp4 --output output/mosaic.mp4
```

### Render from image directory
```bash
python -m audio_vis render --audio path/to/song.mp3 --images-dir path/to/images --output output/mosaic.mp4
```

### Watch a folder
This watches for `stem.mp3` plus either `stem.mp4` or a folder named `stem/`.
```bash
python -m audio_vis watch --input-dir input --output-dir output
```

### Output
For each render, the pipeline produces:
- `output.mp4` (final muxed video)
- `output.metadata.json` (BPM and measure count)

## Technical Specification Mapping
### Module I: Audio Analysis & Event Mapping
- **Transient analysis + BPM**: `librosa.beat.beat_track`
- **Beat timestamps**: beat frame times → absolute seconds
- **Measure grouping**: groups of 4 beats per measure
- **Intensity per beat**: normalized RMS + spectral centroid

### Module II: Visual Segmentation Engine
- **Mode A (Images)**: load a directory of images
- **Mode B (Video)**: sample a frame every N seconds
- **Segmentation**:
  - Optional YOLOv8-seg (if configured)
  - SLIC superpixels (scikit-image) for contiguous, organic chunks
  - Fallback K-means + connected components
- **Shard metadata**: area, centroid, mean RGB color

### Module III: The Glitch-Void Compositor
- **Sorting modes**: growth (area), sweep (centroid X), explosion (distance from center)
- **Partitioning**: shards split into beat buckets per measure
- **Render loop**:
  - Determine current beat
  - Select active bucket(s) (accumulative or stroboscopic)
  - Apply random offsets scaled by intensity
  - Composite on a black canvas

### Module IV: Post-Processing
- **Void decay (ghosting)**: `current + previous * decay`
- **Encoding**: H.264 MP4 muxed with original audio via `ffmpeg`

## Configuration
Default parameters live in `audio_vis/config.py`:
- FPS, resolution, sample rate for video frames
- Segmentation settings (backend, clusters, shard area)
- Mosaic behavior (accumulative vs strobe, ghost decay, glitch offset)
YOLOv8 (Hugging Face) example:
```python
from audio_vis.config import RenderConfig
config = RenderConfig(
    segmentation_backend="yolo",
    yolo_hf_model_id="Ultralytics/YOLOv8",
    yolo_hf_filename="yolov8n-seg.pt",
)
```

## Notes
- The K-means segmentation fallback is fully local and does not require GPU.
- For stronger instance segmentation, provide YOLOv8 weights locally and switch backends.

## Troubleshooting
- **ffmpeg not found**: Install and ensure it is in your PATH.
- **Slow rendering**: Lower resolution, reduce shard count, or increase `sample_every_sec`.
