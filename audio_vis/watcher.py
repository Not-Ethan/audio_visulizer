from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from .config import RenderConfig
from .pipeline import run_pipeline


@dataclass
class InputPair:
    audio: Path
    video: Optional[Path] = None
    images_dir: Optional[Path] = None


def _find_pairs(input_dir: Path) -> Dict[str, InputPair]:
    audio_files = {p.stem: p for p in input_dir.glob("*.mp3")}
    video_files = {p.stem: p for p in input_dir.glob("*.mp4")}
    image_dirs = {p.name: p for p in input_dir.iterdir() if p.is_dir()}
    pairs: Dict[str, InputPair] = {}
    for stem, audio in audio_files.items():
        video = video_files.get(stem)
        images_dir = image_dirs.get(stem)
        if video or images_dir:
            pairs[stem] = InputPair(
                audio=audio,
                video=video,
                images_dir=images_dir,
            )
    return pairs


class PairHandler(FileSystemEventHandler):
    def __init__(self, input_dir: Path, output_dir: Path, config: RenderConfig) -> None:
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.config = config
        self.processed: Dict[str, float] = {}

    def on_any_event(self, event) -> None:
        pairs = _find_pairs(self.input_dir)
        for stem, pair in pairs.items():
            if stem in self.processed:
                continue
            output_path = self.output_dir / f"{stem}.mp4"
            run_pipeline(
                audio_path=str(pair.audio),
                output_path=str(output_path),
                config=self.config,
                images_dir=str(pair.images_dir) if pair.images_dir else None,
                video_path=str(pair.video) if pair.video else None,
            )
            self.processed[stem] = time.time()


def watch_directory(input_dir: str, output_dir: str, config: Optional[RenderConfig] = None) -> None:
    config = config or RenderConfig()
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    handler = PairHandler(input_path, output_path, config)
    observer = Observer()
    observer.schedule(handler, str(input_path), recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
