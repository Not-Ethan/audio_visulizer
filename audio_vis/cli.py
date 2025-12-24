from __future__ import annotations

import argparse
from pathlib import Path

from .config import RenderConfig
from .pipeline import run_pipeline
from .watcher import watch_directory


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rhythmic mosaic video generator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    render_parser = subparsers.add_parser("render", help="Render a single audio + image/video input")
    render_parser.add_argument("--audio", required=True, help="Path to input audio (mp3)")
    render_parser.add_argument("--video", help="Path to input video (mp4) for frame sampling")
    render_parser.add_argument("--images-dir", help="Path to input image directory")
    render_parser.add_argument("--output", required=True, help="Path to output mp4")

    watch_parser = subparsers.add_parser("watch", help="Watch a directory for mp3 + media pairs")
    watch_parser.add_argument("--input-dir", required=True, help="Directory with input pairs")
    watch_parser.add_argument("--output-dir", required=True, help="Directory for rendered videos")

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = RenderConfig()

    if args.command == "render":
        if not args.video and not args.images_dir:
            raise SystemExit("Provide --video or --images-dir for rendering.")
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        run_pipeline(
            audio_path=args.audio,
            output_path=str(output_path),
            config=config,
            images_dir=args.images_dir,
            video_path=args.video,
        )
    elif args.command == "watch":
        watch_directory(args.input_dir, args.output_dir, config)


if __name__ == "__main__":
    main()
