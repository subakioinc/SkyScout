"""Export a separately obtained Ultralytics VisDrone checkpoint to ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a VisDrone YOLO checkpoint to ONNX")
    parser.add_argument("--weights", type=Path, required=True, help="local best.pt after checking its use conditions")
    parser.add_argument("--image-size", type=int, default=640)
    args = parser.parse_args()
    if not args.weights.is_file():
        parser.error(f"weights not found: {args.weights}")
    if args.image_size <= 0:
        parser.error("image-size must be positive")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install export-only dependency: python -m pip install ultralytics") from exc
    exported = YOLO(str(args.weights)).export(format="onnx", imgsz=args.image_size, nms=False, simplify=False)
    print(f"ONNX model: {exported}")


if __name__ == "__main__":
    main()
