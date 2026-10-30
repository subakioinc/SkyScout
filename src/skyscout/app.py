"""SkyScout command-line application."""

from __future__ import annotations

import argparse
import logging
import signal
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .camera import VideoSource
from .detector import YoloDetector
from .overlay import draw_detections
from .stream import RtspStreamer

LOG = logging.getLogger(__name__)


class LatestFrameCapture:
    """Keep only the newest frame so slow inference does not accumulate lag."""

    def __init__(self, source: VideoSource, stop: threading.Event):
        self.source = source
        self.stop = stop
        self.condition = threading.Condition()
        self.frame: np.ndarray | None = None
        self.sequence = 0
        self.finished = False
        self.error: Exception | None = None
        self.thread = threading.Thread(target=self._capture, name="capture", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _capture(self) -> None:
        try:
            while not self.stop.is_set():
                started = time.monotonic()
                frame = self.source.read(self.stop)
                if frame is None:
                    break
                with self.condition:
                    self.frame = frame
                    self.sequence += 1
                    self.condition.notify_all()
                if self.source.is_file:
                    self.stop.wait(max(0.0, 1 / self.source.fps - (time.monotonic() - started)))
        except Exception as exc:
            self.error = exc
        finally:
            self.source.close()
            with self.condition:
                self.finished = True
                self.condition.notify_all()

    def next_frame(self, last_sequence: int) -> tuple[int, np.ndarray] | None:
        with self.condition:
            self.condition.wait_for(lambda: self.sequence > last_sequence or self.finished or self.stop.is_set())
            if self.error:
                raise self.error
            if self.sequence > last_sequence and self.frame is not None:
                return self.sequence, self.frame
            return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Onboard object detection with RTSP output")
    parser.add_argument("--model", type=Path, required=True, help="VisDrone YOLO11 ONNX model")
    parser.add_argument("--source", default="0", help="USB camera index (default: 0) or video file")
    parser.add_argument("--loop", action="store_true", help="loop a video file")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=15, help="camera request and RTSP output rate")
    parser.add_argument("--image-size", type=int, default=640, help="inference size for dynamic ONNX inputs")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--provider", default="CPUExecutionProvider", help="ONNX Runtime execution provider")
    parser.add_argument("--port", type=int, default=8554)
    parser.add_argument("--mount", default="/skyscout")
    parser.add_argument("--bitrate-kbps", type=int, default=2500)
    parser.add_argument("--encoder-pipeline", help="GStreamer H.264 encoder element and options; default: x264enc")
    args = parser.parse_args(argv)
    if not args.model.is_file():
        parser.error(f"model does not exist: {args.model}")
    if args.width <= 0 or args.height <= 0 or args.fps <= 0 or args.image_size <= 0:
        parser.error("width, height, fps and image-size must be positive")
    if args.width % 2 or args.height % 2:
        parser.error("H.264 output width and height must be even")
    if not 0 <= args.confidence <= 1 or not 0 <= args.iou <= 1:
        parser.error("confidence and iou must be between 0 and 1")
    if not args.mount.startswith("/"):
        parser.error("mount must begin with /")
    if not 1 <= args.port <= 65535 or args.bitrate_kbps <= 0:
        parser.error("port must be 1..65535 and bitrate-kbps must be positive")
    if args.loop and not Path(args.source).is_file():
        parser.error("--loop requires a video file source")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    source_id: int | str = int(args.source) if args.source.isdecimal() else args.source
    source = VideoSource(source_id, args.width, args.height, args.fps, args.loop)
    detector = YoloDetector(str(args.model), args.image_size, args.confidence, args.iou, args.provider)
    streamer = RtspStreamer(args.width, args.height, args.fps, args.port, args.mount,
                            args.bitrate_kbps, args.encoder_pipeline)
    stop = threading.Event()
    capture = LatestFrameCapture(source, stop)
    process_error: list[Exception] = []

    def process() -> None:
        last_sequence = 0
        processed = 0
        last_log = time.monotonic()
        try:
            while not stop.is_set():
                item = capture.next_frame(last_sequence)
                if item is None:
                    break
                last_sequence, frame = item
                if frame.shape[1] != args.width or frame.shape[0] != args.height:
                    frame = cv2.resize(frame, (args.width, args.height))
                detections = detector.detect(frame)
                streamer.push(draw_detections(frame.copy(), detections))
                processed += 1
                now = time.monotonic()
                if now - last_log >= 10:
                    LOG.info("processed %.1f fps; latest frame has %d detections", processed / (now - last_log), len(detections))
                    last_log, processed = now, 0
        except Exception as exc:
            process_error.append(exc)
            LOG.exception("processing failed")
        finally:
            stop.set()
            streamer.stop()

    def request_stop(_signum, _frame) -> None:
        stop.set()
        streamer.stop()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    capture.start()
    worker = threading.Thread(target=process, name="process", daemon=True)
    worker.start()
    try:
        streamer.run()
    finally:
        stop.set()
        worker.join(timeout=5)
        capture.thread.join(timeout=5)
    return 1 if process_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
