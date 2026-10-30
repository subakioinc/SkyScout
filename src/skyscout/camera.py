"""USB camera and video-file capture with reconnect support."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

LOG = logging.getLogger(__name__)


class VideoSource:
    def __init__(self, source: int | str, width: int, height: int, fps: int, loop: bool = False):
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.loop = loop
        self.is_file = isinstance(source, str) and Path(source).is_file()
        self.capture: cv2.VideoCapture | None = None

    def open(self) -> bool:
        self.close()
        backend = cv2.CAP_V4L2 if isinstance(self.source, int) and hasattr(cv2, "CAP_V4L2") else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.source, backend)
        if not cap.isOpened():
            cap.release()
            return False
        if not self.is_file:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            LOG.info("camera opened: %.0fx%.0f at %.1f fps (reported)", cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FPS))
        self.capture = cap
        return True

    def read(self, stop_event) -> np.ndarray | None:
        while not stop_event.is_set():
            if self.capture is None and not self.open():
                if self.is_file:
                    raise RuntimeError(f"cannot open video file: {self.source}")
                LOG.warning("camera unavailable; retrying in 2 seconds")
                stop_event.wait(2)
                continue
            assert self.capture is not None
            ok, frame = self.capture.read()
            if ok and frame is not None:
                return frame
            if self.is_file and not self.loop:
                return None
            LOG.warning("video source ended or disconnected; reopening")
            self.close()
            stop_event.wait(0.2 if self.is_file else 2)
        return None

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
