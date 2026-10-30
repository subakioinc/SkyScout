"""Draw detection results on the original video frame."""

import cv2
import numpy as np

from .detector import Detection


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    for item in detections:
        x1, y1, x2, y2 = item.box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"{item.label} {item.score:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        text_top = max(0, y1 - text_height - baseline - 6)
        cv2.rectangle(frame, (x1, text_top), (min(frame.shape[1] - 1, x1 + text_width + 8), text_top + text_height + baseline + 6), (0, 0, 255), -1)
        cv2.putText(frame, label, (x1 + 4, text_top + text_height + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return frame
