"""Ultralytics YOLO detection ONNX inference and postprocessing."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort


VISDRONE_NAMES = (
    "pedestrian", "people", "bicycle", "car", "van", "truck",
    "tricycle", "awning-tricycle", "bus", "motor", "others",
)
TARGET_LABELS = {0: "person", 1: "person", 3: "vehicle", 4: "vehicle", 5: "vehicle", 8: "vehicle"}


@dataclass(frozen=True)
class Detection:
    box: tuple[int, int, int, int]
    score: float
    label: str


def letterbox(frame: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize into (height, width) with padding; return image, scale, left/top pad."""
    height, width = frame.shape[:2]
    target_h, target_w = size
    scale = min(target_w / width, target_h / height)
    resized_w, resized_h = round(width * scale), round(height * scale)
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    left = (target_w - resized_w) // 2
    top = (target_h - resized_h) // 2
    canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    canvas[top:top + resized_h, left:left + resized_w] = resized
    return canvas, scale, (left, top)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    order = np.argsort(scores)[::-1]
    keep: list[int] = []
    while order.size:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        x1 = np.maximum(boxes[current, 0], boxes[rest, 0])
        y1 = np.maximum(boxes[current, 1], boxes[rest, 1])
        x2 = np.minimum(boxes[current, 2], boxes[rest, 2])
        y2 = np.minimum(boxes[current, 3], boxes[rest, 3])
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        areas = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
        union = areas[current] + areas[rest] - intersection
        iou = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=np.float32), where=union > 0)
        order = rest[iou <= iou_threshold]
    return keep


class YoloDetector:
    def __init__(self, model_path: str, image_size: int = 640, confidence: float = 0.25, iou: float = 0.45, provider: str = "CPUExecutionProvider"):
        available = ort.get_available_providers()
        if provider not in available:
            raise RuntimeError(f"ONNX provider {provider} unavailable; available: {available}")
        self.session = ort.InferenceSession(model_path, providers=[provider])
        model_input = self.session.get_inputs()[0]
        if len(model_input.shape) != 4 or model_input.shape[1] not in (3, "3"):
            raise ValueError(f"expected NCHW RGB model input, got {model_input.shape}")
        input_h, input_w = model_input.shape[2:]
        self.input_size = (input_h if isinstance(input_h, int) else image_size,
                           input_w if isinstance(input_w, int) else image_size)
        self.input_name = model_input.name
        self.confidence = confidence
        self.iou = iou

    def detect(self, frame: np.ndarray) -> list[Detection]:
        image, scale, (left, top) = letterbox(frame, self.input_size)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        output = self.session.run(None, {self.input_name: tensor})[0]
        return decode_yolo(output, frame.shape[:2], scale, (left, top), self.confidence, self.iou)


def decode_yolo(output: np.ndarray, frame_size: tuple[int, int], scale: float,
                pad: tuple[int, int], confidence: float, iou: float) -> list[Detection]:
    """Decode raw YOLOv8/11 ONNX output [1, 4+C, N] (or [1, N, 4+C])."""
    if output.ndim != 3 or output.shape[0] != 1:
        raise ValueError(f"unsupported YOLO output shape: {output.shape}")
    data = output[0]
    expected = 4 + len(VISDRONE_NAMES)
    if data.shape[0] == expected:
        data = data.T
    elif data.shape[1] != expected:
        raise ValueError(f"expected {len(VISDRONE_NAMES)} VisDrone classes and raw YOLO output; got {output.shape}")
    classes = np.argmax(data[:, 4:], axis=1)
    scores = np.max(data[:, 4:], axis=1)
    selected = np.array([int(c) in TARGET_LABELS for c in classes]) & (scores >= confidence)
    if not selected.any():
        return []
    rows, classes, scores = data[selected], classes[selected], scores[selected]
    centers = rows[:, :4]
    boxes = np.column_stack((centers[:, 0] - centers[:, 2] / 2,
                             centers[:, 1] - centers[:, 3] / 2,
                             centers[:, 0] + centers[:, 2] / 2,
                             centers[:, 1] + centers[:, 3] / 2))
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad[0]) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad[1]) / scale
    height, width = frame_size
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height - 1)
    results: list[Detection] = []
    for cls in np.unique(classes):
        indices = np.flatnonzero(classes == cls)
        for kept in nms(boxes[indices], scores[indices], iou):
            idx = indices[kept]
            x1, y1, x2, y2 = boxes[idx].round().astype(int)
            if x2 > x1 and y2 > y1:
                results.append(Detection((x1, y1, x2, y2), float(scores[idx]), TARGET_LABELS[int(cls)]))
    return sorted(results, key=lambda detection: detection.score, reverse=True)
