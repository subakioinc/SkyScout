import unittest

import numpy as np

from skyscout.detector import decode_yolo, letterbox
from skyscout.overlay import draw_detections


class DetectionTests(unittest.TestCase):
    def test_letterbox_and_decode_rescale_to_original_frame(self):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        image, scale, pad = letterbox(frame, (640, 640))
        self.assertEqual(image.shape, (640, 640, 3))
        self.assertEqual(pad, (0, 140))
        # Raw YOLO center/size in 640px model coordinates for a 300x300 box.
        predictions = np.zeros((1, 15, 2), dtype=np.float32)
        predictions[0, :4, 0] = [150, 240, 100, 100]
        predictions[0, 0, 1] = 400
        predictions[0, 1, 1] = 400
        predictions[0, 2, 1] = 50
        predictions[0, 3, 1] = 50
        predictions[0, 4 + 0, 0] = 0.9
        predictions[0, 4 + 2, 1] = 0.95  # bicycle: not in current scope
        detections = decode_yolo(predictions, frame.shape[:2], scale, pad, 0.25, 0.45)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].box, (300, 150, 600, 450))
        self.assertEqual(detections[0].label, "person")
        result = draw_detections(frame.copy(), detections)
        self.assertTrue(np.array_equal(result[150, 300], [0, 0, 255]))

    def test_same_class_overlap_is_suppressed(self):
        predictions = np.zeros((1, 15, 2), dtype=np.float32)
        predictions[0, :4, :] = np.array([[50, 51], [50, 51], [30, 30], [30, 30]], dtype=np.float32)
        predictions[0, 4 + 3, :] = [0.9, 0.8]
        detections = decode_yolo(predictions, (100, 100), 1.0, (0, 0), 0.25, 0.45)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].label, "vehicle")

    def test_unsupported_output_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "VisDrone classes"):
            decode_yolo(np.zeros((1, 84, 8400), dtype=np.float32), (100, 100), 1.0, (0, 0), 0.25, 0.45)


if __name__ == "__main__":
    unittest.main()
