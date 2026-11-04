import unittest
from pathlib import Path
import cv2
import numpy as np
from examples.face_webcam.app.processing.anonymizer import Mode, anonymize
from examples.face_webcam.app.detectors.face_detector import Face, FaceDetector


class PipelineTests(unittest.TestCase):
    def test_modes_and_outside_region(self):
        frame = np.random.default_rng(42).integers(0, 256, (80, 100, 3), dtype=np.uint8)
        face = Face((20, 20, 60, 60), 0.95)
        for mode in Mode:
            with self.subTest(mode=mode):
                result = anonymize(frame.copy(), [face], mode)
                np.testing.assert_array_equal(result[:20], frame[:20])
                np.testing.assert_array_equal(result[:, :20], frame[:, :20])
                if mode == Mode.DETECT_ONLY:
                    np.testing.assert_array_equal(result, frame)
                elif mode == Mode.BLACKOUT:
                    self.assertFalse(result[20:60, 20:60].any())
                else:
                    self.assertFalse(np.array_equal(result[20:60, 20:60], frame[20:60, 20:60]))

    def test_empty_and_tiny_regions(self):
        for mode in Mode:
            frame = np.full((10, 10, 3), 100, dtype=np.uint8)
            np.testing.assert_array_equal(anonymize(frame.copy(), [], mode), frame)
            anonymize(frame, [Face((0, 0, 1, 1), 0.9)], mode)

    def test_real_detector_blank_frame(self):
        detector = FaceDetector(Path(__file__).resolve().parents[2] / 'models/face_detection_yunet_2026may.onnx')
        faces, milliseconds = detector.detect(np.zeros((720, 1280, 3), dtype=np.uint8))
        self.assertEqual(faces, [])
        self.assertGreaterEqual(milliseconds, 0)


if __name__ == '__main__':
    unittest.main()
