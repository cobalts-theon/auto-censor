import unittest
import cv2
import numpy as np
from app.detectors.face_detector import Face
from app.detectors.face_tracker import FaceTracker


class TrackerTests(unittest.TestCase):
    def test_hold_expire_and_reacquire(self):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        tracker = FaceTracker()
        face = Face((20, 20, 70, 70), 0.9)
        tracker.update(frame, [face], 0)
        held = tracker.update(frame, [], 0.2)
        self.assertEqual(held[0].source, 'Held')
        self.assertEqual(held[0].confidence, 0.9)
        reacquired = tracker.update(frame, [Face((25, 20, 75, 70), 0.8)], 0.3)
        self.assertEqual(len(reacquired), 1)
        self.assertEqual(reacquired[0].source, 'Detected')
        self.assertLessEqual(reacquired[0].box[0], 25)
        self.assertGreaterEqual(reacquired[0].box[2], 75)
        self.assertEqual(tracker.update(frame, [], 1.0), [])

    def test_flow_follows_translation(self):
        frame = np.zeros((160, 200, 3), dtype=np.uint8)
        frame[40:100, 40:100] = np.random.default_rng(4).integers(0, 256, (60, 60, 3), dtype=np.uint8)
        tracker = FaceTracker()
        tracker.update(frame, [Face((35, 35, 105, 105), 0.9)], 0)
        moved = cv2.warpAffine(frame, np.float32([[1, 0, 5], [0, 1, 3]]), (200, 160))
        tracked = tracker.update(moved, [], 0.1)[0]
        self.assertEqual(tracked.source, 'Tracked')
        self.assertAlmostEqual(tracked.box[0], 40, delta=1)
        self.assertAlmostEqual(tracked.box[1], 38, delta=1)

    def test_multiple_faces_remain_separate(self):
        frame = np.zeros((120, 200, 3), dtype=np.uint8)
        tracker = FaceTracker()
        faces = [Face((10, 10, 50, 50), 0.9), Face((130, 10, 170, 50), 0.8)]
        tracker.update(frame, faces, 0)
        self.assertEqual(len(tracker.update(frame, list(reversed(faces)), 0.1)), 2)
