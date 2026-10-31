"""Short optical-flow fallback; detections remain the source of confidence."""
from dataclasses import dataclass
from math import ceil, floor
import cv2
import numpy as np
from app.detectors.face_detector import Face


@dataclass
class _Track:
    face: Face
    last_seen: float
    points: object = None


def _iou(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union else 0.0


class FaceTracker:
    def __init__(self, hold_seconds=0.6):
        self.hold_seconds = hold_seconds
        self.tracks = []
        self.previous = None

    def update(self, frame, detections, now):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        if self.previous is not None and self.previous.shape != gray.shape:
            self.tracks = []
            self.previous = None
        self.tracks = [t for t in self.tracks if now-t.last_seen <= self.hold_seconds]
        for track in self.tracks:
            track.face = Face(track.face.box, track.face.confidence, "Held")
            if self.previous is not None and track.points is not None:
                moved, status, _ = cv2.calcOpticalFlowPyrLK(self.previous, gray, track.points, None)
                if moved is not None:
                    valid = status.ravel() == 1
                    backward, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, self.previous, moved, None)
                    if backward is not None:
                        valid &= back_status.ravel() == 1
                        valid &= np.linalg.norm((backward-track.points).reshape(-1, 2), axis=1) < 1.5
                        if valid.sum() >= 4:
                            dx, dy = np.median((moved-track.points).reshape(-1, 2)[valid], axis=0)
                            x1, y1, x2, y2 = track.face.box
                            track.face = Face((int(x1+dx), int(y1+dy), int(x2+dx), int(y2+dy)),
                                              track.face.confidence, "Tracked")

        # Greedy global IoU matching assigns each detection and track at most once.
        pairs = sorted(((_iou(t.face.box, d.box), ti, di)
                        for ti, t in enumerate(self.tracks) for di, d in enumerate(detections)), reverse=True)
        used_tracks, used_detections = set(), set()
        for score, ti, di in pairs:
            if score < 0.2 or ti in used_tracks or di in used_detections:
                continue
            track, detection = self.tracks[ti], detections[di]
            # Smooth small jitter, but include the entire current detection so the
            # filtered box cannot lag behind a fast-moving face.
            smooth = [0.25*a+0.75*b for a, b in zip(track.face.box, detection.box)]
            box = (floor(min(smooth[0], detection.box[0])), floor(min(smooth[1], detection.box[1])),
                   ceil(max(smooth[2], detection.box[2])), ceil(max(smooth[3], detection.box[3])))
            track.face = Face(box, detection.confidence)
            track.last_seen = now
            used_tracks.add(ti)
            used_detections.add(di)
        for di, detection in enumerate(detections):
            if di not in used_detections:
                self.tracks.append(_Track(detection, now))
        result = []
        for track in self.tracks:
            x1, y1, x2, y2 = track.face.box
            box = (max(0, x1), max(0, y1), min(w, x2), min(h, y2))
            if box[2] <= box[0] or box[3] <= box[1]:
                track.points = None
                continue
            track.face = Face(box, track.face.confidence, track.face.source)
            mask = np.zeros_like(gray)
            mask[box[1]:box[3], box[0]:box[2]] = 255
            track.points = cv2.goodFeaturesToTrack(gray, maxCorners=40, qualityLevel=0.01,
                                                   minDistance=5, mask=mask)
            result.append(track.face)
        self.previous = gray
        return result
