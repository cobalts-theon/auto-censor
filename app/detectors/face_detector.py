"""YuNet detection with coordinates mapped to the original frame."""
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
import cv2


@dataclass(frozen=True)
class Face:
    box: tuple[int, int, int, int]
    confidence: float
    source: str = "Detected"


class FaceDetector:
    def __init__(self, model_path, detect_width=640, threshold=0.6):
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"Không tìm thấy model: {model_path}")
        self.detect_width = detect_width
        self.detector = cv2.FaceDetectorYN.create(
            str(model_path), "", (640, 360), threshold, 0.3, 5000
        )

    def detect(self, frame):
        h, w = frame.shape[:2]
        dw = min(w, self.detect_width)
        dh = max(1, round(h * dw / w))
        small = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_AREA)
        self.detector.setInputSize((dw, dh))
        started = perf_counter()
        _, detections = self.detector.detect(small)
        inference_ms = (perf_counter() - started) * 1000
        faces = []
        if detections is not None:
            for detection in detections:
                x, y, fw, fh = detection[:4]
                x1, y1 = max(0, int(x*w/dw)), max(0, int(y*h/dh))
                x2, y2 = min(w, int((x+fw)*w/dw)), min(h, int((y+fh)*h/dh))
                if x2 > x1 and y2 > y1:
                    px, py = int((x2-x1)*0.2), int((y2-y1)*0.2)
                    faces.append(Face((max(0, x1-px), max(0, y1-py),
                                       min(w, x2+px), min(h, y2+py)),
                                      float(detection[-1])))
        return faces, inference_ms
