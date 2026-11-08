"""Bộ phát hiện khuôn mặt thời gian thực sử dụng YuNet OpenCV DNN."""
import cv2
from .base import BaseDetector
from privacy_vision.schemas import Detection


class YuNetDetector(BaseDetector):
    """Phát hiện khuôn mặt trên CPU với độ trễ thấp thông qua mô hình YuNet ONNX."""
    source = "face"

    def __init__(self, config):
        if not config.checkpoint.is_file():
            raise FileNotFoundError(config.checkpoint)
        self.size = config.imgsz
        self.detector = cv2.FaceDetectorYN.create(
            str(config.checkpoint), "", (640, 640),
            config.confidence, 0.3, 5000,
            cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU,
        )

    def detect(self, image):
        """Thực hiện suy luận nhận diện khuôn mặt trên khung hình."""
        h, w = image.shape[:2]
        scale = min(1.0, self.size / max(h, w))
        dw, dh = max(1, round(w * scale)), max(1, round(h * scale))
        small = cv2.resize(image, (dw, dh))
        self.detector.setInputSize((dw, dh))
        _, rows = self.detector.detect(small)
        output = []
        if rows is not None:
            for row in rows:
                x, y, bw, bh = map(float, row[:4])
                output.append(
                    Detection(
                        "face", 0, "face", float(row[-1]),
                        (x * w / dw, y * h / dw if h == 0 else y * h / dh,
                         (x + bw) * w / dw, (y + bh) * h / dh),
                    )
                )
        return output
