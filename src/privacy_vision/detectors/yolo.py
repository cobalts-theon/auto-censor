from .base import BaseDetector
from privacy_vision.schemas import Detection

def resolve_device(requested: str) -> str:
    import torch
    if requested == "cpu":
        return "cpu"
    if requested != "auto" and not requested.isdecimal():
        raise ValueError("device must be auto, cpu, or a CUDA index")
    if not torch.cuda.is_available():
        return "cpu"
    index = 0 if requested == "auto" else int(requested)
    if index >= torch.cuda.device_count():
        raise ValueError("CUDA device index unavailable")
    return str(index)

class YoloDetector(BaseDetector):
    def __init__(self, config, device):
        if not config.checkpoint.is_file():
            raise FileNotFoundError(config.checkpoint)
        from ultralytics import YOLO
        self.source, self.config, self.device = config.source, config, device
        self.model = YOLO(str(config.checkpoint), task="detect")

    def detect(self, image):
        # Ultralytics maps boxes.xyxy back to the original ndarray coordinates.
        kwargs = {"quantize": 16} if self.device != "cpu" else {}
        results = self.model.predict(source=image, conf=self.config.confidence,
            imgsz=self.config.imgsz, device=self.device,
            batch=1, verbose=False, save=False, save_txt=False, save_crop=False, show=False, **kwargs)
        output = []
        for result in results:
            if result.boxes is None:
                continue
            for row in result.boxes.data.cpu().tolist():
                x1, y1, x2, y2, confidence, cls = row[:6]
                cls_name = result.names[int(cls)]
                if self.config.classes is not None and cls_name not in self.config.classes:
                    continue
                output.append(Detection(self.source, int(cls), cls_name,
                    float(confidence), (x1, y1, x2, y2)))
        return output
