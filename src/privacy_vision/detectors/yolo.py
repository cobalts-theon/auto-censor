"""Bộ nhận diện đối tượng sử dụng kiến trúc Ultralytics YOLO.

Hỗ trợ suy luận phát hiện biển số xe, thẻ căn cước, tài liệu và các đối tượng nhạy cảm
với khả năng tự động tối ưu hóa phần cứng (CUDA / CPU) và bán chính xác (FP16/half precision).
"""
from pathlib import Path
from .base import BaseDetector
from privacy_vision.schemas import Detection


def resolve_device(requested: str) -> str:
    """Xác định và kiểm tra thiết bị tính toán khả dụng (CUDA hoặc CPU).

    Args:
        requested: Chuỗi yêu cầu thiết bị ('auto', 'cpu', hoặc chỉ số GPU như '0', '1').

    Returns:
        Tên thiết bị hợp lệ ('cpu' hoặc chỉ số GPU dạng chuỗi).
    """
    import torch
    if requested == "cpu":
        return "cpu"
    if requested != "auto" and not requested.isdecimal():
        raise ValueError("device phải là 'auto', 'cpu' hoặc chỉ số CUDA hợp lệ")
    if not torch.cuda.is_available():
        return "cpu"
    index = 0 if requested == "auto" else int(requested)
    if index >= torch.cuda.device_count():
        raise ValueError("Chỉ số thiết bị CUDA vượt quá số lượng GPU khả dụng")
    return str(index)


class YoloDetector(BaseDetector):
    """Lớp điều phối mô hình YOLO thực hiện suy luận phát hiện đối tượng trên khung hình."""

    def __init__(self, config, device: str):
        """Khởi tạo detector từ file checkpoint trọng số mô hình.

        Args:
            config: Cấu hình detector (DetectorConfig) chứa đường dẫn checkpoint, confidence, imgsz.
            device: Chuỗi thiết bị tính toán ('cpu' hoặc chỉ số CUDA).
        """
        if not config.checkpoint.is_file():
            raise FileNotFoundError(f"Không tìm thấy file checkpoint mô hình: {config.checkpoint}")
        from ultralytics import YOLO
        self.source, self.config, self.device = config.source, config, device
        self.model = YOLO(str(config.checkpoint), task="detect")

    def detect(self, image):
        """Thực hiện suy luận trên ảnh và chuyển đổi kết quả sang danh sách Detection chuẩn.

        Args:
            image: Khung hình BGR (numpy.ndarray).

        Returns:
            Danh sách đối tượng Detection chứa tọa độ bounding box đã chuẩn hóa.
        """
        # Ultralytics tự động chuyển đổi boxes.xyxy về đúng hệ tọa độ gốc của khung hình ndarray.
        kwargs = {"quantize": 16} if self.device != "cpu" else {}
        results = self.model.predict(
            source=image,
            conf=self.config.confidence,
            imgsz=self.config.imgsz,
            device=self.device,
            half=self.device != "cpu",  # Sử dụng FP16 khi chạy trên GPU CUDA để tăng tốc độ
            batch=1,
            verbose=False,
            save=False,
            save_txt=False,
            save_crop=False,
            show=False,
            **kwargs
        )
        output = []
        for result in results:
            if result.boxes is None:
                continue
            for row in result.boxes.data.cpu().tolist():
                x1, y1, x2, y2, confidence, cls = row[:6]
                cls_name = result.names[int(cls)]
                if self.config.classes is not None and cls_name not in self.config.classes:
                    continue
                output.append(Detection(
                    source=self.source,
                    class_id=int(cls),
                    class_name=cls_name,
                    confidence=float(confidence),
                    bbox=(x1, y1, x2, y2)
                ))
        return output
