"""Lớp cơ sở trừu tượng cho tất cả các bộ phát hiện (BaseDetector)."""
from abc import ABC, abstractmethod
import numpy as np
from privacy_vision.schemas import Detection


class BaseDetector(ABC):
    """Giao diện chuẩn cho các mô hình AI nhận diện đối tượng."""
    source: str

    @abstractmethod
    def detect(self, image: np.ndarray) -> list[Detection]:
        """Trả về danh sách phát hiện trên hệ tọa độ ảnh BGR gốc. Ngoại lệ được truyền ra ngoài."""
