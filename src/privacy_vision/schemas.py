"""Cấu trúc dữ liệu và lược đồ (Schemas) của hệ thống PrivacyVision.

Bao gồm Detection (phát hiện thô), DocumentRegion (vùng tài liệu), và PipelineResult (kết quả xử lý).
"""
from dataclasses import dataclass, field, asdict
from typing import Any
import numpy as np


@dataclass(frozen=True)
class Detection:
    """Đối tượng phát hiện bởi mô hình AI hoặc thuật toán suy luận."""
    source: str
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DocumentRegion:
    """Vùng bao tài liệu/thẻ được xác định qua detector hoặc thuật toán gom cụm."""
    bbox: tuple[int, int, int, int]
    confidence: float | None
    document_type: str | None
    source: str
    field_count: int = 0
    supporting_detections: list[Detection] = field(default_factory=list)


@dataclass
class PipelineResult:
    """Kết quả tổng thể sau khi đưa khung hình qua đường ống bảo vệ quyền riêng tư."""
    image: np.ndarray
    detections: list[Detection]
    timings_ms: dict[str, float]

    document_regions: list[DocumentRegion] = field(default_factory=list)
    coverage_status: str = "coverage_unverified"
    coverage_reasons: list[str] = field(default_factory=list)
    strict_blackout: bool = False

    def to_metadata(self) -> dict:
        """Chuyển đổi thông tin kết quả sang định dạng từ điển để xuất tệp JSON metadata."""
        return {
            "document_regions": [asdict(r) for r in self.document_regions],
            "coverage_status": self.coverage_status,
            "coverage_reasons": self.coverage_reasons,
            "strict_blackout": self.strict_blackout,
            "detections": [asdict(d) for d in self.detections],
            "timings_ms": self.timings_ms,
            "image_size": list(self.image.shape[:2]),
            "warning": "Hệ thống AI có thể bỏ sót thông tin nhạy cảm; vui lòng kiểm tra trước khi chia sẻ.",
        }
