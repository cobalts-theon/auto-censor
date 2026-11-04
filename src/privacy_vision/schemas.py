from dataclasses import dataclass, field, asdict
from typing import Any
import numpy as np

@dataclass(frozen=True)
class Detection:
    source: str
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class DocumentRegion:
    bbox: tuple[int, int, int, int]
    confidence: float | None
    document_type: str | None
    source: str
    field_count: int = 0
    supporting_detections: list[Detection] = field(default_factory=list)

@dataclass
class PipelineResult:
    image: np.ndarray
    detections: list[Detection]
    timings_ms: dict[str, float]

    document_regions: list[DocumentRegion] = field(default_factory=list)
    coverage_status: str = "coverage_unverified"
    coverage_reasons: list[str] = field(default_factory=list)
    strict_blackout: bool = False

    def to_metadata(self) -> dict:
        return {"document_regions": [asdict(r) for r in self.document_regions],
                "coverage_status": self.coverage_status, "coverage_reasons": self.coverage_reasons,
                "strict_blackout": self.strict_blackout, "detections": [asdict(d) for d in self.detections],
                "timings_ms": self.timings_ms, "image_size": list(self.image.shape[:2]),
                "warning": "Detection may miss sensitive information; review before sharing."}
