from dataclasses import dataclass, field
from pathlib import Path
import math

MODEL_FILES = {"face": "face_detection_yunet_2026may.onnx", "plate": "plate_best.pt",
               "card": "vietnamese_card_best.pt", "cccd": "cccd_information_best.pt"}

@dataclass(frozen=True)
class DetectorConfig:
    source: str
    checkpoint: Path
    confidence: float = 0.25
    imgsz: int = 640
    classes: tuple[str, ...] | None = None

    def __post_init__(self):
        if self.source not in (*MODEL_FILES, "document") or not 0 <= self.confidence <= 1:
            raise ValueError("Invalid source/confidence")
        if self.imgsz < 32 or self.imgsz % 32:
            raise ValueError("imgsz must be a positive multiple of 32")

@dataclass
class PipelineConfig:
    detectors: list[DetectorConfig] = field(default_factory=lambda: [
        DetectorConfig(s, Path("models") / p, 0.65 if s == "face" else 0.25)
        for s, p in MODEL_FILES.items()])
    device: str = "auto"
    mode: str = "black"
    margin: float = 0.1
    document_rotations: tuple[int, ...] = (0,)
    whole_documents: bool = False

    document_redaction_mode: str | None = None
    document_detection_strategy: str = "heuristic"
    document_margin: tuple[float, float] = (0.25, 0.30)
    document_min_fields: int = 3
    strict_privacy_mode: bool = False
    document_detector_config: DetectorConfig | None = None
    max_document_area_ratio: float = 0.45

    def __post_init__(self):
        if self.document_redaction_mode is None:
            self.document_redaction_mode = "full" if self.whole_documents else "fields"
        if self.document_redaction_mode not in ("fields", "full"):
            raise ValueError("Invalid document redaction mode")
        if self.whole_documents and self.document_redaction_mode != "full":
            raise ValueError("whole_documents conflicts with document_redaction_mode")
        self.whole_documents = self.document_redaction_mode == "full"
        if self.document_detection_strategy not in ("heuristic", "detector", "hybrid"):
            raise ValueError("Invalid document detection strategy")
        if len(self.document_margin) != 2 or any(not math.isfinite(v) or not 0 <= v <= .5 for v in self.document_margin):
            raise ValueError("document_margin must contain two ratios in [0, .5]")
        if not isinstance(self.document_min_fields, int) or self.document_min_fields < 2:
            raise ValueError("document_min_fields must be >= 2")
        if any(d.source == "document" for d in self.detectors):
            raise ValueError("Use document_detector_config for document detector")
        if self.document_detector_config is not None and self.document_detector_config.source != "document":
            raise ValueError("Document detector source must be document")
        if self.document_detection_strategy != "heuristic":
            if self.document_detector_config is None or not self.document_detector_config.checkpoint.is_file():
                raise ValueError("Document detector strategy requires a local checkpoint")
        if not self.document_rotations or len(set(self.document_rotations)) != len(self.document_rotations) or any(a not in (0, 90, 180, 270) for a in self.document_rotations):
            raise ValueError("document_rotations must contain unique angles from 0,90,180,270")
        if self.mode not in ("black", "blur", "pixelate"):
            raise ValueError("Invalid redaction mode")
        if not math.isfinite(self.margin) or self.margin < 0:
            raise ValueError("margin must be finite and nonnegative")
        sources = [d.source for d in self.detectors]
        if len(sources) != len(set(sources)):
            raise ValueError("Duplicate detector source")
