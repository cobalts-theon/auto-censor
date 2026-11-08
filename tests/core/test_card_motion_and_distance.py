"""Kiểm thử tính năng nhận diện thẻ phẳng/đơn sắc ở khoảng cách xa (Card distance & flat background safety)."""
import numpy as np
import pytest
from pathlib import Path
from privacy_vision.schemas import Detection
from privacy_vision.pipeline import PrivacyPipeline
from privacy_vision.config import PipelineConfig, DetectorConfig


def test_card_document_never_rejected_by_flat_background_filter(tmp_path):
    """Xác nhận rằng thẻ phẳng/đơn sắc (card_document) không bao giờ bị loại bỏ bởi bộ lọc nền phẳng."""
    # Tạo một khung hình giả lập thẻ ATM đơn sắc (ví dụ thẻ đen mờ hoặc thẻ xanh phẳng)
    frame = np.full((480, 640, 3), 30, dtype=np.uint8)

    class DummyFaceDetector:
        source = 'face'
        def detect(self, img):
            return []

    class DummyCardDetector:
        source = 'document'
        def detect(self, img):
            # Giả lập phát hiện thẻ ngân hàng ở xa (80x50 pixels)
            return [Detection('document', 0, 'card_document', 0.85, (200, 150, 280, 200))]

    dummy_doc_ckpt = tmp_path / "doc.pt"
    dummy_doc_ckpt.touch()
    dummy_face_ckpt = tmp_path / "face.onnx"
    dummy_face_ckpt.touch()

    pipeline = PrivacyPipeline(
        PipelineConfig(
            [DetectorConfig('face', dummy_face_ckpt)],
            whole_documents=True,
            document_detection_strategy='detector',
            document_detector_config=DetectorConfig('document', dummy_doc_ckpt),
        ),
        detectors=[DummyFaceDetector()],
        document_detector=DummyCardDetector(),
    )

    result = pipeline.process(frame)
    # Thẻ phải được giữ lại và tạo thành document_region được che chắn
    doc_regions = [d for d in result.detections if d.class_name == 'document_region']
    assert len(doc_regions) == 1
    assert doc_regions[0].bbox == (200, 150, 280, 200)
