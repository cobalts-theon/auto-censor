"""Kiểm thử tính năng bám vết thẻ khi di chuyển nhanh và ở khoảng cách xa (Card fast motion & distance tracking)."""
import numpy as np
import pytest
from pathlib import Path
from privacy_vision.schemas import Detection
from privacy_vision.processing.tracking import RegionTracker
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


def test_fast_moving_card_tracking_association():
    """Kiểm tra việc bám vết thẻ di chuyển nhanh với bước nhảy tọa độ lớn."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    tracker = RegionTracker(hold_seconds=0.8, document_hold_seconds=1.2)

    # Frame 0: Thẻ ở vị trí (100, 100, 200, 160)
    d0 = Detection('document', 0, 'document_region', 0.9, (100, 100, 200, 160))
    res0 = tracker.update(frame, [d0], 0.0)
    assert len(res0) == 1

    # Frame 1: Thẻ di chuyển nhanh sang (180, 120, 280, 180) sau 0.05s (vượt ngưỡng IoU thông thường)
    d1 = Detection('document', 0, 'document_region', 0.9, (180, 120, 280, 180))
    res1 = tracker.update(frame, [d1], 0.05)
    assert len(res1) == 1
    assert res1[0].metadata['tracking_state'] == 'detected'

    # Frame 2: Không có phát hiện từ model (ví dụ bị chói sáng 1 frame), tracking phải giữ ở trạng thái 'held'
    res2 = tracker.update(frame, [], 0.10)
    assert len(res2) == 1
    assert res2[0].metadata['tracking_state'] == 'held'

    # Frame 3: Sau 0.5s không có model, với document_hold_seconds=1.2s, vệt vẫn được bảo vệ
    res3 = tracker.update(frame, [], 0.60)
    assert len(res3) == 1
    assert res3[0].metadata['tracking_state'] == 'held'

    # Frame 4: Sau 1.5s quá hạn (1.5 - 0.05 = 1.45s > 1.2s), vệt mới hết hạn giải phóng
    res4 = tracker.update(frame, [], 1.50)
    assert len(res4) == 0
