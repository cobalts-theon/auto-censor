"""Pipeline bảo vệ quyền riêng tư thời gian thực (PrivacyVision Pipeline).

Kết hợp các mô hình nhận diện khuôn mặt, biển số xe, tài liệu/thẻ,
áp dụng bộ lọc làm mờ (blur), che đen (blackout), hoặc làm mờ điểm ảnh (pixelate).
"""
from dataclasses import replace
from time import perf_counter
import math
import cv2
import numpy as np
from .config import PipelineConfig
from .schemas import PipelineResult, DocumentRegion
from .processing.boxes import normalize_box, unrotate_box
from .processing.redaction import redact
from .image_io import read_image, write_image


def _is_flat_background(roi: np.ndarray) -> bool:
    """Kiểm tra vùng ROI tài liệu có phải mảng tường trống, bảng trắng hoặc cửa không.

    Tránh việc mô hình nhận diện nhầm nền trống là tài liệu giấy lớn (paper_document).
    """
    h, w = roi.shape[:2]
    # Thẻ nhỏ ở xa (dưới 50x50) không bao giờ bị coi là bức tường
    if h < 50 or w < 50:
        return False
    # Loại bỏ viền ngoài (10%) để tránh viền bóng đổ của tường/bàn ghế
    mx, my = max(2, int(w * 0.10)), max(2, int(h * 0.10))
    inner = roi[my:h - my, mx:w - mx]
    if inner.size == 0 or np.all(inner == inner[0, 0]):
        return False

    # Phân tích nhanh trên ảnh thu nhỏ (120x90 chia đều cho lưới 3x3: 40x30)
    small = cv2.resize(inner, (120, 90))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 1. Đo lường biên cạnh Canny và độ biến thiên Sobel toàn cục
    edges = cv2.Canny(blurred, 30, 85)
    edge_density = float(np.count_nonzero(edges) / edges.size)
    gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)
    grad_mean = float(np.mean(grad_mag))
    lap_var = float(cv2.Laplacian(blurred, cv2.CV_64F).var())

    # Trường hợp tường trơn: hầu như không có cạnh hoặc biến thiên độ sáng
    if edge_density < 0.010 and grad_mean < 14.0 and lap_var < 10.0:
        return True

    # 2. Kiểm tra phân bố không gian trên lưới 3x3
    # Phát hiện trường hợp đối tượng nằm ở 1 góc và phần còn lại là tường trống > 50%
    flat_cells = 0
    cell_w, cell_h = 40, 30
    for r in range(3):
        for c in range(3):
            c_edges = edges[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w]
            c_edge_ratio = np.count_nonzero(c_edges) / c_edges.size
            c_grad = float(np.mean(grad_mag[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w]))
            c_lap = float(cv2.Laplacian(blurred[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w], cv2.CV_64F).var())
            if c_edge_ratio < 0.010 and c_grad < 14.0 and c_lap < 10.0:
                flat_cells += 1

    return flat_cells >= 5


class PrivacyPipeline:
    """Đường ống suy luận và làm mờ các đối tượng nhạy cảm."""
    def __init__(self, config: PipelineConfig | None = None, detectors=None, document_detector=None):
        self.config = config or PipelineConfig()
        if detectors is None:
            from .detectors.yunet import YuNetDetector
            from .detectors.yolo import YoloDetector, resolve_device
            self.device = resolve_device(self.config.device)
            detectors = [
                YuNetDetector(c) if c.source == "face" else YoloDetector(c, self.device)
                for c in self.config.detectors
            ]
        self.detectors = list(detectors)
        self.document_detector = document_detector
        if self.config.document_detection_strategy != 'heuristic' and document_detector is None:
            from .detectors.document import DocumentDetector
            from .detectors.yolo import resolve_device
            self.device = resolve_device(self.config.device)
            self.document_detector = DocumentDetector(self.config.document_detector_config, self.device)
        if [d.source for d in self.detectors] != [c.source for c in self.config.detectors]:
            raise ValueError("Mô hình được truyền vào phải khớp với cấu hình")

    def process(self, image: np.ndarray, tracker=None) -> PipelineResult:
        """Xử lý một khung hình: phát hiện đối tượng, theo dõi liên khung hình và làm mờ."""
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
            raise ValueError("Ảnh đầu vào phải là mảng numpy BGR uint8 hợp lệ")
        start = perf_counter()
        detections, timings = [], {}
        h, w = image.shape[:2]
        active_sources = {c.source: c for c in self.config.detectors}
        pairs = [(d, active_sources[d.source]) for d in self.detectors if d.source in active_sources]
        if self.config.document_detection_strategy != 'heuristic' and self.document_detector and self.config.document_detector_config:
            pairs.append((self.document_detector, self.config.document_detector_config))

        for detector, config in pairs:
            tick = perf_counter()
            try:
                rows = []
                angles = self.config.document_rotations if detector.source in ('card', 'cccd', 'document') else (0,)
                for angle in angles:
                    rotated = image if angle == 0 else np.ascontiguousarray(np.rot90(image, angle // 90))
                    found_angle = False
                    for d in detector.detect(rotated):
                        if len(d.bbox) != 4 or not all(math.isfinite(v) for v in d.bbox) or d.bbox[2] <= d.bbox[0] or d.bbox[3] <= d.bbox[1]:
                            raise ValueError('Tọa độ khung bao từ detector không hợp lệ')
                        rows.append(replace(d, bbox=unrotate_box(d.bbox, w, h, angle),
                                            metadata={**d.metadata, 'rotation_degrees': angle}))
                        if d.confidence >= config.confidence:
                            found_angle = True
                    # Xoay thích ứng: khi đã phát hiện ở góc hiện tại, dừng xoay tiếp để duy trì 30+ FPS
                    if detector.source == 'document' and found_angle:
                        break

                for d in rows:
                    if d.source != detector.source or not math.isfinite(d.confidence) or not 0 <= d.confidence <= 1:
                        raise ValueError("Kết quả từ detector không hợp lệ")
                    if detector.source == 'document':
                        from .detectors.document import DOCUMENT_CLASSES
                        if d.class_name not in DOCUMENT_CLASSES:
                            raise ValueError('Lớp tài liệu không được hỗ trợ')
                    box = normalize_box(d.bbox, w, h, 0 if detector.source == 'document' else self.config.margin)
                    if box and d.confidence >= config.confidence:
                        # Chỉ kiểm tra nền phẳng đối với paper_document chiếm diện tích lớn
                        # KHÔNG BAO GIỜ loại bỏ card_document (thẻ ngân hàng, CCCD, GPLX)
                        if detector.source == 'document' and d.class_name == 'paper_document':
                            x1, y1, x2, y2 = box
                            doc_area = (x2 - x1) * (y2 - y1)
                            if doc_area > 0.12 * w * h:
                                roi = image[y1:y2, x1:x2]
                                if _is_flat_background(roi):
                                    continue
                        detections.append(replace(d, bbox=box))
            except Exception as exc:
                raise RuntimeError(f"Detector {detector.source} gặp sự cố; không xuất ảnh") from exc
            timings[detector.source] = (perf_counter() - tick) * 1000

        timings["inference"] = (perf_counter() - start) * 1000
        tick = perf_counter()
        current = list(detections)
        fields = [d for d in current if d.source in ('card', 'cccd')]
        documents = [d for d in current if d.source == 'document']
        detections = [d for d in current if d.source != 'document']
        regions = []

        def contains(box, field):
            a = field.bbox
            return box[0] <= a[0] and box[1] <= a[1] and box[2] >= a[2] and box[3] >= a[3]

        for d in documents:
            evidence = [f for f in fields if contains(d.bbox, f)]
            source = 'hybrid' if evidence and self.config.document_detection_strategy == 'hybrid' else 'detector'
            regions.append(DocumentRegion(d.bbox, d.confidence, d.class_name, source, len(evidence), evidence))

        if self.config.whole_documents and self.config.document_detection_strategy in ('heuristic', 'hybrid'):
            from .processing.documents import document_regions
            remaining = [f for f in fields if not any(contains(d.bbox, f) for d in documents)]
            proposals = document_regions(image, remaining, self.config.document_margin, self.config.document_min_fields)
            for d in proposals:
                if d.class_name == 'document_region':
                    evidence = d.metadata['supporting_detections']
                    hints = {f.class_name for f in evidence} & {'cccd', 'bhyt', 'gplx'}
                    kind = next(iter(hints)) if len(hints) == 1 else None
                    regions.append(DocumentRegion(d.bbox, None, kind, 'heuristic', d.metadata['field_count'], evidence))

        region_detections = []
        if self.config.whole_documents:
            from .schemas import Detection
            for r in regions:
                region_detections.append(
                    Detection(
                        'document' if r.source != 'heuristic' else 'card',
                        -1,
                        'document_region',
                        r.confidence if r.confidence is not None else 0.0,
                        r.bbox,
                        {
                            'derived_region': True,
                            'region_source': r.source,
                            'confidence_kind': 'unscored_heuristic' if r.confidence is None else 'model_score',
                        },
                    )
                )
            detections.extend(region_detections)

        timings['document_regions'] = (perf_counter() - tick) * 1000
        tick = perf_counter()

        if tracker is not None:
            # Chạy theo dõi trên ảnh gốc trước khi thực hiện làm mờ
            tracked, untracked = [], []
            for d in detections:
                field = d.source in ('card', 'cccd') and d.class_name != 'document_region'
                (untracked if field else tracked).append(d)
            detections = tracker.update(image, tracked, perf_counter()) + untracked
            timings['tracking'] = (perf_counter() - tick) * 1000
            tick = perf_counter()

        reasons = ['scene_not_verified']
        risk = []
        if not current:
            risk.append('no_current_detections')
        if any(not any(contains(d.bbox, f) for d in documents) for f in fields):
            risk.append('fields_without_detector_region')
        if any(r.source == 'heuristic' for r in regions):
            risk.append('heuristic_boundary')
        if self.config.whole_documents and not documents:
            risk.append('no_detector_document_region')
        if any(d.metadata.get('tracking_state', 'detected') != 'detected' for d in detections):
            risk.append('tracking_unconfirmed')

        blackout = self.config.strict_privacy_mode and bool(risk)
        output = np.zeros_like(image) if blackout else redact(image, detections, self.config.mode, self.config.pixel_size)
        validate_output(output, image.shape)
        timings["redaction"] = (perf_counter() - tick) * 1000
        timings["total"] = (perf_counter() - start) * 1000

        return PipelineResult(output, detections, timings, regions, "coverage_unverified", reasons + risk, blackout)

    def process_file(self, input_path, output_path):
        """Xử lý một tệp ảnh từ đĩa và ghi kết quả ra tệp mới."""
        from pathlib import Path
        if Path(input_path).resolve() == Path(output_path).resolve():
            raise ValueError("Đường dẫn đầu vào và đầu ra không được trùng nhau")
        result = self.process(read_image(input_path))
        write_image(output_path, result.image)
        return result


def validate_output(output, shape):
    """Kiểm tra tính hợp lệ của mảng hình ảnh đầu ra."""
    if not isinstance(output, np.ndarray) or output.dtype != np.uint8 or output.shape != shape:
        raise ValueError('Kết quả làm mờ không hợp lệ')
