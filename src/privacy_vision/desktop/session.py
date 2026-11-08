"""Phiên xử lý Webcam độc lập không phụ thuộc Qt; chỉ đưa frame đã xử lý ra ngoài."""
from dataclasses import dataclass, field
from threading import Lock
from time import perf_counter
import numpy as np
import logging
import math
from uuid import uuid4
from privacy_vision.pipeline import PrivacyPipeline, validate_output
from .capture import LatestCapture, ERROR_STATUS
from privacy_vision.processing.tracking import RegionTracker


@dataclass(frozen=True)
class Preview:
    """Dữ liệu khung hình và thống kê hiệu năng hiển thị trên giao diện người dùng."""
    image: np.ndarray
    fps: float
    processing_ms: float
    timings_ms: dict[str, float]
    device: str
    capture_fps: float = 0.0
    capture_wait_ms: float = 0.0
    frame_age_ms: float = 0.0
    camera_info: str = ''
    document_fallback: bool = False
    detection_counts: dict[str, int] = field(default_factory=dict)
    coverage_status: str = 'coverage_unverified'
    coverage_reasons: tuple[str, ...] = ()
    strict_blackout: bool = False
    published_at: float = 0.0
    session_id: str = ''
    enabled_detectors: tuple[str, ...] = ()
    detections: tuple = ()
    vcam_active: bool = False
    vcam_device: str = ''


class PreviewBuffer:
    """Hộp thư đơn vị (single-slot buffer); tránh tình trạng dồn hàng đợi sự kiện của Qt."""
    def __init__(self):
        self._lock = Lock()
        self._preview = None
        self._status = 'Đang chờ'

    def publish(self, preview):
        """Đưa khung hình mới nhất vào vùng đệm."""
        with self._lock:
            self._preview = preview
            self._status = 'Đang xử lý'

    def clear(self, status):
        """Xóa khung hình hiện tại và cập nhật trạng thái."""
        with self._lock:
            self._preview = None
            self._status = status

    def take(self):
        """Lấy khung hình mới nhất ra khỏi bộ đệm."""
        with self._lock:
            preview, self._preview = self._preview, None
            return preview, self._status


def open_capture(index):
    """Mở thiết bị ghi hình (camera)."""
    from .camera import open_camera, CameraOptions
    return open_camera(index, CameraOptions())


def device_label(pipeline):
    """Mô tả thiết bị tính toán đang được pipeline sử dụng (CPU/CUDA)."""
    if pipeline.config.document_detection_strategy == 'heuristic' and all(c.source == 'face' for c in pipeline.config.detectors):
        return 'CPU (YuNet; không chạy YOLO GPU)'
    device = getattr(pipeline, 'device', 'cpu')
    if device == 'cpu':
        return 'CPU (CUDA không dùng)'
    import torch
    index = int(device)
    return f'CUDA {index}: {torch.cuda.get_device_name(index)}'


def run_session(config, camera, buffer, stopped, pipeline_factory=PrivacyPipeline,
                capture_factory=open_capture, describe_device=device_label, session_id=None,
                vcam_broadcaster=None):
    """Vòng lặp xử lý chính của camera thời gian thực."""
    capture = None
    session_id = session_id or uuid4().hex
    stage = 'load'
    try:
        buffer.clear('Đang tải model')
        if not config.detectors:
            raise ValueError('Cần ít nhất một detector cho webcam')
        pipeline = pipeline_factory(config)  # Khởi tạo mô hình nằm trong tiến trình nền
        tracker = RegionTracker(hold_seconds=0.6, document_hold_seconds=0.6)
        device = describe_device(pipeline)
        if stopped():
            return
        capture = LatestCapture(camera, capture_factory, buffer)
        capture.start()
        frames, tick = 0, perf_counter()
        fps = 0.0
        while not stopped():
            wait_started = perf_counter()
            stage = 'capture'
            captured = capture.take(stopped)
            capture_wait_ms = (perf_counter() - wait_started) * 1000
            if captured is None:
                break
            stage = 'pipeline'
            result = pipeline.process(captured.image, tracker=tracker)
            stage = 'validate'
            validate_output(result.image, captured.image.shape)
            if np.shares_memory(result.image, captured.image):
                raise ValueError('Pipeline trả về bộ nhớ gốc chưa sao chép')
            if 'total' not in result.timings_ms or any(not math.isfinite(v) or v < 0 for v in result.timings_ms.values()):
                raise ValueError('Thời gian thực thi pipeline không hợp lệ')
            if stopped():
                break
            frames += 1
            now = perf_counter()
            if now - tick >= 1:
                fps, frames, tick = frames / (now - tick), 0, now
            stage = 'publish'
            if vcam_broadcaster is not None:
                vcam_broadcaster.broadcast_frame(result.image, fps=int(captured.fps) if captured.fps > 0 else 30)

            capture.publish(
                Preview(
                    result.image.copy(),
                    fps,
                    result.timings_ms['total'],
                    dict(result.timings_ms),
                    device,
                    captured.fps,
                    capture_wait_ms,
                    (now - captured.read_at) * 1000,
                    captured.camera_info,
                    any(d.class_name == 'document_boundary_unresolved' for d in getattr(result, 'detections', [])),
                    {
                        source: sum(
                            (d.source == source or (source == 'document' and d.class_name == 'document_region'))
                            and d.metadata.get('tracking_state', 'detected') == 'detected'
                            for d in getattr(result, 'detections', [])
                        )
                        for source in ('face', 'plate', 'card', 'cccd', 'document')
                    },
                    result.coverage_status,
                    tuple(result.coverage_reasons),
                    result.strict_blackout,
                    now,
                    session_id,
                    tuple(c.source for c in config.detectors)
                    + (('document',) if config.document_detection_strategy != 'heuristic' else ()),
                    tuple(getattr(result, 'detections', [])),
                    vcam_active=vcam_broadcaster.is_active if vcam_broadcaster else False,
                    vcam_device=vcam_broadcaster.device_name if vcam_broadcaster else '',
                )
            )
    except Exception as exc:
        logging.getLogger(__name__).error('webcam_%s_failure:%s', stage, type(exc).__name__)
        labels = {
            'load': 'tải model',
            'capture': 'camera',
            'pipeline': 'AI/redaction',
            'validate': 'kiểm tra kết quả',
            'publish': 'xuất preview',
        }
        buffer.clear(
            f'Lỗi {labels[stage]} [{type(exc).__name__}] — đã chặn hiển thị. '
            'Dừng rồi kiểm tra cấu hình/camera/model và bắt đầu lại.'
        )
    finally:
        if vcam_broadcaster is not None:
            vcam_broadcaster.stop()
        if capture is not None:
            capture.stop()
        # Không bao giờ để lại frame cũ sau khi dừng hoặc lỗi
        _, status = buffer.take()
        buffer.clear(status if status.startswith('Lỗi') else 'Đã dừng')
