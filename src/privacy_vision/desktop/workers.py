"""Tiến trình nền (Worker Threads) cho PrivacyVision Desktop.

Quản lý luồng webcam thời gian thực (WebcamWorker) và luồng xử lý video ngoại tuyến (VideoFileWorker)
trong các QThread riêng biệt để đảm bảo giao diện đồ họa chính không bị giật lag (Zero-freeze UI).
"""
from functools import partial
from pathlib import Path
from uuid import uuid4
import sys
from PySide6.QtCore import QThread, Signal

from .session import run_session
from .camera import CameraOptions, open_camera


class WebcamWorker(QThread):
    """Tiến trình nền nhận luồng hình ảnh từ camera và thực thi pipeline bảo vệ quyền riêng tư."""

    def __init__(self, config, camera, buffer, parent=None, camera_options=None, vcam_broadcaster=None):
        """Khởi tạo worker camera.

        Args:
            config: Cấu hình PipelineConfig.
            camera: Chỉ số cổng camera (int) hoặc chuỗi đường dẫn video/thiết bị.
            buffer: Bộ đệm PreviewBuffer để trao đổi khung hình với giao diện Qt.
            parent: Đối tượng cha Qt.
            camera_options: Cấu hình tham số phần cứng CameraOptions (FPS, độ phân giải, phơi sáng).
            vcam_broadcaster: Bộ phát webcam ảo VirtualCamBroadcaster (Meet/Zoom).
        """
        super().__init__(parent)
        self.session_id = uuid4().hex
        self.config = config
        self.camera = camera
        self.buffer = buffer
        self.camera_options = camera_options or CameraOptions()
        self.vcam_broadcaster = vcam_broadcaster

    def run(self):
        """Thực thi vòng lặp bắt khung hình và suy luận mô hình AI."""
        # Hỗ trợ monkeypatch linh hoạt trên window.run_session (để đảm bảo tương thích với các bài test)
        mod_win = sys.modules.get('privacy_vision.desktop.window')
        session_func = getattr(mod_win, 'run_session', run_session) if mod_win else run_session
        session_func(
            self.config,
            self.camera,
            self.buffer,
            self.isInterruptionRequested,
            capture_factory=partial(open_camera, options=self.camera_options),
            session_id=self.session_id,
            vcam_broadcaster=self.vcam_broadcaster
        )


class VideoFileWorker(QThread):
    """Tiến trình nền xử lý che mờ từng khung hình của file video ngoại tuyến theo lô."""

    # Tín hiệu phát tiến độ: (frame_hiện_tại, tổng_frame, fps_xử_lý, thời_gian_đã_chạy, frame_mẫu)
    progress = Signal(int, int, float, float, object)
    finished_signal = Signal(dict)
    error_signal = Signal(str)

    def __init__(self, input_path, output_path, pipeline, tracker=None, parent=None):
        """Khởi tạo worker xử lý file video.

        Args:
            input_path: Đường dẫn file video đầu vào.
            output_path: Đường dẫn file video đầu ra đã che.
            pipeline: Đối tượng PrivacyPipeline.
            tracker: Bộ bám vết RegionTracker (tùy chọn).
            parent: Đối tượng cha Qt.
        """
        super().__init__(parent)
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.pipeline = pipeline
        self.tracker = tracker
        self._interrupted = False

    def requestInterruption(self):
        """Yêu cầu dừng sớm tiến trình xử lý video."""
        self._interrupted = True
        super().requestInterruption()

    def run(self):
        """Thực hiện đọc từng khung hình, che mờ thông tin và ghi ra video mới."""
        try:
            from privacy_vision.cli.video import process_video

            def cb(current, total, fps, elapsed, frame=None):
                if self._interrupted or self.isInterruptionRequested():
                    return False
                self.progress.emit(current, total, fps, elapsed, frame)
                return True

            report = process_video(
                self.input_path,
                self.output_path,
                self.pipeline,
                tracker=self.tracker,
                progress_callback=cb
            )
            self.finished_signal.emit(report)
        except Exception as e:
            self.error_signal.emit(str(e))
