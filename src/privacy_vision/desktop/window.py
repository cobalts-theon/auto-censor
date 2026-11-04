from pathlib import Path
import numpy as np
from PySide6.QtCore import Qt, QThread, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QPushButton, QCheckBox, QComboBox, QSpinBox, QDoubleSpinBox, QFileDialog)
from privacy_vision.config import PipelineConfig, DetectorConfig, MODEL_FILES
from .session import PreviewBuffer, run_session
from .camera import CameraOptions, open_camera
from functools import partial
from time import perf_counter
from uuid import uuid4


class WebcamWorker(QThread):
    def __init__(self, config, camera, buffer, parent=None, camera_options=None):
        super().__init__(parent)
        self.session_id = uuid4().hex
        self.config, self.camera, self.buffer = config, camera, buffer
        self.camera_options = camera_options or CameraOptions()

    def run(self):
        run_session(self.config, self.camera, self.buffer, self.isInterruptionRequested,
                    capture_factory=partial(open_camera, options=self.camera_options), session_id=self.session_id)


class WebcamWindow(QMainWindow):
    def __init__(self, models_dir, default_camera=None):
        super().__init__()
        self.models_dir = Path(models_dir)
        self.worker = None
        self.closing = False
        self.last_preview_at = 0.0
        self.gui_frames = 0
        self.gui_tick = perf_counter()
        self.gui_fps = 0.0
        self.document_checkpoint = None
        self.buffer = PreviewBuffer()
        self.setWindowTitle('PrivacyVision — Webcam')
        self.resize(1000, 760)
        root = QWidget(); self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        self.preview = QLabel('Chưa bật camera')
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(320, 240)
        self.preview.setStyleSheet('background: black; color: white;')
        layout.addWidget(self.preview, 1)
        layout.addWidget(QLabel('AI có thể bỏ sót. Tắt detector sẽ bỏ bảo vệ loại thông tin tương ứng.'))
        controls = QHBoxLayout(); layout.addLayout(controls)
        self.checks = {}
        for source in MODEL_FILES:
            check = QCheckBox(source); check.setChecked(True)
            controls.addWidget(check); self.checks[source] = check
        self.mode = QComboBox(); self.mode.addItems(['black', 'blur', 'pixelate'])
        controls.addWidget(self.mode)
        self.device = QComboBox(); self.device.addItems(['auto', 'cpu', '0'])
        controls.addWidget(self.device)
        self.camera = QSpinBox(); self.camera.setRange(0, 20)
        if default_camera is not None:
            self.camera.setValue(default_camera)
        elif Path('/dev/video1').exists():
            self.camera.setValue(1)
        controls.addWidget(QLabel('Camera')); controls.addWidget(self.camera)
        self.rotate = QCheckBox('Thử 4 góc giấy tờ (chậm)'); layout.addWidget(self.rotate)
        self.whole_documents = QCheckBox('Che vùng giấy từ cụm field (ít field: che từng trường)')
        self.whole_documents.setChecked(True); layout.addWidget(self.whole_documents)
        self.whole_documents.hide()  # Preserve compatibility for existing callers/tests.
        self.document_mode = QComboBox(); self.document_mode.addItems(['Fields only', 'Full document'])
        self.document_mode.setCurrentIndex(1)
        self.document_mode.currentIndexChanged.connect(lambda i: self.whole_documents.setChecked(i == 1))
        layout.addWidget(QLabel('Document protection')); layout.addWidget(self.document_mode)
        self.strategy = QComboBox(); self.strategy.addItems(['heuristic', 'detector', 'hybrid'])
        for i in (1, 2): self.strategy.model().item(i).setEnabled(False)
        self.strategy.setToolTip('Detector/hybrid cần checkpoint toàn giấy với taxonomy phù hợp; chưa có model kèm theo.')
        layout.addWidget(self.strategy)
        self.checkpoint_button = QPushButton('Chọn checkpoint Document Detector cục bộ')
        self.checkpoint_button.clicked.connect(self.select_document_checkpoint)
        layout.addWidget(self.checkpoint_button)
        doc_checkpoint = self.models_dir / 'document_best.pt'
        if doc_checkpoint.is_file():
            self.document_checkpoint = doc_checkpoint
            for i in (1, 2): self.strategy.model().item(i).setEnabled(True)
            self.strategy.setCurrentText('detector')
            self.checkpoint_button.setText(f'Đã chọn: {doc_checkpoint.name}')
            if 'card' in self.checks: self.checks['card'].setChecked(False)
            if 'cccd' in self.checks: self.checks['cccd'].setChecked(False)
        self.strict = QCheckBox('Strict privacy mode (có thể che đen cả cảnh không chứa giấy tờ)')
        layout.addWidget(self.strict)
        layout.addWidget(QLabel('Heuristic không xác nhận mép giấy. Full + strict chưa có document detector sẽ che đen.'))
        camera_controls = QHBoxLayout(); layout.addLayout(camera_controls)
        self.camera_fps = QSpinBox(); self.camera_fps.setRange(5,60); self.camera_fps.setValue(30)
        camera_controls.addWidget(QLabel('FPS camera yêu cầu')); camera_controls.addWidget(self.camera_fps)
        self.manual_exposure = QCheckBox('Exposure thủ công để thử tăng FPS (ảnh có thể tối)')
        camera_controls.addWidget(self.manual_exposure)
        self.exposure = QSpinBox(); self.exposure.setRange(1,10000); self.exposure.setValue(160)
        camera_controls.addWidget(self.exposure)
        quality_controls = QHBoxLayout(); layout.addLayout(quality_controls)
        self.resolution = QComboBox(); self.resolution.addItems(['640x480','1280x720','1920x1080'])
        self.resolution.setCurrentText('1280x720')
        quality_controls.addWidget(QLabel('Độ phân giải camera')); quality_controls.addWidget(self.resolution)
        self.inference_size = QComboBox(); self.inference_size.addItems(['640','800','960'])
        self.inference_size.setCurrentText('800')
        quality_controls.addWidget(QLabel('imgsz giấy tờ (cao hơn sẽ chậm hơn)')); quality_controls.addWidget(self.inference_size)
        self.document_confidence = QDoubleSpinBox()
        self.document_confidence.setRange(.05,.95); self.document_confidence.setSingleStep(.05)
        self.document_confidence.setValue(.25)
        quality_controls.addWidget(QLabel('Conf giấy tờ')); quality_controls.addWidget(self.document_confidence)
        self.start_button = QPushButton('Bắt đầu / áp dụng cấu hình')
        self.stop_button = QPushButton('Dừng'); self.stop_button.setEnabled(False)
        layout.addWidget(self.start_button); layout.addWidget(self.stop_button)
        self.status = QLabel('Đang chờ'); layout.addWidget(self.status)
        self.metrics = QLabel('FPS: — | Xử lý: — | Thiết bị: —'); layout.addWidget(self.metrics)
        self.metrics.setWordWrap(True)
        self.start_button.clicked.connect(self.start_session)
        self.stop_button.clicked.connect(self.stop_session)
        self.timer = QTimer(self); self.timer.timeout.connect(self.refresh); self.timer.start(16)

    def editable(self, enabled):
        for widget in [*self.checks.values(), self.mode, self.device, self.camera, self.rotate,
                       self.whole_documents, self.camera_fps, self.manual_exposure, self.exposure,
                       self.resolution, self.inference_size, self.document_confidence, self.start_button,
                       self.document_mode, self.strategy, self.checkpoint_button, self.strict]:
            widget.setEnabled(enabled)
        self.stop_button.setEnabled(not enabled)

    def select_document_checkpoint(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Checkpoint tin cậy', str(self.models_dir), 'YOLO (*.pt)')
        if path:
            self.document_checkpoint = Path(path)
            for i in (1, 2): self.strategy.model().item(i).setEnabled(True)
            self.checkpoint_button.setText(self.document_checkpoint.name)

    def start_session(self):
        if self.worker is not None:
            return
        selected = [s for s, c in self.checks.items() if c.isChecked()]
        if not selected:
            self.status.setText('Cần bật ít nhất một detector.'); return
        try:
            config = PipelineConfig([DetectorConfig(s, self.models_dir/MODEL_FILES[s],
                .65 if s == 'face' else self.document_confidence.value() if s in ('card','cccd') else .25,
                int(self.inference_size.currentText()) if s in ('card','cccd') else 640) for s in selected],
                device=self.device.currentText(), mode=self.mode.currentText(),
                document_rotations=(0,90,180,270) if self.rotate.isChecked() else (0,),
                whole_documents=self.whole_documents.isChecked(),
                document_detection_strategy=self.strategy.currentText(), strict_privacy_mode=self.strict.isChecked(),
                document_detector_config=DetectorConfig('document', self.document_checkpoint) if self.document_checkpoint else None)
        except (ValueError, OSError):
            self.preview.clear()
            self.status.setText('Lỗi cấu hình — kiểm tra checkpoint và strategy giấy tờ')
            return
        self.last_preview_at = 0.0
        self.gui_frames, self.gui_tick, self.gui_fps = 0, perf_counter(), 0.0
        self.preview.clear(); self.metrics.setText('FPS: — | Xử lý: — | Thiết bị: —')
        self.buffer.clear('Đang khởi động')
        self.editable(False)
        self.worker = WebcamWorker(config, self.camera.value(), self.buffer, self,
            CameraOptions(self.camera_fps.value(), self.manual_exposure.isChecked(), self.exposure.value(),
                          width=int(self.resolution.currentText().split('x')[0]),
                          height=int(self.resolution.currentText().split('x')[1])))
        self.worker.finished.connect(self.session_finished)
        self.worker.start()

    def stop_session(self):
        if self.worker:
            self.worker.requestInterruption()
        self.preview.clear()
        self.buffer.clear('Đang dừng')
        self.stop_button.setEnabled(False)

    def session_finished(self):
        self.refresh()
        self.preview.clear()
        self.worker.deleteLater(); self.worker = None
        self.editable(True)
        if self.closing:
            self.close()

    def refresh(self):
        item, status = self.buffer.take()
        if status != 'Đang xử lý':
            self.status.setText(status)
        if status != 'Đang xử lý':
            self.preview.clear()
            self.metrics.setText('FPS: — | Xử lý: — | Thiết bị: —')
        elif item is not None and self.worker and not self.worker.isInterruptionRequested():
            if item.session_id != self.worker.session_id or perf_counter() - item.published_at > 2:
                self.preview.clear()
                self.status.setText('Đang chờ kết quả mới — preview đã hết hạn')
                return
            self.last_preview_at = item.published_at
            self.gui_frames += 1
            now = perf_counter()
            if now - self.gui_tick >= 1:
                self.gui_fps = self.gui_frames / (now - self.gui_tick)
                self.gui_frames, self.gui_tick = 0, now
            self.status.setText('Strict blackout — ' + ', '.join(item.coverage_reasons) if item.strict_blackout
                                else 'Coverage unverified — độ bao phủ chưa được xác minh')
            if item.document_fallback and not item.strict_blackout:
                self.status.setText('Cụm chưa đủ field — chỉ che từng trường; có thể còn sót thông tin')
            # Own the Qt image storage; numpy buffer never escapes by reference.
            bgr = np.ascontiguousarray(item.image)
            h, w = bgr.shape[:2]
            image = QImage(bgr.data, w, h, bgr.strides[0], QImage.Format.Format_BGR888).copy()
            pixmap = QPixmap.fromImage(image)
            self.preview.setPixmap(pixmap.scaled(self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                                Qt.TransformationMode.FastTransformation))
            times = ' | '.join(f'{k}: {v:.0f} ms' for k,v in item.timings_ms.items() if k not in ('total','inference'))
            counts=' | '.join(f'{k}: {v}' for k,v in item.detection_counts.items())
            self.metrics.setText(f'Bật: {", ".join(item.enabled_detectors)} | FPS GUI: {self.gui_fps:.1f} | FPS AI: {item.fps:.1f} | Camera: {item.capture_fps:.1f} | AI: {item.processing_ms:.0f} ms | {item.device}\n'
                                 f'Chờ frame: {item.capture_wait_ms:.0f} ms | Từ đọc đến che: {item.frame_age_ms:.0f} ms | {times}\n{counts}\n{item.camera_info}')

        if status == 'Đang xử lý' and self.last_preview_at and perf_counter() - self.last_preview_at > 2:
            self.preview.clear()
            self.status.setText('Đang chờ kết quả mới — preview đã hết hạn')

    def closeEvent(self, event):
        if self.worker is not None:
            self.closing = True
            self.stop_session()
            event.ignore()  # Keep event loop alive until native read/inference returns.
        else:
            event.accept()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='PrivacyVision PySide6 webcam')
    parser.add_argument('--models-dir', type=Path, default=Path(__file__).resolve().parents[3]/'models')
    parser.add_argument('--camera', type=int, default=None, help='Chỉ số camera mặc định')
    args = parser.parse_args()
    app = QApplication([])
    window = WebcamWindow(args.models_dir, default_camera=args.camera); window.show()
    return app.exec()
