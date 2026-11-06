from datetime import datetime
from functools import partial
from pathlib import Path
from time import perf_counter
from uuid import uuid4
import cv2
import numpy as np

from PySide6.QtCore import Qt, QThread, QTimer, Signal, QUrl, QSize, QPointF, QRectF
from PySide6.QtGui import (
    QImage, QPixmap, QDesktopServices, QIcon, QPainter, QColor,
    QPen, QBrush, QPolygonF
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QCheckBox, QComboBox, QSpinBox, QDoubleSpinBox,
    QFileDialog, QGroupBox, QFrame, QSlider, QGridLayout, QStyle
)

from privacy_vision.config import PipelineConfig, DetectorConfig, MODEL_FILES
from .session import PreviewBuffer, run_session
from .camera import CameraOptions, open_camera
from privacy_vision.virtualcam import check_virtual_cam_support, VirtualCamBroadcaster


def get_icon(name: str) -> QIcon:
    """Create or load a button icon using theme icons, standard pixmaps, or vector fallbacks."""
    theme_fallbacks = {
        'camera': ['camera-photo', 'camera', 'camera-web'],
        'folder': ['folder-open', 'folder', 'document-open'],
        'image': ['image-x-generic', 'image', 'applications-graphics'],
        'video': ['video-x-generic', 'video', 'applications-multimedia'],
        'start': ['media-playback-start', 'media-play', 'player_play'],
        'stop': ['media-playback-stop', 'media-stop', 'player_stop'],
        'record': ['media-record'],
    }
    for theme_name in theme_fallbacks.get(name, [name]):
        if QIcon.hasThemeIcon(theme_name):
            icon = QIcon.fromTheme(theme_name)
            if not icon.isNull():
                return icon

    app = QApplication.instance()
    style = app.style() if app else None
    if style:
        sp_map = {
            'camera': QStyle.StandardPixmap.SP_FileDialogDetailedView,
            'folder': QStyle.StandardPixmap.SP_DirOpenIcon,
            'image': QStyle.StandardPixmap.SP_FileDialogDetailedView,
            'video': QStyle.StandardPixmap.SP_MediaPlay,
            'start': QStyle.StandardPixmap.SP_MediaPlay,
            'stop': QStyle.StandardPixmap.SP_MediaStop,
            'record': QStyle.StandardPixmap.SP_DialogYesButton,
        }
        if name in sp_map:
            icon = style.standardIcon(sp_map[name])
            if not icon.isNull():
                return icon

    pix = QPixmap(32, 32)
    pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(QColor('#e4e4e7'), 2)
    p.setPen(pen)

    if name == 'camera':
        p.setBrush(QBrush(QColor('#3f3f46')))
        p.drawRoundedRect(QRectF(4, 8, 24, 18), 3, 3)
        p.drawRoundedRect(QRectF(11, 4, 10, 5), 2, 2)
        p.setBrush(QBrush(QColor('#18181b')))
        p.drawEllipse(QRectF(10, 11, 12, 12))
    elif name == 'folder':
        p.setBrush(QBrush(QColor('#eab308')))
        p.setPen(QPen(QColor('#ca8a04'), 1.5))
        p.drawRoundedRect(QRectF(4, 9, 24, 17), 2, 2)
        p.drawRoundedRect(QRectF(4, 6, 11, 5), 2, 2)
    elif name == 'image':
        p.setBrush(QBrush(QColor('#3f3f46')))
        p.drawRoundedRect(QRectF(4, 5, 24, 22), 3, 3)
        p.setBrush(QBrush(QColor('#e4e4e7')))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QRectF(8, 9, 5, 5))
        poly = QPolygonF([QPointF(6, 23), QPointF(14, 15), QPointF(19, 20), QPointF(23, 16), QPointF(26, 23)])
        p.drawPolygon(poly)
    elif name == 'video':
        p.setBrush(QBrush(QColor('#3f3f46')))
        p.drawRoundedRect(QRectF(4, 6, 17, 20), 3, 3)
        p.setBrush(QBrush(QColor('#e4e4e7')))
        poly = QPolygonF([QPointF(21, 11), QPointF(28, 7), QPointF(28, 25), QPointF(21, 21)])
        p.drawPolygon(poly)
    elif name == 'start':
        p.setBrush(QBrush(QColor('#22c55e')))
        p.setPen(Qt.PenStyle.NoPen)
        poly = QPolygonF([QPointF(8, 6), QPointF(26, 16), QPointF(8, 26)])
        p.drawPolygon(poly)
    elif name == 'stop':
        p.setBrush(QBrush(QColor('#ef4444')))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(QRectF(7, 7, 18, 18), 3, 3)
    elif name == 'record':
        p.setBrush(QBrush(QColor('#ef4444')))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QRectF(6, 6, 20, 20))
    p.end()
    return QIcon(pix)


DARK_STYLESHEET = """
QMainWindow {
    background-color: #121214;
    color: #f4f4f5;
}
QWidget {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 13px;
    color: #e4e4e7;
}
QGroupBox {
    border: 1px solid #27272a;
    border-radius: 8px;
    margin-top: 8px;
    padding-top: 10px;
    font-weight: 600;
    color: #f4f4f5;
    background-color: #18181b;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 5px;
    background-color: #18181b;
    color: #f4f4f5;
}
QLabel {
    color: #d4d4d8;
}
QCheckBox {
    color: #f4f4f5;
    spacing: 7px;
}
QCheckBox::indicator {
    width: 17px;
    height: 17px;
    border-radius: 4px;
    border: 1px solid #52525b;
    background: #27272a;
}
QCheckBox::indicator:checked {
    background: #e4e4e7;
    border-color: #f4f4f5;
}
QComboBox, QSpinBox, QDoubleSpinBox {
    background-color: #27272a;
    color: #f4f4f5;
    border: 1px solid #3f3f46;
    border-radius: 6px;
    padding: 4px 8px;
    min-height: 24px;
}
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1px solid #71717a;
}
QComboBox::drop-down {
    border: none;
    padding-right: 6px;
}
QPushButton {
    background-color: #27272a;
    color: #f4f4f5;
    border: 1px solid #3f3f46;
    border-radius: 6px;
    padding: 6px 12px;
    font-weight: 600;
    min-height: 26px;
}
QPushButton:hover {
    background-color: #3f3f46;
    border-color: #52525b;
}
QPushButton:pressed {
    background-color: #18181b;
}
QPushButton:disabled {
    background-color: #18181b;
    color: #52525b;
    border-color: #27272a;
}
QSlider::groove:horizontal {
    border: 1px solid #3f3f46;
    height: 6px;
    background: #27272a;
    border-radius: 3px;
}
QSlider::sub-page:horizontal {
    background: #71717a;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #f4f4f5;
    border: 1px solid #71717a;
    width: 14px;
    margin-top: -5px;
    margin-bottom: -5px;
    border-radius: 7px;
}
QSlider::handle:horizontal:hover {
    background: #ffffff;
}
"""


class WebcamWorker(QThread):
    def __init__(self, config, camera, buffer, parent=None, camera_options=None, vcam_broadcaster=None):
        super().__init__(parent)
        self.session_id = uuid4().hex
        self.config, self.camera, self.buffer = config, camera, buffer
        self.camera_options = camera_options or CameraOptions()
        self.vcam_broadcaster = vcam_broadcaster

    def run(self):
        run_session(self.config, self.camera, self.buffer, self.isInterruptionRequested,
                    capture_factory=partial(open_camera, options=self.camera_options), session_id=self.session_id,
                    vcam_broadcaster=self.vcam_broadcaster)


class VideoFileWorker(QThread):
    progress = Signal(int, int, float, float, object)  # current, total, fps, elapsed, frame
    finished_signal = Signal(dict)
    error_signal = Signal(str)

    def __init__(self, input_path, output_path, pipeline, tracker=None, parent=None):
        super().__init__(parent)
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.pipeline = pipeline
        self.tracker = tracker
        self._interrupted = False

    def requestInterruption(self):
        self._interrupted = True
        super().requestInterruption()

    def run(self):
        try:
            from privacy_vision.cli.video import process_video

            def cb(current, total, fps, elapsed, frame=None):
                if self._interrupted or self.isInterruptionRequested():
                    return False
                self.progress.emit(current, total, fps, elapsed, frame)
                return True

            report = process_video(
                self.input_path, self.output_path, self.pipeline,
                tracker=self.tracker, progress_callback=cb
            )
            self.finished_signal.emit(report)
        except Exception as e:
            self.error_signal.emit(str(e))


class WebcamWindow(QMainWindow):
    def __init__(self, models_dir, default_camera=None):
        super().__init__()
        self.models_dir = Path(models_dir)
        self.worker = None
        self.video_file_worker = None
        self.closing = False
        self.last_preview_at = 0.0
        self.gui_frames = 0
        self.gui_tick = perf_counter()
        self.gui_fps = 0.0
        self.document_checkpoint = None
        self.buffer = PreviewBuffer()

        # State for snapshot and recording
        self.last_frame = None
        self.is_recording = False
        self.video_writer = None
        self.record_frames = 0
        self.record_start_time = 0.0
        self.record_path = None

        # State for Virtual Camera (pyvirtualcam / Google Meet / Zoom)
        self.vcam_support = check_virtual_cam_support()
        self.vcam_broadcaster = VirtualCamBroadcaster(
            device=self.vcam_support.get('device'),
            enabled=False
        )

        self.setWindowTitle('PrivacyVision — Hệ thống che mờ bảo vệ dữ liệu nhạy cảm')
        self.resize(1200, 800)
        self.setStyleSheet(DARK_STYLESHEET)

        root = QWidget()
        self.setCentralWidget(root)
        main_layout = QVBoxLayout(root)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        # =========================================================================
        # 1. KHU VỰC TRÊN: PREVIEW (TRÁI) & BẢNG CẤU HÌNH AI (PHẢI)
        # =========================================================================
        top_split = QHBoxLayout()
        top_split.setSpacing(12)

        # --- BÊN TRÁI: MÀN HÌNH CAMERA / XEM TRƯỚC ---
        preview_box = QWidget()
        prev_layout = QVBoxLayout(preview_box)
        prev_layout.setContentsMargins(0, 0, 0, 0)
        prev_layout.setSpacing(4)

        self.preview = QLabel('Chưa bật camera hoặc chưa tải tệp')
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(540, 380)
        self.preview.setStyleSheet(
            'background-color: #09090b; color: #a1a1aa; border: 2px solid #27272a; '
            'border-radius: 8px; font-size: 14px; font-weight: 500;'
        )
        prev_layout.addWidget(self.preview, 1)
        top_split.addWidget(preview_box, 3)

        # --- BÊN PHẢI: BẢNG CẤU HÌNH AI & TÍNH NĂNG ĐẦY ĐỦ ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        # --- Nhóm 1: Đối tượng Nhận diện ---
        grp_detectors = QGroupBox('🎯 Đối tượng Nhận diện')
        det_layout = QVBoxLayout(grp_detectors)
        det_layout.setSpacing(5)

        checks_layout = QVBoxLayout()
        checks_layout.setSpacing(4)
        self.checks = {}

        self.check_face = QCheckBox('Khuôn mặt (Face)')
        self.check_face.setChecked(True)
        checks_layout.addWidget(self.check_face)
        self.checks['face'] = self.check_face

        self.check_plate = QCheckBox('Biển số xe (Plate)')
        self.check_plate.setChecked(True)
        checks_layout.addWidget(self.check_plate)
        self.checks['plate'] = self.check_plate

        self.check_doc = QCheckBox('Giấy tờ & Thẻ (CCCD, ATM, Bằng lái)')
        self.check_doc.setChecked(True)
        checks_layout.addWidget(self.check_doc)
        self.checks['document'] = self.check_doc

        # Legacy keys for test compatibility (hidden)
        self.check_card = QCheckBox('Thẻ ATM / Card')
        self.check_card.setChecked(True)
        self.check_card.hide()
        self.checks['card'] = self.check_card

        self.check_cccd = QCheckBox('CCCD')
        self.check_cccd.setChecked(True)
        self.check_cccd.hide()
        self.checks['cccd'] = self.check_cccd

        det_layout.addLayout(checks_layout)

        row_conf = QHBoxLayout()
        row_conf.addWidget(QLabel('Độ nhạy AI (Conf):'))
        self.document_confidence = QDoubleSpinBox()
        self.document_confidence.setRange(.05, .95)
        self.document_confidence.setSingleStep(.05)
        self.document_confidence.setValue(.25)
        row_conf.addWidget(self.document_confidence)
        det_layout.addLayout(row_conf)

        lbl_conf_help = QLabel('ℹ️ Số thấp bắt nhạy hơn; số cao tăng độ chuẩn xác')
        lbl_conf_help.setStyleSheet('color: #a1a1aa; font-size: 11px;')
        det_layout.addWidget(lbl_conf_help)

        right_layout.addWidget(grp_detectors)

        # --- Nhóm 2: Hiệu ứng Che Bảo Vệ ---
        grp_effect = QGroupBox('🎨 Hiệu ứng Che')
        effect_layout = QVBoxLayout(grp_effect)
        effect_layout.setSpacing(5)

        row_mode = QHBoxLayout()
        row_mode.addWidget(QLabel('Kiểu che:'))
        self.mode = QComboBox()
        self.mode.addItems(['pixelate', 'black', 'blur'])
        row_mode.addWidget(self.mode)
        effect_layout.addLayout(row_mode)

        lbl_mode_help = QLabel('ℹ️ pixelate: hạt vuông | blur: làm mờ mượt | black: che đen')
        lbl_mode_help.setStyleSheet('color: #a1a1aa; font-size: 11px;')
        effect_layout.addWidget(lbl_mode_help)

        # Container độ to hạt che (Tự động ẩn đi khi chọn black hoặc blur)
        self.row_pixel_widget = QWidget()
        row_pixel = QHBoxLayout(self.row_pixel_widget)
        row_pixel.setContentsMargins(0, 0, 0, 0)
        row_pixel.addWidget(QLabel('Độ to hạt che:'))
        self.pixel_size = QSpinBox()
        self.pixel_size.setRange(4, 64)
        self.pixel_size.setSingleStep(4)
        self.pixel_size.setValue(16)
        self.pixel_size.setSuffix(' px')
        row_pixel.addWidget(self.pixel_size)

        self.pixel_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.pixel_size_slider.setRange(4, 64)
        self.pixel_size_slider.setSingleStep(4)
        self.pixel_size_slider.setValue(16)
        self.pixel_size_slider.valueChanged.connect(self.pixel_size.setValue)
        self.pixel_size.valueChanged.connect(self.pixel_size_slider.setValue)
        row_pixel.addWidget(self.pixel_size_slider)
        effect_layout.addWidget(self.row_pixel_widget)

        self.lbl_pixel_help = QLabel('ℹ️ Kéo thanh trượt để phóng to hoặc thu nhỏ hạt pixel')
        self.lbl_pixel_help.setStyleSheet('color: #a1a1aa; font-size: 11px;')
        effect_layout.addWidget(self.lbl_pixel_help)

        right_layout.addWidget(grp_effect)

        # --- Nhóm 3: MỤC RIÊNG CHO CHẾ ĐỘ KHUNG & NHÃN NHẬN DIỆN (THEO YÊU CẦU NGƯỜI DÙNG) ---
        grp_frame = QGroupBox('📦 Chế độ Khung & Nhãn Nhận diện')
        frame_layout = QVBoxLayout(grp_frame)
        frame_layout.setSpacing(5)

        # Bật/tắt hiển thị khung viền và nhãn AI trên màn hình
        self.show_boxes_check = QCheckBox('Hiển thị khung & nhãn nhận diện (AI Boxes)')
        self.show_boxes_check.setChecked(False)
        frame_layout.addWidget(self.show_boxes_check)

        lbl_boxes_help = QLabel('ℹ️ Vẽ khung chữ nhật và tên nhãn AI trực tiếp lên màn hình')
        lbl_boxes_help.setStyleSheet('color: #a1a1aa; font-size: 11px;')
        frame_layout.addWidget(lbl_boxes_help)

        row_f_mode = QHBoxLayout()
        row_f_mode.addWidget(QLabel('Phạm vi che khung:'))
        self.doc_frame_combo = QComboBox()
        self.doc_frame_combo.addItems(['Che toàn bộ khung thẻ', 'Chỉ che vùng chữ thông tin'])
        row_f_mode.addWidget(self.doc_frame_combo)
        frame_layout.addLayout(row_f_mode)

        lbl_f_mode_help = QLabel('ℹ️ Che kín toàn bộ phôi thẻ hoặc chỉ che các vùng chữ')
        lbl_f_mode_help.setStyleSheet('color: #a1a1aa; font-size: 11px;')
        frame_layout.addWidget(lbl_f_mode_help)

        right_layout.addWidget(grp_frame)

        # --- Nhóm 4: BẢNG THỐNG KÊ ĐỐI TƯỢNG ĐANG PHÁT HIỆN (BẢNG TRỰC QUAN) ---
        grp_stats = QGroupBox('📊 Bảng Thống kê Đối tượng Đang phát hiện')
        stats_layout = QVBoxLayout(grp_stats)
        stats_layout.setSpacing(4)

        table_frame = QFrame()
        table_frame.setStyleSheet('background-color: #121214; border: 1px solid #27272a; border-radius: 6px;')
        table_grid = QGridLayout(table_frame)
        table_grid.setContentsMargins(8, 6, 8, 6)
        table_grid.setSpacing(6)

        table_grid.addWidget(QLabel('Khuôn mặt:'), 0, 0)
        self.tbl_face = QLabel('0')
        self.tbl_face.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.tbl_face.setStyleSheet('font-weight: bold; color: #f4f4f5;')
        table_grid.addWidget(self.tbl_face, 0, 1)

        table_grid.addWidget(QLabel('Biển số xe:'), 1, 0)
        self.tbl_plate = QLabel('0')
        self.tbl_plate.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.tbl_plate.setStyleSheet('font-weight: bold; color: #f4f4f5;')
        table_grid.addWidget(self.tbl_plate, 1, 1)

        table_grid.addWidget(QLabel('Giấy tờ & Thẻ:'), 2, 0)
        self.tbl_doc = QLabel('0')
        self.tbl_doc.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.tbl_doc.setStyleSheet('font-weight: bold; color: #f4f4f5;')
        table_grid.addWidget(self.tbl_doc, 2, 1)

        table_grid.addWidget(QLabel('Tổng số vùng che:'), 3, 0)
        self.tbl_total = QLabel('0')
        self.tbl_total.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.tbl_total.setStyleSheet('font-weight: bold; color: #22c55e;')
        table_grid.addWidget(self.tbl_total, 3, 1)

        stats_layout.addWidget(table_frame)
        right_layout.addWidget(grp_stats)
        right_layout.addStretch()

        # --- CÁC THÀNH PHẦN NỘI BỘ (ẨN ĐI ĐỂ TƯƠNG THÍCH HOÀN TOÀN VỚI BỘ TEST) ---
        self.document_mode = QComboBox()
        self.document_mode.addItems(['Fields only', 'Full document'])
        self.document_mode.setCurrentIndex(1)
        self.document_mode.hide()

        self.whole_documents = QCheckBox('Che vùng giấy từ cụm field')
        self.whole_documents.setChecked(True)
        self.whole_documents.hide()
        self.document_mode.currentIndexChanged.connect(lambda i: self.whole_documents.setChecked(i == 1))

        self.strategy = QComboBox()
        self.strategy.addItems(['heuristic', 'detector', 'hybrid'])
        for i in (1, 2):
            self.strategy.model().item(i).setEnabled(False)
        self.strategy.hide()

        self.checkpoint_button = QPushButton('Chọn model (.pt)...')
        self.checkpoint_button.clicked.connect(self.select_document_checkpoint)
        self.checkpoint_button.hide()

        self.inference_size = QComboBox()
        self.inference_size.addItems(['640', '800', '960'])
        self.inference_size.setCurrentText('800')
        self.inference_size.hide()

        self.rotate = QCheckBox('Quét xoay thẻ')
        self.rotate.setChecked(False)
        self.rotate.hide()

        self.strict = QCheckBox('Bảo mật tuyệt đối')
        self.strict.setChecked(False)
        self.strict.hide()

        # Kiểm tra file checkpoint
        doc_checkpoint = self.models_dir / 'document_best.pt'
        if doc_checkpoint.is_file():
            self.document_checkpoint = doc_checkpoint
            for i in (1, 2):
                self.strategy.model().item(i).setEnabled(True)
            self.strategy.setCurrentText('detector')
            self.checkpoint_button.setText(f'Model giấy tờ: {doc_checkpoint.name}')
            if 'card' in self.checks:
                self.checks['card'].setChecked(False)
            if 'cccd' in self.checks:
                self.checks['cccd'].setChecked(False)

        top_split.addWidget(right_panel, 2)
        main_layout.addLayout(top_split, 1)

        # =========================================================================
        # 2. KHU VỰC DƯỚI: ĐIỀU KHIỂN CHỨC NĂNG, PHẦN CỨNG & ĐỘ PHÂN GIẢI (BOTTOM DECK)
        # =========================================================================
        bottom_deck = QGroupBox('⚙️ Điều khiển Chức năng & Camera')
        bottom_layout = QVBoxLayout(bottom_deck)
        bottom_layout.setContentsMargins(10, 10, 10, 10)
        bottom_layout.setSpacing(8)

        # Hàng 1: Nút bấm chức năng (Có icon, không có dấu ...)
        btn_bar = QHBoxLayout()
        btn_bar.setSpacing(8)

        self.start_button = QPushButton('Bắt đầu')
        self.start_button.setIcon(get_icon('start'))
        self.start_button.setIconSize(QSize(18, 18))
        btn_bar.addWidget(self.start_button)

        self.stop_button = QPushButton('Dừng')
        self.stop_button.setIcon(get_icon('stop'))
        self.stop_button.setIconSize(QSize(18, 18))
        self.stop_button.setEnabled(False)
        btn_bar.addWidget(self.stop_button)

        self.snapshot_button = QPushButton('Chụp ảnh')
        self.snapshot_button.setIcon(get_icon('camera'))
        self.snapshot_button.setIconSize(QSize(18, 18))
        self.snapshot_button.setEnabled(False)
        btn_bar.addWidget(self.snapshot_button)

        self.record_button = QPushButton('Ghi video')
        self.record_button.setIcon(get_icon('record'))
        self.record_button.setIconSize(QSize(18, 18))
        self.record_button.setEnabled(False)
        btn_bar.addWidget(self.record_button)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet('color: #3f3f46; max-height: 24px;')
        btn_bar.addWidget(sep)

        self.open_output_button = QPushButton('Mở thư mục kết quả')
        self.open_output_button.setIcon(get_icon('folder'))
        self.open_output_button.setIconSize(QSize(18, 18))
        btn_bar.addWidget(self.open_output_button)

        self.btn_process_image = QPushButton('Che ảnh')
        self.btn_process_image.setIcon(get_icon('image'))
        self.btn_process_image.setIconSize(QSize(18, 18))
        btn_bar.addWidget(self.btn_process_image)

        self.btn_process_video = QPushButton('Che video')
        self.btn_process_video.setIcon(get_icon('video'))
        self.btn_process_video.setIconSize(QSize(18, 18))
        btn_bar.addWidget(self.btn_process_video)

        bottom_layout.addLayout(btn_bar)

        # Hàng 2: Thiết lập Camera, Cổng và Độ phân giải (Đổi trực tiếp được)
        cam_bar = QHBoxLayout()
        cam_bar.setSpacing(10)

        cam_bar.addWidget(QLabel('Cổng Camera:'))
        self.camera = QSpinBox()
        self.camera.setRange(0, 20)
        if default_camera is not None:
            self.camera.setValue(default_camera)
        elif Path('/dev/video1').exists():
            self.camera.setValue(1)
        cam_bar.addWidget(self.camera)

        cam_bar.addWidget(QLabel('Độ phân giải:'))
        self.resolution = QComboBox()
        self.resolution.addItems(['1280x720', '960x540', '640x480', '640x360'])
        self.resolution.setCurrentText('1280x720')
        cam_bar.addWidget(self.resolution)

        cam_bar.addWidget(QLabel('FPS:'))
        self.camera_fps = QSpinBox()
        self.camera_fps.setRange(5, 60)
        self.camera_fps.setValue(30)
        cam_bar.addWidget(self.camera_fps)

        cam_bar.addWidget(QLabel('Thiết bị:'))
        self.device = QComboBox()
        self.device.addItems(['auto', 'cpu', '0'])
        cam_bar.addWidget(self.device)

        # Virtual Camera (Meet / Zoom)
        vsep = QFrame()
        vsep.setFrameShape(QFrame.Shape.VLine)
        vsep.setStyleSheet('color: #3f3f46; max-height: 24px;')
        cam_bar.addWidget(vsep)

        self.virtual_cam_check = QCheckBox('🎥 Virtual Camera (Meet/Zoom)')
        self.virtual_cam_check.setToolTip('Phát luồng video đã che mờ vào webcam ảo để gọi Google Meet, Zoom, Teams')
        cam_bar.addWidget(self.virtual_cam_check)

        vcam_init_text = '🔴 VCam: Tắt'
        self.vcam_badge = QLabel(vcam_init_text)
        self.vcam_badge.setStyleSheet('color: #a1a1aa; font-size: 11px;')
        cam_bar.addWidget(self.vcam_badge)

        # Ẩn phơi sáng thủ công để tránh rườm rà
        self.manual_exposure = QCheckBox('Manual Exposure')
        self.manual_exposure.setChecked(False)
        self.manual_exposure.hide()

        self.exposure = QSpinBox()
        self.exposure.setRange(1, 10000)
        self.exposure.setValue(160)
        self.exposure.hide()

        cam_bar.addStretch()
        bottom_layout.addLayout(cam_bar)

        # Hàng 3: Trạng thái & Thống kê (Màu mặc định, hoàn toàn không có dấu _)
        status_bar = QHBoxLayout()
        status_bar.setSpacing(10)

        self.status = QLabel('Đang chờ')
        self.status.setStyleSheet('color: #f4f4f5; font-weight: 600; min-width: 200px;')
        status_bar.addWidget(self.status, 1)

        self.metrics = QLabel('FPS: — | Xử lý: — | Thiết bị: —')
        self.metrics.setStyleSheet('background-color: #121214; border: 1px solid #27272a; border-radius: 4px; padding: 4px 8px; color: #a1a1aa; font-family: monospace; font-size: 11px;')
        status_bar.addWidget(self.metrics, 2)

        bottom_layout.addLayout(status_bar)
        main_layout.addWidget(bottom_deck)

        # =========================================================================
        # 3. KẾT NỐI SỰ KIỆN (SIGNALS & SLOTS) VÀ ĐIỀU CHỈNH THÔNG SỐ LIVE
        # =========================================================================
        self.start_button.clicked.connect(self.start_session)
        self.stop_button.clicked.connect(self.stop_session)
        self.snapshot_button.clicked.connect(self.take_snapshot)
        self.record_button.clicked.connect(self.toggle_recording)
        self.btn_process_image.clicked.connect(self.process_image_file)
        self.btn_process_video.clicked.connect(self.process_video_file)
        self.open_output_button.clicked.connect(self.open_output_folder)

        # Chuyển đổi độ phân giải trực tiếp
        self.resolution.currentIndexChanged.connect(self.on_resolution_changed)

        # Đồng bộ chế độ khung hình
        self.doc_frame_combo.currentIndexChanged.connect(self.on_frame_mode_changed)

        # Ẩn hiện độ to hạt che theo kiểu che (chỉ hiện khi kiểu che là pixelate)
        self.mode.currentTextChanged.connect(self.on_mode_changed)

        # CẬP NHẬT THÔNG SỐ TRỰC TIẾP (LIVE)
        self.pixel_size.valueChanged.connect(self.apply_live_settings)
        self.document_confidence.valueChanged.connect(self.apply_live_settings)
        self.check_face.stateChanged.connect(self.apply_live_settings)
        self.check_plate.stateChanged.connect(self.apply_live_settings)
        self.check_doc.stateChanged.connect(self.apply_live_settings)
        self.virtual_cam_check.toggled.connect(self.on_virtual_cam_toggled)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(16)

    def on_virtual_cam_toggled(self, checked: bool):
        if checked and not self.vcam_support.get('available', False):
            self.status.setText('Chưa có webcam ảo (v4l2loopback). Chạy lệnh: sudo modprobe v4l2loopback')
            self.virtual_cam_check.setChecked(False)
            return

        self.vcam_broadcaster.enabled = checked
        if checked:
            dev = self.vcam_broadcaster.device_name
            self.vcam_badge.setText(f'🟢 VCam: {dev}')
            self.vcam_badge.setStyleSheet('color: #22c55e; font-size: 11px; font-weight: bold;')
            self.status.setText(f'🎥 Đã bật Virtual Camera ({dev}) — Sẵn sàng cho Meet/Zoom!')
        else:
            self.vcam_badge.setText('🔴 VCam: Tắt')
            self.vcam_badge.setStyleSheet('color: #a1a1aa; font-size: 11px;')
            self.status.setText('🎥 Đã tắt Virtual Camera.')

    def on_frame_mode_changed(self, idx):
        # idx 0: Full document, idx 1: Fields only
        self.document_mode.setCurrentIndex(1 if idx == 0 else 0)
        self.apply_live_settings()

    def on_mode_changed(self, mode_text):
        is_pixel = (mode_text == 'pixelate')
        self.row_pixel_widget.setVisible(is_pixel)
        self.lbl_pixel_help.setVisible(is_pixel)
        self.apply_live_settings()

    def on_resolution_changed(self):
        """Khi người dùng đổi độ phân giải bên dưới, nếu camera đang chạy thì tự khởi động lại với kích thước mới."""
        if self.worker is not None and not self.worker.isInterruptionRequested():
            self.status.setText(f'Đang đổi độ phân giải sang {self.resolution.currentText()}...')
            self.stop_session()
            QTimer.singleShot(250, self.start_session)

    def apply_live_settings(self):
        """Cập nhật tức thời thông số che mờ trực tiếp mà không cần khởi động lại camera."""
        if self.worker is None or not hasattr(self.worker, 'config') or self.worker.config is None:
            return
        cfg = self.worker.config
        cfg.mode = self.mode.currentText()
        cfg.pixel_size = self.pixel_size.value()
        cfg.whole_documents = (self.document_mode.currentIndex() == 1)
        conf = self.document_confidence.value()
        imgsz = int(self.inference_size.currentText())

        has_doc = (
            self.checks.get('document', QCheckBox()).isChecked()
            and self.document_checkpoint is not None
            and self.document_checkpoint.is_file()
        )

        selected = []
        for s, c in self.checks.items():
            if s in MODEL_FILES and c.isChecked():
                if has_doc and s in ('card', 'cccd'):
                    continue
                selected.append(s)

        cfg.detectors = [
            DetectorConfig(
                s, self.models_dir / MODEL_FILES[s],
                0.65 if s == 'face' else conf if s in ('card', 'cccd') else 0.25,
                imgsz if s in ('card', 'cccd') else 640
            ) for s in selected
        ]
        if has_doc:
            cfg.document_detector_config = DetectorConfig('document', self.document_checkpoint,
                                                          confidence=conf, imgsz=imgsz)
            cfg.document_detection_strategy = 'detector'
        else:
            cfg.document_detector_config = None
            cfg.document_detection_strategy = self.strategy.currentText()

    def editable(self, enabled):
        """Khóa các điều khiển phần cứng cần khởi động lại. Các thông số che mờ luôn mở để chỉnh live!"""
        for widget in [self.device, self.camera, self.camera_fps, self.manual_exposure,
                       self.exposure, self.start_button, self.checkpoint_button,
                       self.btn_process_image, self.btn_process_video]:
            widget.setEnabled(enabled)

        # Cho phép chỉnh độ phân giải, kiểu che, độ to hạt pixel, độ nhạy conf, bật/tắt đối tượng trực tiếp
        for widget in [*self.checks.values(), self.mode, self.pixel_size, self.pixel_size_slider,
                       self.document_confidence, self.resolution, self.doc_frame_combo, self.show_boxes_check,
                       self.virtual_cam_check]:
            widget.setEnabled(True)

        self.stop_button.setEnabled(not enabled)
        self.snapshot_button.setEnabled(not enabled)
        self.record_button.setEnabled(not enabled)

    def select_document_checkpoint(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Chọn mô hình giấy tờ', str(self.models_dir), 'YOLO (*.pt)')
        if path:
            self.document_checkpoint = Path(path)
            for i in (1, 2):
                self.strategy.model().item(i).setEnabled(True)
            self.strategy.setCurrentText('detector')
            self.checkpoint_button.setText(f'Model giấy tờ: {self.document_checkpoint.name}')
            self.apply_live_settings()

    def open_output_folder(self):
        outputs_dir = self.models_dir.parent / 'outputs'
        outputs_dir.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(outputs_dir.resolve())))
        self.status.setText(f'📁 Đã mở thư mục kết quả ({outputs_dir.name}/)')

    def take_snapshot(self):
        if self.last_frame is None:
            self.status.setText('Chưa có khung hình để chụp ảnh.')
            return
        snapshots_dir = self.models_dir.parent / 'outputs' / 'snapshots'
        snapshots_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        snap_path = snapshots_dir / f'snapshot_{ts}.png'
        cv2.imwrite(str(snap_path), self.last_frame)
        self.status.setText(f'📸 Đã chụp và lưu ảnh: {snap_path.name}')
        print(f'[SNAPSHOT] Đã lưu ảnh: {snap_path}')

    def toggle_recording(self):
        if not self.is_recording:
            if self.last_frame is None:
                self.status.setText('Chưa có khung hình để ghi video.')
                return
            recordings_dir = self.models_dir.parent / 'outputs' / 'recordings'
            recordings_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.record_path = recordings_dir / f'record_{ts}.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            h, w = self.last_frame.shape[:2]
            rec_fps = max(15.0, self.gui_fps) if self.gui_fps > 5.0 else 30.0
            self.video_writer = cv2.VideoWriter(str(self.record_path), fourcc, rec_fps, (w, h))
            self.is_recording = True
            self.record_start_time = perf_counter()
            self.record_frames = 0
            self.record_button.setIcon(get_icon('stop'))
            self.record_button.setText('Dừng ghi (00:00)')
            self.status.setText('🔴 Đang ghi video...')
            print(f'[RECORD] Bắt đầu ghi video: {self.record_path} ({rec_fps:.1f} FPS)')
        else:
            self.is_recording = False
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            dur = perf_counter() - self.record_start_time
            self.record_button.setIcon(get_icon('record'))
            self.record_button.setText('Ghi video')
            self.status.setText(f'Đã lưu video: {self.record_path.name} ({self.record_frames}f, {dur:.1f}s)')
            print(f'[RECORD] Đã kết thúc và lưu video: {self.record_path} ({self.record_frames}f, {dur:.1f}s)')

    def create_pipeline_config(self):
        has_doc = (
            self.checks.get('document', QCheckBox()).isChecked()
            and self.document_checkpoint is not None
            and self.document_checkpoint.is_file()
        )
        selected = []
        for s, c in self.checks.items():
            if s in MODEL_FILES and c.isChecked():
                if has_doc and s in ('card', 'cccd'):
                    continue
                selected.append(s)

        if not selected and not has_doc:
            self.status.setText('Cần bật ít nhất một detector.')
            raise ValueError('Cần bật ít nhất một detector.')

        detectors = [
            DetectorConfig(
                s, self.models_dir / MODEL_FILES[s],
                0.65 if s == 'face' else self.document_confidence.value() if s in ('card', 'cccd') else 0.25,
                int(self.inference_size.currentText()) if s in ('card', 'cccd') else 640
            ) for s in selected
        ]
        doc_cfg = None
        if has_doc:
            doc_cfg = DetectorConfig('document', self.document_checkpoint,
                                     confidence=self.document_confidence.value(),
                                     imgsz=int(self.inference_size.currentText()))

        return PipelineConfig(
            detectors=detectors,
            device=self.device.currentText(),
            mode=self.mode.currentText(),
            document_rotations=(0, 90) if self.rotate.isChecked() else (0,),
            whole_documents=self.whole_documents.isChecked(),
            document_detection_strategy='detector' if has_doc else self.strategy.currentText(),
            strict_privacy_mode=self.strict.isChecked(),
            document_detector_config=doc_cfg,
            pixel_size=self.pixel_size.value()
        )

    def start_session(self):
        if self.worker is not None or self.video_file_worker is not None:
            return
        selected = [s for s, c in self.checks.items() if c.isChecked()]
        if not selected:
            self.status.setText('Cần bật ít nhất một detector.')
            return
        try:
            config = self.create_pipeline_config()
        except (ValueError, OSError):
            self.preview.clear()
            self.status.setText('Cần bật ít nhất một detector.')
            return

        self.last_preview_at = 0.0
        self.gui_frames, self.gui_tick, self.gui_fps = 0, perf_counter(), 0.0
        self.preview.clear()
        self.metrics.setText('FPS: — | Xử lý: — | Thiết bị: —')
        self.buffer.clear('Đang khởi động')
        self.editable(False)

        res_parts = self.resolution.currentText().split('x')
        self.vcam_broadcaster.enabled = self.virtual_cam_check.isChecked()
        self.worker = WebcamWorker(
            config, self.camera.value(), self.buffer, self,
            CameraOptions(
                self.camera_fps.value(), self.manual_exposure.isChecked(), self.exposure.value(),
                width=int(res_parts[0]), height=int(res_parts[1])
            ),
            vcam_broadcaster=self.vcam_broadcaster
        )
        self.worker.finished.connect(self.session_finished)
        self.worker.start()

    def stop_session(self):
        if self.video_file_worker is not None and self.video_file_worker.isRunning():
            self.video_file_worker.requestInterruption()
            self.status.setText('Đang dừng xử lý file video...')
            return
        if self.is_recording:
            self.toggle_recording()
        if self.vcam_broadcaster is not None:
            self.vcam_broadcaster.stop()
        if self.worker:
            self.worker.requestInterruption()
        self.preview.clear()
        self.buffer.clear('Đang dừng')
        self.stop_button.setEnabled(False)
        self.snapshot_button.setEnabled(False)
        self.record_button.setEnabled(False)

    def session_finished(self):
        self.refresh()
        self.preview.clear()
        self.worker.deleteLater()
        self.worker = None
        self.editable(True)
        if self.closing:
            self.close()

    def process_image_file(self):
        if self.worker is not None or self.video_file_worker is not None:
            self.status.setText('Vui lòng dừng camera trước khi xử lý file ảnh.')
            return
        selected = [s for s, c in self.checks.items() if c.isChecked()]
        if not selected:
            self.status.setText('Cần bật ít nhất một detector.')
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self, 'Chọn hình ảnh cần che', str(Path.home()),
            'Hình ảnh (*.png *.jpg *.jpeg *.webp *.bmp)'
        )
        if not file_path:
            return

        img = cv2.imread(file_path)
        if img is None:
            self.status.setText(f'Không thể đọc file ảnh: {Path(file_path).name}')
            return

        try:
            config = self.create_pipeline_config()
            from privacy_vision.pipeline import PrivacyPipeline
            pipeline = PrivacyPipeline(config)
            t0 = perf_counter()
            result = pipeline.process(img)
            elapsed_ms = (perf_counter() - t0) * 1000.0

            out_dir = self.models_dir.parent / 'outputs' / 'images'
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            in_name = Path(file_path).stem
            out_path = out_dir / f'{in_name}_redacted_{ts}.png'
            cv2.imwrite(str(out_path), result.image)

            self.last_frame = result.image.copy()

            bgr = np.ascontiguousarray(result.image)
            h, w = bgr.shape[:2]
            image = QImage(bgr.data, w, h, bgr.strides[0], QImage.Format.Format_BGR888).copy()
            pixmap = QPixmap.fromImage(image)
            self.preview.setPixmap(
                pixmap.scaled(self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation)
            )

            counts = {}
            for d in result.detections:
                label = 'Mặt' if d.source == 'face' else 'Biển số' if d.source == 'plate' else 'Giấy tờ'
                counts[label] = counts.get(label, 0) + 1
            counts_str = ' | '.join(f'{k}: {v}' for k, v in counts.items()) if counts else 'Không có đối tượng cần che'

            self.status.setText(f'Đã che và lưu ảnh: {out_path.name}')
            self.metrics.setText(
                f'File ảnh: {Path(file_path).name} ({w}x{h}) | Thời gian AI: {elapsed_ms:.1f} ms\n'
                f'Phát hiện: {counts_str}\nĐã lưu: {out_path}'
            )
            print(f'[IMAGE] Đã xử lý ảnh: {out_path}')
        except Exception as e:
            self.status.setText(f'Lỗi khi xử lý ảnh: {e}')

    def process_video_file(self):
        if self.worker is not None or self.video_file_worker is not None:
            self.status.setText('Vui lòng dừng camera/tiến trình hiện tại trước.')
            return
        selected = [s for s, c in self.checks.items() if c.isChecked()]
        if not selected:
            self.status.setText('Cần bật ít nhất một detector.')
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self, 'Chọn video cần che', str(Path.home()),
            'Video (*.mp4 *.avi *.mov *.mkv *.webm)'
        )
        if not file_path:
            return

        try:
            config = self.create_pipeline_config()
            from privacy_vision.pipeline import PrivacyPipeline
            from privacy_vision.processing.tracking import RegionTracker
            pipeline = PrivacyPipeline(config)
            tracker = RegionTracker()

            out_dir = self.models_dir.parent / 'outputs' / 'videos'
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            in_name = Path(file_path).stem
            out_path = out_dir / f'{in_name}_redacted_{ts}.mp4'

            self.editable(False)
            self.stop_button.setEnabled(True)
            self.status.setText(f'Đang che video: {Path(file_path).name}...')

            self.video_file_worker = VideoFileWorker(file_path, out_path, pipeline, tracker, self)
            self.video_file_worker.progress.connect(self._on_video_file_progress)
            self.video_file_worker.finished_signal.connect(self._on_video_file_finished)
            self.video_file_worker.error_signal.connect(self._on_video_file_error)
            self.video_file_worker.start()
        except Exception as e:
            self.editable(True)
            self.status.setText(f'Lỗi cấu hình xử lý video: {e}')

    def _on_video_file_progress(self, current, total, fps, elapsed, frame):
        pct = (current / total * 100.0) if total > 0 else 0.0
        self.status.setText(f'Đang che video: {current}/{total} frames ({pct:.1f}%) | {fps:.1f} FPS')
        if frame is not None:
            self.last_frame = frame.copy()
            bgr = np.ascontiguousarray(frame)
            h, w = bgr.shape[:2]
            image = QImage(bgr.data, w, h, bgr.strides[0], QImage.Format.Format_BGR888).copy()
            pixmap = QPixmap.fromImage(image)
            self.preview.setPixmap(
                pixmap.scaled(self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation)
            )

    def _on_video_file_finished(self, report):
        self.video_file_worker = None
        self.editable(True)
        out_name = Path(report["output_path"]).name
        total_time = report["total_time_seconds"]
        fps = report["processing_fps"]
        frames = report["frames_processed"]
        self.status.setText(f'Hoàn thành che video! Đã lưu: {out_name}')
        self.metrics.setText(
            f'Xử lý: {frames} frames ({total_time:.1f}s) | Tốc độ: {fps:.1f} FPS\n'
            f'Đã lưu: {report["output_path"]}'
        )

    def _on_video_file_error(self, err_msg):
        self.video_file_worker = None
        self.editable(True)
        self.status.setText(f'Lỗi khi che video: {err_msg}')

    def refresh(self):
        item, status = self.buffer.take()
        if status != 'Đang xử lý':
            self.status.setText(status)
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

            self.status.setText(
                'Strict blackout — ' + ', '.join(item.coverage_reasons) if item.strict_blackout
                else 'Coverage unverified — bảo vệ riêng tư'
            )

            bgr = np.ascontiguousarray(item.image)
            self.last_frame = bgr.copy()

            if self.is_recording and self.video_writer is not None:
                self.video_writer.write(bgr)
                self.record_frames += 1
                elapsed = int(perf_counter() - self.record_start_time)
                m, s = divmod(elapsed, 60)
                self.record_button.setText(f'Dừng ghi ({m:02d}:{s:02d})')

            # Nếu bật hiển thị khung và nhãn nhận diện (AI Boxes): Vẽ lên khung hình preview
            disp_bgr = bgr.copy()
            if self.show_boxes_check.isChecked() and hasattr(item, 'detections') and item.detections:
                for d in item.detections:
                    if len(d.bbox) == 4:
                        x1, y1, x2, y2 = map(int, d.bbox)
                        color = (0, 255, 0) if d.source == 'face' else (0, 255, 255) if d.source == 'plate' else (255, 200, 0)
                        cv2.rectangle(disp_bgr, (x1, y1), (x2, y2), color, 2)
                        lbl_name = 'Mặt' if d.source == 'face' else 'Biển số' if d.source == 'plate' else 'Giấy tờ'
                        conf_text = f'{lbl_name} {d.confidence:.2f}' if d.confidence is not None else lbl_name
                        cv2.putText(disp_bgr, conf_text, (x1, max(18, y1 - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            h, w = disp_bgr.shape[:2]
            image = QImage(disp_bgr.data, w, h, disp_bgr.strides[0], QImage.Format.Format_BGR888).copy()
            pixmap = QPixmap.fromImage(image)
            self.preview.setPixmap(
                pixmap.scaled(self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation)
            )

            # Cập nhật bảng thống kê số lượng đối tượng (Hoàn toàn không có chữ có dấu _)
            faces = item.detection_counts.get('face', 0)
            plates = item.detection_counts.get('plate', 0)
            docs = item.detection_counts.get('document', 0) + item.detection_counts.get('card', 0) + item.detection_counts.get('cccd', 0)
            total = faces + plates + docs

            self.tbl_face.setText(str(faces))
            self.tbl_plate.setText(str(plates))
            self.tbl_doc.setText(str(docs))
            self.tbl_total.setText(str(total))

            counts_list = []
            if faces > 0: counts_list.append(f'Mặt: {faces}')
            if plates > 0: counts_list.append(f'Biển số: {plates}')
            if docs > 0: counts_list.append(f'Giấy tờ: {docs}')
            counts_display = ' | '.join(counts_list) if counts_list else 'Không có đối tượng cần che'

            # Cập nhật trạng thái Virtual Camera
            if getattr(item, 'vcam_active', False):
                dev_str = getattr(item, 'vcam_device', '') or self.vcam_broadcaster.device_name
                self.vcam_badge.setText(f'🟢 VCam: {dev_str} (Live)')
                self.vcam_badge.setStyleSheet('color: #22c55e; font-size: 11px; font-weight: bold;')
            elif not self.virtual_cam_check.isChecked():
                self.vcam_badge.setText('🔴 VCam: Tắt')
                self.vcam_badge.setStyleSheet('color: #a1a1aa; font-size: 11px;')

            vcam_stat = f' | 🎥 VCam: Live ({getattr(item, "vcam_device", "") or self.vcam_broadcaster.device_name})' if (getattr(item, 'vcam_active', False) or self.vcam_broadcaster.is_active) else ''
            self.metrics.setText(
                f'Camera: {item.capture_fps:.1f} FPS | AI: {item.fps:.1f} FPS | Hiển thị: {self.gui_fps:.1f} FPS | Độ trễ: {item.processing_ms:.0f} ms{vcam_stat}\n'
                f'Đang che: {counts_display}'
            )

        if status == 'Đang xử lý' and self.last_preview_at and perf_counter() - self.last_preview_at > 2:
            self.preview.clear()
            self.status.setText('Đang chờ kết quả mới — preview đã hết hạn')

    def closeEvent(self, event):
        if self.timer.isActive():
            self.timer.stop()
        if self.is_recording:
            self.toggle_recording()
        if self.vcam_broadcaster is not None:
            self.vcam_broadcaster.stop()
        if self.video_file_worker is not None and self.video_file_worker.isRunning():
            self.video_file_worker.requestInterruption()
            self.video_file_worker.wait(1000)
        if self.worker is not None:
            self.closing = True
            self.stop_session()
            event.ignore()
        else:
            event.accept()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='PrivacyVision PySide6 webcam')
    parser.add_argument('--models-dir', type=Path, default=Path(__file__).resolve().parents[3] / 'models')
    parser.add_argument('--camera', type=int, default=None, help='Chỉ số camera mặc định')
    args = parser.parse_args()
    app = QApplication([])
    window = WebcamWindow(args.models_dir, default_camera=args.camera)
    window.show()
    return app.exec()
