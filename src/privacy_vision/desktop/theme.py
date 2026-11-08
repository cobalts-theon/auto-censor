"""Giao diện và biểu tượng đồ họa (Theme & Icons) cho PrivacyVision Desktop.

Cung cấp bảng mã định kiểu nền tối hiện đại (Dark Theme QSS) và cơ chế tải biểu tượng thông minh:
ưu tiên biểu tượng chủ đề hệ điều hành, fallback sang chuẩn QStyle và tự vẽ vector với QPainter.
"""
from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import (
    QIcon, QPixmap, QPainter, QColor, QPen, QBrush, QPolygonF
)
from PySide6.QtWidgets import QApplication, QStyle


def get_icon(name: str) -> QIcon:
    """Tải hoặc tự động vẽ biểu tượng vector cho các nút chức năng trong ứng dụng.

    Thứ tự ưu tiên:
    1. Biểu tượng hệ thống chuẩn (System Theme Icon).
    2. Biểu tượng chuẩn của Qt Style (QStyle.StandardPixmap).
    3. Tự vẽ đồ họa vector chất lượng cao bằng QPainter nếu hệ thống không có sẵn.

    Args:
        name: Tên biểu tượng ('camera', 'folder', 'image', 'video', 'start', 'stop', 'record').

    Returns:
        Đối tượng QIcon hợp lệ sẵn sàng hiển thị trên giao diện.
    """
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

    # Tự vẽ vector 32x32 mượt mà chống răng cưa (Antialiased Vector Rendering)
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
