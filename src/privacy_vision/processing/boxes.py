"""Các hàm xử lý tọa độ khung bao (Bounding Box).

Hỗ trợ chuẩn hóa khung bao nguyên số (normalize_box) và hoàn trả tọa độ sau khi xoay ảnh (unrotate_box).
"""
import math


def normalize_box(box, width: int, height: int, margin: float = 0):
    """Chuẩn hóa khung bao về số nguyên nửa mở [x1, y1, x2, y2], mở rộng lề trước khi cắt theo kích thước ảnh."""
    if len(box) != 4 or not all(math.isfinite(v) for v in box):
        return None
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return None
    dx, dy = (x2 - x1) * margin, (y2 - y1) * margin
    x1, y1 = max(0, min(width, math.floor(x1 - dx))), max(0, min(height, math.floor(y1 - dy)))
    x2, y2 = max(0, min(width, math.ceil(x2 + dx))), max(0, min(height, math.ceil(y2 + dy)))
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def unrotate_box(box, width: int, height: int, angle: int):
    """Ánh xạ tọa độ khung bao từ ảnh đã xoay np.rot90(image, angle // 90) về tọa độ ảnh gốc."""
    x1, y1, x2, y2 = box
    if angle == 0:
        return tuple(box)
    if angle == 90:
        return (width - y2, x1, width - y1, x2)
    if angle == 180:
        return (width - x2, height - y2, width - x1, height - y1)
    if angle == 270:
        return (y1, height - x2, y2, height - x1)
    raise ValueError('Góc xoay không được hỗ trợ')
