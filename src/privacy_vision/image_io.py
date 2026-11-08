"""Các hàm tiện ích đọc và ghi tệp hình ảnh an toàn.

Hỗ trợ đọc ảnh Unicode đa nền tảng và ghi tệp nguyên tử (atomic write).
"""
from pathlib import Path
import os
import tempfile
import cv2
import numpy as np


def read_image(path) -> np.ndarray:
    """Đọc tệp hình ảnh từ đĩa thành mảng BGR uint8 (hỗ trợ đường dẫn tiếng Việt/Unicode)."""
    data = np.fromfile(Path(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if image is None:
        raise ValueError("Không thể giải mã hình ảnh đầu vào")
    return image


def atomic_write(path, data: bytes):
    """Ghi dữ liệu ra tệp theo phương thức nguyên tử để chống hỏng file khi gặp sự cố."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".privacy-", delete=False) as f:
            name = f.name
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


def write_image(path, image):
    """Mã hóa và ghi mảng hình ảnh ra tệp định dạng PNG, JPG hoặc WebP."""
    suffix = Path(path).suffix.lower()
    if suffix not in (".png", ".jpg", ".jpeg", ".webp"):
        raise ValueError("Tệp xuất phải có định dạng PNG, JPEG hoặc WebP")
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise OSError("Mã hóa hình ảnh thất bại")
    atomic_write(path, encoded.tobytes())
