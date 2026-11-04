from pathlib import Path
import os
import tempfile
import cv2
import numpy as np

def read_image(path) -> np.ndarray:
    data = np.fromfile(Path(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if image is None:
        raise ValueError("Cannot decode input image")
    return image

def atomic_write(path, data: bytes):
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
    suffix = Path(path).suffix.lower()
    if suffix not in (".png", ".jpg", ".jpeg", ".webp"):
        raise ValueError("Output must be PNG, JPEG or WebP")
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise OSError("Image encoding failed")
    atomic_write(path, encoded.tobytes())
