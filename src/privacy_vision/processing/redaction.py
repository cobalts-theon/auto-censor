"""Xử lý làm mờ, che đen hoặc pixelate các vùng bảo mật trên khung hình."""
import cv2
import numpy as np


def redact(image: np.ndarray, detections, mode: str = "black", pixel_size: int = 16) -> np.ndarray:
    """Áp dụng hiệu ứng che mờ (black, blur, pixelate) lên các vùng phát hiện trên ảnh.

    Xử lý theo các thành phần liên thông của mặt nạ (connected components),
    luôn lấy mẫu từ ảnh gốc để tránh hiện tượng làm mờ lặp tích lũy.
    """
    if mode not in ("black", "blur", "pixelate"):
        raise ValueError("Chế độ che mờ không hợp lệ")
    if pixel_size < 2:
        pixel_size = 16
    mask = np.zeros(image.shape[:2], np.uint8)
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        mask[y1:y2, x1:x2] = 255
    output = image.copy()
    if mode == "black":
        output[mask != 0] = 0
        return output

    # Xử lý từng cụm mặt nạ liên thông một lần duy nhất
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        roi = image[y:y + h, x:x + w]
        small = cv2.resize(roi, (max(1, w // pixel_size), max(1, h // pixel_size)), interpolation=cv2.INTER_AREA)
        if mode == "blur":
            small = cv2.GaussianBlur(small, (0, 0), 3)
        effect = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST if mode == "pixelate" else cv2.INTER_LINEAR)
        if area == w * h:
            output[y:y + h, x:x + w] = effect
        else:
            selected = (labels[y:y + h, x:x + w] == index)[..., None]
            np.copyto(output[y:y + h, x:x + w], effect, where=selected)
    return output
