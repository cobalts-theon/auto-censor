import cv2
import numpy as np

def redact(image: np.ndarray, detections, mode: str = "black", pixel_size: int = 16) -> np.ndarray:
    if mode not in ("black", "blur", "pixelate"):
        raise ValueError("Invalid mode")
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
    # Process connected mask components once, always sampling the original image.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        roi = image[y:y+h, x:x+w]
        small = cv2.resize(roi, (max(1, w // pixel_size), max(1, h // pixel_size)), interpolation=cv2.INTER_AREA)
        if mode == "blur":
            small = cv2.GaussianBlur(small, (0, 0), 3)
        effect = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST if mode == "pixelate" else cv2.INTER_LINEAR)
        if area == w * h:
            output[y:y+h, x:x+w] = effect
        else:
            selected = (labels[y:y+h, x:x+w] == index)[..., None]
            np.copyto(output[y:y+h, x:x+w], effect, where=selected)
    return output
