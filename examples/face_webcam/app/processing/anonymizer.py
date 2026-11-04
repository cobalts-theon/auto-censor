"""Apply the selected effect in place to padded face boxes."""
from enum import Enum
import cv2


class Mode(str, Enum):
    DETECT_ONLY = "Detect Only"
    BLUR = "Blur"
    PIXELATE = "Pixelate"
    BLACKOUT = "Blackout"


def anonymize(frame, faces, mode=Mode.BLUR):
    mode = Mode(mode)
    if mode == Mode.DETECT_ONLY:
        return frame
    h, w = frame.shape[:2]
    for face in faces:
        x1, y1, x2, y2 = face.box
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        roi = frame[y1:y2, x1:x2]
        if mode == Mode.BLUR:
            # Bound Gaussian work independently of the original face size.
            # Scale sigma with the reduced image to retain a strong blur.
            rw, rh = x2-x1, y2-y1
            scale = min(1.0, 64 / max(rw, rh))
            small = cv2.resize(
                roi, (max(1, round(rw*scale)), max(1, round(rh*scale))),
                interpolation=cv2.INTER_AREA,
            )
            blurred = cv2.GaussianBlur(
                small, (0, 0), max(3, min(rw, rh) / 6) * scale,
            )
            roi[:] = cv2.resize(blurred, (rw, rh), interpolation=cv2.INTER_LINEAR)
        elif mode == Mode.PIXELATE:
            small = cv2.resize(roi, (max(1, round(16*(x2-x1)/max(x2-x1, y2-y1))),
                                      max(1, round(16*(y2-y1)/max(x2-x1, y2-y1)))),
                               interpolation=cv2.INTER_AREA)
            roi[:] = cv2.resize(small, (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)
        else:
            roi[:] = 0
    return frame
