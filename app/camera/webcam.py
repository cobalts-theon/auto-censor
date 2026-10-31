"""Capture webcam frames at the requested resolution."""
import cv2


def open_webcam(camera_id=1, width=1280, height=720):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Không thể mở webcam {camera_id}!")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap
