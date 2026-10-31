"""Interactive face privacy experiment. Press 1–4 to change modes, Q to quit."""
import argparse
import csv
from pathlib import Path
from time import perf_counter
import cv2
from app.processing.anonymizer import Mode, anonymize
from app.detectors.face_detector import FaceDetector
from app.detectors.face_tracker import FaceTracker
from app.camera.webcam import open_webcam

BASE_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=1)
    parser.add_argument("--mode", choices=[m.value for m in Mode], default=Mode.BLUR.value)
    parser.add_argument("--log", type=Path, help="CSV metrics file (no images)")
    args = parser.parse_args()
    detector = FaceDetector(BASE_DIR / "models/face_detection_yunet_2026may.onnx")
    tracker = FaceTracker()
    mode = Mode(args.mode)
    cap = open_webcam(args.camera)
    log = None
    fps, frames, frame_id = 0.0, 0, 0
    started = fps_started = perf_counter()
    print("1: Detect Only | 2: Blur | 3: Pixelate | 4: Blackout | Q: thoát")
    try:
        writer = None
        if args.log:
            args.log.parent.mkdir(parents=True, exist_ok=True)
            log = args.log.open("w", newline="", encoding="utf-8")
            writer = csv.writer(log)
            writer.writerow(["frame", "elapsed_s", "mode", "faces", "min_confidence", "fps", "inference_ms", "tracked_faces", "detected_faces"])
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Không đọc được frame!")
            detections, inference_ms = detector.detect(frame)
            faces = tracker.update(frame, detections, perf_counter())
            anonymize(frame, faces, mode)
            for face in faces:
                x1, y1, x2, y2 = face.box
                cv2.rectangle(frame, (x1, y1), (x2-1, y2-1), (0, 255, 0), 2)
                label = (f"Confidence: {face.confidence:.1%}" if face.source == "Detected"
                         else f"{face.source} | Last conf: {face.confidence:.1%}")
                cv2.putText(frame, label,
                            (x1, max(20, y1-8)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)
            frames += 1
            frame_id += 1
            now = perf_counter()
            if now - fps_started >= 1:
                fps = frames / (now - fps_started)
                frames, fps_started = 0, now
            cv2.putText(frame, f"{mode.value} | Faces: {len(faces)} | 1/2/3/4: mode | Q: quit",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"FPS: {fps:.1f} | YuNet: {inference_ms:.1f} ms",
                        (10, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 255), 2)
            if writer:
                writer.writerow([frame_id, round(now-started, 4), mode.value, len(faces),
                                 min((f.confidence for f in detections), default=""),
                                 round(fps, 2), round(inference_ms, 3),
                                 sum(f.source != "Detected" for f in faces), len(detections)])
            cv2.imshow("Privacy Vision", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            modes = {ord(str(i)): m for i, m in enumerate(Mode, 1)}
            mode = modes.get(key, mode)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if log:
            log.close()


if __name__ == "__main__":
    main()
