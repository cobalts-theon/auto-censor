#!/usr/bin/env python3
"""Interactive HD camera test with Face + Plate + Document (document_best.pt).

High-Performance Pipeline:
- Native Camera Auto-Exposure: Sáng rõ tự nhiên.
- 16x16 Pixelate Redaction: Tối ưu numpy SIMD, siêu nhẹ (< 3.5ms).
- Fast Motion & Motion Blur Resistance:
  * Optical Flow Search Window (31x31, Level 4) bắt kịp chuyển động 80-100 px/frame.
  * Continuous point replenishment duy trì bám vết liên tục khi lướt nhanh.
  * Velocity-guided hold (1.0s) bám theo hướng di chuyển kể cả khi nhòe ảnh.
- Full Close-up Support: Che cả khi đưa sát camera (lên tới 98% màn hình).
- All Documents Enabled By Default: Nhận diện cả Thẻ (CCCD/ATM) và Giấy tờ tài liệu A4.
- License Plate Detection Enabled By Default: Che cả biển số xe (bấm 'p' để bật/tắt).
- Pre-warm GPU CUDA: Loại bỏ hoàn toàn lỗi đơ popup "Force Quit / Wait" của GNOME.
- Native 30 FPS.
"""
import argparse
from datetime import datetime
from pathlib import Path
import sys
from threading import Thread, Condition
from time import perf_counter, sleep
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from privacy_vision.config import PipelineConfig, DetectorConfig
from privacy_vision.pipeline import PrivacyPipeline
from privacy_vision.processing.tracking import RegionTracker
from privacy_vision.desktop.camera import open_camera, CameraOptions
from privacy_vision.virtualcam import check_virtual_cam_support, VirtualCamBroadcaster


class ThreadedCamera:
    """Đọc luồng camera nền non-blocking với 4 buffer để tối đa hóa FPS."""
    def __init__(self, cap):
        self.cap = cap
        self.frame = None
        self.ok = False
        self.running = True
        self.new_frame = False
        self.cond = Condition()
        self.thread = Thread(target=self._worker, name="camera-reader", daemon=True)
        self.thread.start()
        with self.cond:
            if not self.ok:
                self.cond.wait(timeout=1.5)

    def _worker(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                sleep(0.002)
                continue
            with self.cond:
                self.frame = frame
                self.ok = True
                self.new_frame = True
                self.cond.notify_all()

    def read(self, wait_new=True, timeout=0.04):
        with self.cond:
            if wait_new and not self.new_frame:
                self.cond.wait(timeout=timeout)
            self.new_frame = False
            return self.ok, self.frame

    def release(self):
        self.running = False
        self.thread.join(timeout=0.5)
        self.cap.release()


def build_pipeline(models_dir, enable_face, enable_plate, enable_doc, card_only, rotate, conf_face, conf_plate, conf_doc, mode="pixelate", device="auto", max_area=0.95):
    detectors = []
    if enable_face:
        # imgsz=480 chạy cực nhanh trên CPU (~3.3ms) mà vẫn bắt mặt nhạy
        detectors.append(DetectorConfig("face", models_dir / "face_detection_yunet_2026may.onnx", confidence=conf_face, imgsz=480))
    if enable_plate:
        detectors.append(DetectorConfig("plate", models_dir / "plate_best.pt", confidence=conf_plate))

    doc_checkpoint = models_dir / "document_best.pt"
    # Mặc định nhận diện CẢ THẺ (card_document) VÀ TÀI LIỆU (paper_document)
    doc_classes = ("card_document",) if card_only else None
    doc_cfg = DetectorConfig("document", doc_checkpoint, confidence=conf_doc, classes=doc_classes) if (enable_doc and doc_checkpoint.is_file()) else None

    # (0, 90) Adaptive Early Exit
    rotations = (0, 90) if rotate else (0,)

    config = PipelineConfig(
        detectors=detectors,
        device=device,
        mode=mode,
        whole_documents=enable_doc,
        document_rotations=rotations,
        document_detection_strategy="detector" if enable_doc else "heuristic",
        document_detector_config=doc_cfg,
        max_document_area_ratio=max_area,
    )
    return PrivacyPipeline(config)


def main():
    parser = argparse.ArgumentParser(description="Test webcam HD 720p Pixelate (Face + Plate + Document)")
    parser.add_argument("--camera", type=int, default=1, help="Chỉ số camera (mặc định: 1)")
    parser.add_argument("--device", default="auto", help="auto, cpu, hoặc CUDA index")
    parser.add_argument("--width", type=int, default=1280, help="Độ phân giải ngang (mặc định: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Độ phân giải dọc (mặc định: 720)")
    parser.add_argument("--fps", type=int, default=30, help="Tốc độ khung hình yêu cầu (mặc định: 30)")
    parser.add_argument("--conf-face", type=float, default=0.65)
    parser.add_argument("--conf-plate", type=float, default=0.50)
    parser.add_argument("--conf-doc", type=float, default=0.20, help="Ngưỡng tin cậy giấy tờ (mặc định: 0.20)")
    parser.add_argument("--no-plate", action="store_true", default=False, help="Tắt detector biển số")
    parser.add_argument("--with-plate", dest="no_plate", action="store_false", help="Bật detector biển số (mặc định: BẬT)")
    parser.add_argument("--card-only", action="store_true", default=False, help="Chỉ nhận diện card_document")
    parser.add_argument("--no-rotate", action="store_true", help="Tắt xoay (chỉ chạy 1 góc 0°)")
    parser.add_argument("--no-tracking", action="store_true", help="Tắt tracking")
    parser.add_argument("--debug-boxes", action="store_true", help="Hiển thị viền và thông tin model")
    parser.add_argument("--max-area", type=float, default=0.98, help="Tỷ lệ diện tích tối đa (mặc định: 0.98)")
    parser.add_argument("--virtual-cam", action="store_true", help="Bật phát Virtual Camera (v4l2loopback/pyvirtualcam) cho Google Meet/Zoom")
    parser.add_argument("--vcam-device", default=None, help="Thiết bị webcam ảo (mặc định: auto-detect hoặc /dev/video0)")
    args = parser.parse_args()

    models_dir = PROJECT_ROOT / "models"
    enable_face = True
    enable_plate = not args.no_plate
    enable_doc = True
    card_only = args.card_only
    rotate = not args.no_rotate
    use_tracking = not args.no_tracking
    show_boxes = args.debug_boxes
    conf_doc = args.conf_doc

    vcam_support = check_virtual_cam_support()
    broadcaster = VirtualCamBroadcaster(
        device=args.vcam_device or vcam_support.get('device'),
        fps=args.fps,
        enabled=args.virtual_cam
    )

    print(f"1. Đang mở camera {args.width}x{args.height} (HD 720p @ {args.fps}fps, Auto Exposure)...")
    options = CameraOptions(fps=args.fps, width=args.width, height=args.height, fourcc="MJPG",
                            manual_exposure=False, buffers=4)
    raw_cap = open_camera(args.camera, options)
    cap = ThreadedCamera(raw_cap)

    print("2. Đang nạp model & khởi động GPU CUDA...")
    pipeline = build_pipeline(
        models_dir, enable_face, enable_plate, enable_doc, card_only, rotate,
        args.conf_face, args.conf_plate, conf_doc, mode="pixelate", device=args.device, max_area=args.max_area
    )

    # Warmup PyTorch và CUDA kernels TRƯỚC KHI tạo cửa sổ để tránh pop-up 'Force Quit / Wait'
    dummy = np.zeros((args.height, args.width, 3), dtype=np.uint8)
    pipeline.process(dummy)
    pipeline.process(dummy)

    # Sticky tracking: 1.0s hold time + Velocity-guided extrapolation + Fast-motion Optical Flow
    tracker = RegionTracker(hold_seconds=0.8, document_hold_seconds=1.0) if use_tracking else None

    snapshots_dir = PROJECT_ROOT / "outputs" / "snapshots"
    recordings_dir = PROJECT_ROOT / "outputs" / "recordings"
    is_recording = False
    video_writer = None
    record_start_time = 0.0
    record_frames = 0
    record_path = None
    banner_msg = ""
    banner_until = 0.0
    banner_color = (0, 255, 0)

    print("\n======================= PHÍM TẮT ĐIỀU KHIỂN =======================")
    print("  's' : CHỤP ẢNH (Snapshot -> outputs/snapshots/)")
    print("  'v' : BẮT ĐẦU / DỪNG GHI VIDEO (outputs/recordings/)")
    print("  'm' : BẬT/TẮT Virtual Camera (Google Meet / Zoom)")
    print(f"        Trạng thái: {vcam_support.get('message', '')}")
    print("  'p' : BẬT/TẮT Biển số xe (Plate)")
    print("  'c' : BẬT/TẮT Card-only (Đang: TẤT CẢ GIẤY TỜ + THẺ)")
    print("  'r' : BẬT/TẮT Thử góc xoay (Adaptive 0° & 90°)")
    print("  'b' : BẬT/TẮT viền Debug (Xem tên model và % confidence của từng hộp)")
    print("  't' : BẬT/TẮT Tracking")
    print("  '+' / '-' : Tăng/giảm ngưỡng tin cậy giấy tờ (+-0.05)")
    print("  'q' hoặc ESC: Thoát")
    print("===================================================================\n")

    print("3. Khởi tạo cửa sổ hiển thị...")
    window_name = "PrivacyVision HD 720p - Pixelate"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    cv2.waitKey(1)

    fps, frames, tick = 0.0, 0, perf_counter()

    try:
        while True:
            ok, frame = cap.read(wait_new=True)
            if not ok or frame is None:
                sleep(0.005)
                continue

            t0 = perf_counter()
            active_tracker = tracker if use_tracking else None
            result = pipeline.process(frame, tracker=active_tracker)
            latency_ms = (perf_counter() - t0) * 1000

            # Phát Virtual Camera (Meet / Zoom) nếu đang bật
            broadcaster.broadcast_frame(result.image, fps=int(fps) if fps > 0 else args.fps)

            # Ghi video nền nếu đang kích hoạt chế độ REC
            if is_recording and video_writer is not None:
                video_writer.write(result.image)
                record_frames += 1

            frames += 1
            now = perf_counter()
            if now - tick >= 1.0:
                fps = frames / (now - tick)
                frames, tick = 0, now

            display_img = result.image.copy()

            counts = {"face": 0, "plate": 0, "document": 0}
            for d in result.detections:
                src = d.source if d.source in counts else ("document" if d.class_name == "document_region" else None)
                if src:
                    counts[src] += 1

            if show_boxes:
                colors = {
                    "face": (0, 255, 0),        # Xanh lá
                    "plate": (255, 200, 0),     # Vàng
                    "document": (0, 0, 255),    # Đỏ
                }
                for d in result.detections:
                    src = d.source if d.source in colors else "document"
                    color = colors.get(src, (255, 255, 255))
                    x1, y1, x2, y2 = d.bbox
                    cv2.rectangle(display_img, (x1, y1), (x2, y2), color, 2)
                    state = d.metadata.get("tracking_state", "live")
                    rot = d.metadata.get("rotation_degrees", 0)
                    rot_str = f" r={rot}°" if rot else ""
                    label = f"{d.class_name} ({d.confidence:.2f}){rot_str} [{state}]"
                    cv2.putText(display_img, label, (x1, max(24, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # HUD hiển thị thông số
            rotate_str = "Adaptive 0° & 90°" if rotate else "1 GÓC (0°)"
            doc_mode_label = "CHỈ THẺ" if card_only else "TẤT CẢ GIẤY TỜ + THẺ"
            vcam_str = f"LIVE ({broadcaster.device_name})" if broadcaster.is_active else ("BAT" if broadcaster.enabled else "OFF")
            hud_1 = f"FPS: {fps:.1f} | Latency: {latency_ms:.0f}ms | Mode: [PIXELATE] | Res: {frame.shape[1]}x{frame.shape[0]}"
            hud_2 = f"Rotate (r): [{rotate_str}] | Mode (c): [{doc_mode_label}] | Plate (p): [{'ON' if enable_plate else 'OFF'}] | Conf: {conf_doc:.2f}"
            hud_3 = f"Detections: Face: {counts['face']} | Doc: {counts['document']} | Plate: {counts['plate']} | VCam (m): [{vcam_str}]"

            for text, y in [(hud_1, 30), (hud_2, 60), (hud_3, 90)]:
                cv2.putText(display_img, text, (12, y + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
                cv2.putText(display_img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if y == 30 else ((255, 255, 0) if y == 60 else (0, 255, 255)), 2)

            # Chỉ báo Recording góc trên bên phải
            if is_recording:
                elapsed_sec = int(perf_counter() - record_start_time)
                m, s = divmod(elapsed_sec, 60)
                rec_label = f"REC {m:02d}:{s:02d} ({record_frames}f)"
                rx = display_img.shape[1] - 250
                # Chấm đỏ nhấp nháy mỗi 0.5s
                if int(perf_counter() * 2) % 2 == 0:
                    cv2.circle(display_img, (rx - 15, 30), 8, (0, 0, 255), -1)
                cv2.putText(display_img, rec_label, (rx + 2, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
                cv2.putText(display_img, rec_label, (rx, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Banner thông báo tạm thời ở cạnh dưới
            if perf_counter() < banner_until and banner_msg:
                by = display_img.shape[0] - 25
                cv2.putText(display_img, banner_msg, (22, by + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
                cv2.putText(display_img, banner_msg, (20, by), cv2.FONT_HERSHEY_SIMPLEX, 0.7, banner_color, 2)

            cv2.imshow(window_name, display_img)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                break
            elif key in (ord('s'), ord('S')):
                snapshots_dir.mkdir(parents=True, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                snap_path = snapshots_dir / f"snapshot_{ts}.png"
                cv2.imwrite(str(snap_path), result.image)
                banner_msg = f"DA LUU SNAPSHOT: {snap_path.name}"
                banner_until = perf_counter() + 2.5
                banner_color = (0, 255, 0)
                print(f"\n[SNAPSHOT] Đã chụp và lưu ảnh: {snap_path}")
            elif key in (ord('v'), ord('V')):
                if not is_recording:
                    recordings_dir.mkdir(parents=True, exist_ok=True)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    record_path = recordings_dir / f"record_{ts}.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    rec_fps = max(15.0, fps) if fps > 5.0 else float(args.fps)
                    video_writer = cv2.VideoWriter(str(record_path), fourcc, rec_fps, (frame.shape[1], frame.shape[0]))
                    is_recording = True
                    record_start_time = perf_counter()
                    record_frames = 0
                    banner_msg = "BAT DAU GHI VIDEO..."
                    banner_until = perf_counter() + 2.0
                    banner_color = (0, 0, 255)
                    print(f"\n[RECORD] Bắt đầu ghi video: {record_path} ({rec_fps:.1f} FPS)")
                else:
                    is_recording = False
                    if video_writer is not None:
                        video_writer.release()
                        video_writer = None
                    dur = perf_counter() - record_start_time
                    banner_msg = f"DA LUU VIDEO: {record_path.name} ({record_frames}f, {dur:.1f}s)"
                    banner_until = perf_counter() + 3.0
                    banner_color = (0, 255, 255)
                    print(f"\n[RECORD] Đã kết thúc và lưu video: {record_path} ({record_frames} frames, {dur:.1f}s)")
            elif key in (ord('p'), ord('P')):
                enable_plate = not enable_plate
                pipeline = build_pipeline(models_dir, enable_face, enable_plate, enable_doc, card_only, rotate,
                                         args.conf_face, args.conf_plate, conf_doc, mode="pixelate", device=args.device, max_area=args.max_area)
            elif key in (ord('m'), ord('M')):
                broadcaster.enabled = not broadcaster.enabled
                if broadcaster.enabled:
                    banner_msg = f"BAT VIRTUAL CAM: {broadcaster.device_name}"
                    banner_color = (0, 255, 0)
                    print(f"\n[VCAM] Đã bật Virtual Camera: {broadcaster.device_name}")
                else:
                    banner_msg = "DA TAT VIRTUAL CAM"
                    banner_color = (0, 0, 255)
                    print("\n[VCAM] Đã tắt Virtual Camera")
                banner_until = perf_counter() + 2.5
            elif key in (ord('r'), ord('R')):
                rotate = not rotate
                pipeline = build_pipeline(models_dir, enable_face, enable_plate, enable_doc, card_only, rotate,
                                         args.conf_face, args.conf_plate, conf_doc, mode="pixelate", device=args.device, max_area=args.max_area)
            elif key in (ord('c'), ord('C')):
                card_only = not card_only
                pipeline = build_pipeline(models_dir, enable_face, enable_plate, enable_doc, card_only, rotate,
                                         args.conf_face, args.conf_plate, conf_doc, mode="pixelate", device=args.device, max_area=args.max_area)
            elif key in (ord('b'), ord('B')):
                show_boxes = not show_boxes
            elif key in (ord('t'), ord('T')):
                use_tracking = not use_tracking
                if not use_tracking and tracker is not None:
                    tracker.reset()
            elif key in (ord('+'), ord('=')):
                conf_doc = min(0.95, round(conf_doc + 0.05, 2))
                pipeline = build_pipeline(models_dir, enable_face, enable_plate, enable_doc, card_only, rotate,
                                         args.conf_face, args.conf_plate, conf_doc, mode="pixelate", device=args.device, max_area=args.max_area)
            elif key in (ord('-'), ord('_')):
                conf_doc = max(0.10, round(conf_doc - 0.05, 2))
                pipeline = build_pipeline(models_dir, enable_face, enable_plate, enable_doc, card_only, rotate,
                                         args.conf_face, args.conf_plate, conf_doc, mode="pixelate", device=args.device, max_area=args.max_area)
    finally:
        broadcaster.stop()
        if is_recording and video_writer is not None:
            video_writer.release()
            print(f"\n[RECORD] Đã lưu video trước khi thoát: {record_path}")
        cap.release()
        cv2.destroyAllWindows()
        print("Đã đóng camera test.")


if __name__ == "__main__":
    main()
