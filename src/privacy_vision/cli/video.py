"""Giao diện dòng lệnh xử lý video tệp tin (Video Redaction CLI)."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import cv2

from ..config import PipelineConfig, DetectorConfig, MODEL_FILES
from ..pipeline import PrivacyPipeline
from ..processing.tracking import RegionTracker
from ..image_io import atomic_write


def parser():
    p = argparse.ArgumentParser(description="PrivacyVision: Xử lý che thông tin nhạy cảm trong file video (Offline/Batch)")
    p.add_argument('--input', required=True, type=Path, help='File video đầu vào (.mp4, .avi, .mov, .mkv)')
    p.add_argument('--output', required=True, type=Path, help='File video đầu ra đã che')
    p.add_argument('--mode', choices=['pixelate', 'black', 'blur'], default='pixelate', help='Chế độ che (mặc định: pixelate)')
    p.add_argument('--device', default='auto', help='auto, cpu hoặc CUDA index (0)')
    p.add_argument('--models-dir', type=Path, default=Path('models'))
    p.add_argument('--enable', default='face,plate,document', help='Các detector kích hoạt, cách nhau bởi dấu phẩy (face, plate, document, card, cccd)')
    p.add_argument('--conf-face', type=float, default=0.65, help='Ngưỡng tin cậy khuôn mặt (mặc định: 0.65)')
    p.add_argument('--conf-plate', type=float, default=0.50, help='Ngưỡng tin cậy biển số (mặc định: 0.50)')
    p.add_argument('--conf-doc', type=float, default=0.20, help='Ngưỡng tin cậy giấy tờ (mặc định: 0.20)')
    p.add_argument('--no-tracking', action='store_true', help='Tắt tracking quang học')
    p.add_argument('--card-only', action='store_true', help='Chỉ nhận diện card_document')
    p.add_argument('--rotate', action='store_true', help='Bật thử góc xoay 90 độ cho giấy tờ (mặc định: tắt để tối ưu tốc độ)')
    p.add_argument('--max-frames', type=int, default=None, help='Chỉ xử lý tối đa N frame đầu tiên (dùng để test nhanh)')
    p.add_argument('--preview', action='store_true', help='Hiển thị cửa sổ xem trực tiếp trong lúc xử lý')
    p.add_argument('--metadata-output', type=Path, help='Lưu thống kê chi tiết ra file JSON')
    return p


def build_video_pipeline(args):
    enabled = [s.strip() for s in args.enable.split(',') if s.strip()]
    detectors = []

    if 'face' in enabled:
        face_model = args.models_dir / "face_detection_yunet_2026may.onnx"
        if face_model.is_file():
            detectors.append(DetectorConfig("face", face_model, confidence=args.conf_face, imgsz=480))

    if 'plate' in enabled:
        plate_model = args.models_dir / "plate_best.pt"
        if plate_model.is_file():
            detectors.append(DetectorConfig("plate", plate_model, confidence=args.conf_plate))

    # Legacy card / cccd models if specified
    for legacy_src in ('card', 'cccd'):
        if legacy_src in enabled and legacy_src in MODEL_FILES:
            legacy_model = args.models_dir / MODEL_FILES[legacy_src]
            if legacy_model.is_file():
                detectors.append(DetectorConfig(legacy_src, legacy_model, confidence=0.25))

    doc_checkpoint = args.models_dir / "document_best.pt"
    enable_doc = 'document' in enabled and doc_checkpoint.is_file()
    doc_classes = ("card_document",) if args.card_only else None
    doc_cfg = DetectorConfig("document", doc_checkpoint, confidence=args.conf_doc, classes=doc_classes) if enable_doc else None

    rotations = (0, 90) if args.rotate else (0,)

    config = PipelineConfig(
        detectors=detectors,
        device=args.device,
        mode=args.mode,
        whole_documents=enable_doc,
        document_rotations=rotations,
        document_detection_strategy="detector" if enable_doc else "heuristic",
        document_detector_config=doc_cfg,
        max_document_area_ratio=0.98,
    )
    return PrivacyPipeline(config)


def _default_progress_bar(current, total, fps, elapsed):
    if total > 0:
        pct = min(100.0, current / total * 100)
        bar_len = 30
        filled = int(bar_len * current // total)
        bar = "=" * filled + (">" if filled < bar_len else "") + " " * (bar_len - filled - (1 if filled < bar_len else 0))
        rem_frames = max(0, total - current)
        eta_sec = int(rem_frames / fps) if fps > 0 else 0
        m, s = divmod(eta_sec, 60)
        eta_str = f"{m:02d}:{s:02d}"
        print(f"\r  [{bar}] {pct:5.1f}% ({current}/{total}) | {fps:5.1f} FPS | ETA: {eta_str} ", end="", flush=True)
    else:
        m, s = divmod(int(elapsed), 60)
        print(f"\r  Đã xử lý: {current} frames | {fps:5.1f} FPS | Thời gian: {m:02d}:{s:02d} ", end="", flush=True)


def process_video(input_path, output_path, pipeline, tracker=None, max_frames=None, preview=False, progress_callback=None):
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file video đầu vào: {input_path}")

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Không thể mở file video: {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or not (1 <= fps <= 120):
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames is not None and max_frames > 0:
        total_frames = min(total_frames, max_frames) if total_frames > 0 else max_frames

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise ValueError(f"Không thể khởi tạo VideoWriter cho đường dẫn: {output_path}")

    frame_idx = 0
    start_time = perf_counter()
    counts = {"face": 0, "plate": 0, "document": 0}

    preview_win = "PrivacyVision - Video Processing"
    if preview:
        cv2.namedWindow(preview_win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(preview_win, min(1280, width), min(720, height))

    try:
        while True:
            if max_frames is not None and frame_idx >= max_frames:
                break

            ok, frame = cap.read()
            if not ok or frame is None:
                break

            frame_idx += 1
            result = pipeline.process(frame, tracker=tracker)
            writer.write(result.image)

            for d in result.detections:
                src = d.source if d.source in counts else ("document" if d.class_name == "document_region" else None)
                if src:
                    counts[src] += 1

            now = perf_counter()
            elapsed = now - start_time
            proc_fps = frame_idx / elapsed if elapsed > 0 else 0

            # Progress bar
            if progress_callback:
                try:
                    stop = progress_callback(frame_idx, total_frames, proc_fps, elapsed, result.image)
                except TypeError:
                    stop = progress_callback(frame_idx, total_frames, proc_fps, elapsed)
                if stop is False:
                    print("\n[INFO] Tiến trình xử lý video bị dừng.")
                    break
            else:
                _default_progress_bar(frame_idx, total_frames, proc_fps, elapsed)

            if preview:
                cv2.imshow(preview_win, result.image)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q'), ord('Q')):
                    print("\n[INFO] Người dùng đã dừng xử lý sớm.")
                    break
    finally:
        cap.release()
        writer.release()
        if preview:
            cv2.destroyAllWindows()

    total_time = perf_counter() - start_time
    avg_fps = frame_idx / total_time if total_time > 0 else 0
    print()  # Xuống dòng sau progress bar

    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "resolution": [width, height],
        "fps": fps,
        "frames_processed": frame_idx,
        "total_frames": total_frames,
        "total_time_seconds": round(total_time, 2),
        "processing_fps": round(avg_fps, 1),
        "counts": counts,
    }


def main():
    p = parser()
    args = p.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.is_file():
        p.exit(1, f"Lỗi: Không tìm thấy file video đầu vào: {input_path}\n")
    if input_path.resolve() == output_path.resolve():
        p.exit(1, "Lỗi: File video đầu vào và đầu ra không được trùng nhau.\n")

    print(f"\n==================== PRIVACY VISION - VIDEO REDACTION ====================")
    print(f"  Input        : {input_path}")
    print(f"  Output       : {output_path}")
    print(f"  Mode         : {args.mode.upper()}")
    print(f"  Device       : {args.device}")
    print(f"  Tracking     : {'TẮT' if args.no_tracking else 'BẬT'}")
    print(f"==========================================================================\n")

    print("1. Đang nạp các mô hình AI và khởi tạo pipeline...")
    pipeline = build_video_pipeline(args)
    tracker = RegionTracker(hold_seconds=0.8, document_hold_seconds=1.0) if not args.no_tracking else None

    print("2. Bắt đầu xử lý video...")
    report = process_video(
        input_path, output_path, pipeline,
        tracker=tracker, max_frames=args.max_frames, preview=args.preview
    )

    print("\n========================= KẾT QUẢ XỬ LÝ =========================")
    print(f"  Tổng frames đã xử lý : {report['frames_processed']}")
    print(f"  Thời gian hoàn thành : {report['total_time_seconds']} giây")
    print(f"  Tốc độ xử lý TB      : {report['processing_fps']} FPS")
    print(f"  Số lượng phát hiện   : Mặt: {report['counts']['face']} | Biển số: {report['counts']['plate']} | Giấy tờ: {report['counts']['document']}")
    print(f"  File video đã lưu tại: {output_path}")
    print("=================================================================\n")

    if args.metadata_output:
        atomic_write(args.metadata_output, json.dumps(report, ensure_ascii=False, indent=2).encode())
        print(f"Đã lưu metadata thống kê tại: {args.metadata_output}")


if __name__ == '__main__':
    main()
