# Báo cáo nâng cấp Document Redaction

Ngày kiểm tra: 2026-11-04. Không train, tải hoặc thay checkpoint. Giữ thay đổi sẵn trong working tree, không reset các file cũ.

## File sửa/tạo trong đợt nâng cấp

- Core: src/privacy_vision/config.py, schemas.py, pipeline.py.
- Processing: src/privacy_vision/processing/documents.py, tracking.py.
- Detector mới: src/privacy_vision/detectors/document.py.
- CLI: src/privacy_vision/cli/image.py.
- Desktop: src/privacy_vision/desktop/capture.py, session.py, window.py.
- Công cụ mới: scripts/benchmark_webcam.py, scripts/evaluate_documents.py.
- Test mới: tests/core/test_document_safety.py, tests/core/test_evaluation_metrics.py.
- Test cập nhật: tests/desktop/test_desktop.py, test_latest_capture.py, test_webcam_session.py; tests/integration/test_core_smoke.py.
- Tài liệu: README.md, configs/example.toml, docs/architecture.md, testing.md; thêm docs/document-protection.md, evaluation.md và báo cáo này.

## Hành vi và an toàn

Camera → latest raw slot trong worker → detector → grouping/DocumentRegion → tracking → union mask/redaction → validation ảnh/timing → processed preview slot → GUI.

Fields giữ che field/mặt/biển số. Full thêm vùng giấy; field trong/ngoài vùng giấy đều được giữ. Heuristic spatial grouping hỗ trợ nhiều cụm, bằng chứng trùng hai model không tăng số loại field; title không phải bbox nguyên thẻ. Margin chỉ là xấp xỉ, không xác nhận đã kín mép giấy. Confidence heuristic là None trong DocumentRegion.

Webcam lỗi capture/load/AI/CUDA/redaction/output xóa preview/dừng phiên; log chỉ stage và exception type. Kết quả dùng chung vùng nhớ raw bị chặn. GUI chỉ nhận đúng session, preview <=2 giây. Queue một frame, frame cũ bị thay thế khi AI chậm. Native call treo chưa thể hủy, nhưng GUI vẫn xóa preview quá hạn và event loop còn phản hồi.

Strict bật rõ ràng: đen khi không detection, thiếu document detector evidence, heuristic hoặc track mất xác nhận. Full+heuristic+strict luôn đen. Không có trạng thái Protected tự động: coverage_unverified kể cả khi document detector thật trả bbox. Các checkpoint toàn giấy chưa tồn tại trong tích hợp mặc định; adapter không phải tính năng detector đã được chứng minh thực tế.

## Kiểm thử

```bash
PRIVACYVISION_SMOKE_IMAGE=inputs/test.jpg .venv/bin/python -m pytest -q
```

Lượt toàn bộ cuối: **110 passed, 4 subtests passed, 4 warnings trong 7.20 giây**, không skip khi bật ảnh smoke. Bốn warnings là NVML không khởi tạo được. Smoke dùng bốn checkpoint thật trên một ảnh cục bộ, kiểm tra fields/full/full+strict, không xuất ảnh. Smoke không đo recall hoặc khẳng định che kín giấy. Unit/Qt dùng mock/synthetic, bao gồm 100 capture frames không tiêu thụ (99 frame bị thay), session mismatch, stale preview, lỗi ảnh/timing, raw storage fallback, lỗi detector/redaction, checkpoint/taxonomy và tracking expiry/reset.

CLI predict_image, benchmark_webcam, evaluate_documents đã kiểm tra --help; compileall src/scripts thành công. Có cảnh báo NVML từ PyTorch và half deprecated từ Ultralytics; không thay dependency trong nhiệm vụ này.

## Benchmark thực tế

Nguồn: inputs/test.jpg có sẵn, CPU, full/heuristic, black, strict tắt, face/plate 640; không tracking, không tính tải model hoặc I/O. Mỗi cấu hình 3 warmup, 30 runs. Hai cấu hình chạy lần lượt sau smoke test.

| Card/CCCD imgsz | Mean pipeline | P50 | P95 |
| --- | --- | --- | --- |
| 640 | 120.07 ms | 118.89 ms | 132.03 ms |
| 800 | 156.41 ms | 156.94 ms | 165.45 ms |

Không thấy /dev/video*, nvidia-smi không giao tiếp được NVIDIA driver. Vì vậy chưa đo webcam/GPU/VRAM, chưa xác minh FPS RTX 2050. Không quy đổi các số ảnh tĩnh thành FPS webcam. Xem docs/document-protection.md để chạy lại công cụ khi có camera/driver.

## Hạn chế và bước tiếp theo

Chưa có dataset webcam gán nhãn độc lập; chưa xác nhận loại giấy nào đạt mức đáng tin cậy bằng quy trình đánh giá mới. Giấy khai sinh/A4 chỉ là mục tiêu taxonomy. Heuristic có thể bỏ sót/gom nhầm; tracking có thể lệch; strict có thể chặn cảnh trống và vẫn không chứng minh không bỏ sót giấy khác khi cho frame detector qua. Blur/pixelate không bảo đảm chống khôi phục.

Trước khi dùng Document Detector thật: chuẩn bị dataset toàn giấy đa điều kiện, taxonomy thống nhất, tách test theo người/giấy/chuỗi; train ngoài nhiệm vụ này; kiểm tra nguồn checkpoint tin cậy; đo recall/FN/FP và mask coverage riêng, latency/VRAM/webcam trên máy đích. Sau đó chọn checkpoint trong GUI hoặc --document-checkpoint và đánh giá detector/hybrid độc lập. Không dùng mAP/IoU thay cho sensitive coverage. Hướng dẫn manifest/chỉ số tại docs/evaluation.md.
