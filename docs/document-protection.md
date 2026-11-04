# Cấu hình bảo vệ giấy tờ

`document_redaction_mode="fields"` che field, mặt và biển số đã phát hiện. `"full"` thêm vùng giấy, không bỏ field bên ngoài hoặc bên trong vùng. `mode="black"|"blur"|"pixelate"` độc lập với lựa chọn này. Ưu tiên black và PNG khi cần che nội dung.

API/CLI mặc định fields; GUI giữ mặc định full của phiên bản trước. `whole_documents=True` / `--whole-documents` là alias full. Alias full kết hợp fields rõ ràng là lỗi. Tham số mới được thêm sau các tham số API cũ; `PipelineResult(image, detections, timings_ms)` vẫn hợp lệ.

- `document_detection_strategy`: heuristic (mặc định), detector, hybrid.
- `document_margin=(0.25, 0.30)`: tỷ lệ mở rộng mỗi phía ngang/dọc; mỗi giá trị trong [0, 0.5]. Không chứng minh bao hết mép giấy.
- `document_min_fields=3`: số loại field khác nhau; hai loại có hint vẫn đủ theo quy tắc cũ. Field trùng hai model không tăng bằng chứng.
- `strict_privacy_mode=False`: phải bật rõ ràng.
- `document_detector_config=DetectorConfig('document', Path(...), confidence=.25, imgsz=640)`: checkpoint tin cậy do người dùng cung cấp. Không tự tải.

Heuristic complete-link gom các field gần nhau, kiểm tra các field singleton để hạn chế nối hai thẻ. Không gom toàn ảnh. Vùng suy ra lớn hơn 75% ảnh bị loại. Tiêu đề cccd/bhyt/gplx chỉ gợi ý loại giấy. DocumentRegion có confidence=None đối với heuristic; Detection tương thích dùng 0 cùng confidence_kind=unscored_heuristic. Không diễn giải giá trị này là xác suất. Supporting detections là bằng chứng, không phải nhãn ground truth.

Detector/hybrid yêu cầu checkpoint toàn giấy và taxonomy gồm một tập con không rỗng của identity_card, health_insurance_card, driver_license, birth_certificate, document_page. Thiếu checkpoint hoặc taxonomy sai báo lỗi, không fallback ngầm. Hybrid giữ vùng detector, thêm heuristic cho field chưa được bao phủ và che tất cả field. Vùng detector có field hỗ trợ ghi nguồn hybrid. Đây mới là adapter; không có checkpoint toàn giấy được tạo/train trong thay đổi này.

## Strict và trạng thái

Mọi frame tự động đều có coverage_status=coverage_unverified: pipeline không xác nhận toàn cảnh an toàn. Coverage reasons ghi nguyên nhân; không có trạng thái Protected dựa trên inference thành công.

Strict che đen toàn frame khi không có detection hiện tại; field không nằm trọn trong vùng detector toàn giấy; có vùng heuristic; full chưa có vùng detector toàn giấy; hoặc track đang giữ/dự đoán/partial. Blackout luôn đen bất kể mode ảnh. Với full+heuristic, strict sẽ che đen mọi frame. Đây là chính sách bảo thủ, có thể chặn cả cảnh trống.

Khi có bbox từ detector thật và không có dấu hiệu trên, strict có thể xuất kết quả đã che, nhưng vẫn coverage_unverified. Không phát hiện thấy giấy khác bị bỏ sót là rủi ro còn lại. Fields+strict không tự biến thành full. Confidence không chứng nhận an toàn.

## Chạy

```bash
.venv/bin/python scripts/predict_image.py --input /path/local/image.png --output outputs/full.png --document-redaction-mode full
.venv/bin/python scripts/predict_image.py --input /path/local/image.png --output outputs/strict.png --document-redaction-mode full --strict-privacy-mode
.venv/bin/python scripts/predict_image.py --input /path/local/image.png --output outputs/hybrid.png --document-redaction-mode full --document-detection-strategy hybrid --document-checkpoint /path/trusted/document.pt
.venv/bin/python scripts/benchmark.py --input /path/local/image.png --warmup 3 --runs 30 --document-redaction-mode full --imgsz-card 640 --imgsz-cccd 640
.venv/bin/python scripts/benchmark_webcam.py --camera 0 --duration 30 --document-redaction-mode full
.venv/bin/python main.py
```

CLI không đọc TOML tự động. Các flag document cũng dùng được trong benchmark ảnh/webcam. GUI chọn checkpoint cục bộ để mở detector/hybrid; taxonomy được kiểm tra trong worker lúc tải. Dừng phiên trước khi thay cấu hình. GUI giữ Card/CCCD 800, có thể hạ 640 hoặc tăng 960; face/plate giữ 640, không bật nhiều góc mặc định.

Webcam có latest raw slot trong capture và processed preview slot; không queue vô hạn. Model nạp một lần mỗi phiên. Lỗi capture/load/detector/CUDA/tracking/redaction/output khiến preview bị xóa và phiên dừng. Log chỉ loại exception và giai đoạn, không ảnh hay thông điệp tùy ý. Đầu ra chỉ được publish sau validation dtype/shape/timing. GUI chặn preview khác phiên hoặc quá 2 giây. Đây là giới hạn tuổi preview, không phải timeout hủy native inference/camera. Native read bị treo vẫn có thể làm chậm dừng worker; GUI vẫn chạy event loop và xóa preview.

Tracking document giữ tối đa 0.25 giây sau xác nhận, smoothing bao gồm bbox hiện tại. Khi mất dấu, vận tốc từ detection gần nhất dự đoán vị trí trong giới hạn 20% và mở rộng mask theo chuyển động. Không bảo đảm bám đúng vật thể nhanh; strict chặn track không xác nhận. Face/plate vẫn flow và TTL 0.6 giây. Reset khi phiên/nguồn/kích thước thay đổi.
