# Kiến trúc và giới hạn bảo vệ

Core public API: PrivacyPipeline(config, detectors=None, document_detector=None), process(BGR uint8, tracker=None), process_file(input, output). Giữ tham số positional cũ. Detection bbox xyxy theo ảnh gốc; sau normalize là half-open integer. PipelineResult giữ image/detections/timings_ms và bổ sung document_regions, coverage_status, coverage_reasons, strict_blackout. Metadata cũ được giữ, khóa mới bổ sung.

## Dòng xử lý

Detector bật → tọa độ ảnh gốc → kiểm tra output detector → clip/margin → DocumentRegion → tracking → hợp mask và redaction một lần → kiểm tra ảnh kết quả → publication.

YuNet resize về ảnh gốc; YOLO trả xyxy gốc. Model chạy tuần tự batch 1, FP16 CUDA nếu có và FP32 CPU; không gọi cùng pipeline đồng thời. Model tải một lần mỗi phiên, không tải theo frame. Không tự download hoặc thay trọng số.

Fields che detection hiện có. Full giữ toàn bộ field/mặt/biển số và thêm vùng giấy. DocumentRegion lưu bbox, confidence, document_type, source, field_count và supporting_detections. Vùng heuristic không có confidence; detection tương thích dùng 0 với confidence_kind=unscored_heuristic. Đây không phải xác suất.

## Vùng giấy

Card/CCCD phát hiện field, không xác định mép thẻ. Heuristic gom cụm không gian complete-link, canonicalize class, hạn chế hai singleton khác vị trí nằm chung cụm; hai model đóng góp cùng cụm. Mặc định ba loại field hoặc hai loại có hint. Tiêu đề cccd/bhyt/gplx không được dùng làm bbox toàn thẻ. Union mở rộng ngang 25%, dọc 30%, clip; vùng >75% ảnh bị loại. Field vẫn che khi cụm thiếu bằng chứng. Chỉ bỏ proposal lồng nhau khi bằng chứng tương thích. Giấy đặt sát, góc chụp khác biệt hoặc field nhỏ có thể làm gom nhầm/tách nhầm.

DocumentDetector là adapter checkpoint toàn giấy do người dùng cung cấp, không phải model mới đã train. Detector/hybrid thiếu checkpoint báo lỗi rõ ràng. Hybrid dùng document bbox và field evidence, thêm heuristic cho field ngoài document bbox; mọi field vẫn được che. Không tuyên bố hỗ trợ giấy khai sinh/A4 từ taxonomy dự kiến.

## Coverage và strict

Mọi kết quả tự động ghi coverage_unverified; inference thành công hoặc không có detection không chứng nhận toàn cảnh an toàn. Strict là opt-in, mặc định tắt. Chính sách blackout khi không detection hiện tại, field không có vùng detector bao phủ, vùng heuristic, full chưa có vùng detector hoặc track đang mất xác nhận. Full+heuristic+strict luôn đen. Có vùng detector thật vẫn chưa chứng minh không bỏ sót giấy khác. [Cấu hình chi tiết](document-protection.md).

## Tracking

Mặt/biển số dùng forward/backward flow, TTL tối đa 0.6 giây. Giấy smoothing bbox, giữ tối đa 0.25 giây sau xác nhận; bbox hiện tại vẫn được che ngay khi chưa đủ hai lần xác nhận để giữ track. Dự đoán ngắn dựa trên vận tốc giữa detection, giới hạn dịch chuyển 20%, mở rộng mask theo chuyển động. Bbox dự đoán không được dùng để gia hạn track. Track partial/held/tracked đều không phải chứng nhận an toàn; strict chặn chúng. Frame field chỉ dùng detection hiện tại.

Tracker mới mỗi phiên, reset khi nguồn mới hoặc kích thước đổi; reset() cho người dùng API tự quản lý nguồn. Không tái sử dụng track giữa video. Test synthetic chuỗi chỉ kiểm tra logic, chưa đánh giá rung/mất dấu ngoài thực tế.

## Webcam fail-closed

Camera producer giữ một raw frame mới nhất trong worker. AI consumer xử lý đầy đủ detector bật, redaction, validate rồi chuyển vào mailbox processed một frame. GUI chỉ nhận processed preview với session ID và timestamp; copy Qt image trước hiển thị. Không đường raw đến GUI hoặc fallback raw.

Camera failure và publication được serialize bằng khóa, không cho inference vừa xong hồi sinh preview sau capture failure. Lỗi load/capture/detector/CUDA/tracking/redaction hoặc kết quả ảnh/timing sai xóa preview, dừng phiên. Log chỉ stage/type lỗi, không ảnh/nội dung exception. Không giữ preview khác phiên hoặc quá 2 giây. GUI không block khi inference chậm.

Stop/close hợp tác chờ native read/inference, chưa cô lập process hoặc hủy được native call treo. Giới hạn tuổi preview không phải inference timeout. Khởi động lại phiên sau khi worker cũ kết thúc. Ảnh/metadata vẫn ghi atomic riêng; output cũ có thể còn nếu lệnh thất bại, phải xem exit code.

## Hiệu năng và kiểm chứng

CLI mặc định từng model 640; GUI face/plate 640, Card/CCCD 800 tùy chọn 640/960. Không bật nhiều góc mặc định. Không queue vô hạn; bỏ raw frame cũ chưa xử lý. GUI báo FPS thực nhận, capture/AI FPS, latency và device. Benchmark webcam không lưu ảnh và báo frame bị thay thế, P50/P95, torch VRAM nếu có. Không lấy FPS driver hoặc benchmark ảnh làm FPS webcam.

Chưa có tập webcam gán nhãn độc lập để khẳng định recall, coverage, rung hoặc hỗ trợ giấy mới. Xem [đánh giá thực tế](evaluation.md), [kiểm thử](testing.md).
