# Đánh giá dữ liệu webcam độc lập

Chưa có bộ dữ liệu webcam gán nhãn độc lập trong repository. Chưa có loại giấy nào được xác nhận đáng tin cậy bằng bộ đánh giá này; giấy khai sinh và A4 chỉ là mục tiêu tương lai. Test synthetic/mock không chứng minh recall thực tế.

Chuẩn bị ảnh hoặc video cục bộ có sự đồng ý của chủ thông tin; không dùng dữ liệu train làm test. Giữ file và manifest trong inputs/ (được ignore); không commit hoặc đưa thông tin giấy tờ vào log. Công cụ chỉ in số liệu tổng hợp, không lưu ảnh/crop/video. Đối với video, chuẩn bị trước offline; ứng dụng webcam bảo vệ không ghi raw để tạo dataset.

Gán nhãn bbox toàn giấy, loại giấy, bbox nhạy cảm nếu có, điều kiện chụp, mức khó; video cần frame_index và ID chuỗi. Bbox xyxy theo pixel gốc, phải có diện tích dương và trong ảnh. Mỗi người/giấy/chuỗi chỉ thuộc một split, không chia frame liền kề vào train và test.

Bao gồm CCCD/CMND trước/sau, BHYT, GPLX, giấy khai sinh, A4 chứa dữ liệu cá nhân; nhỏ/xa, nghiêng, mờ, lóa, che khuất, nhiều giấy sát/xa, nền phức tạp và cảnh không có giấy. Phân tầng báo cáo theo loại/điều kiện bằng các manifest riêng.

Ví dụ manifest (tọa độ chỉ minh họa):

```json
{"samples":[{"path":"sample.png","document_id":"anonymous-001","conditions":["glare"],"documents":[{"bbox":[10,20,300,190],"document_type":"identity_card"}],"sensitive_regions":[{"bbox":[50,70,200,90]}]}]}
```

Với video, path trỏ tới video và thêm `frame_index` (bắt đầu từ 0). Không đặt tên/ID thật trong manifest. `sensitive_regions` thiếu nghĩa là chưa gán nhãn coverage; danh sách rỗng nghĩa là xác nhận không có vùng nhạy cảm đã gán nhãn. Không xem thiếu nhãn như coverage 100%.

```bash
.venv/bin/python scripts/evaluate_documents.py --manifest inputs/private-eval/manifest.json --document-redaction-mode full
.venv/bin/python scripts/evaluate_documents.py --manifest inputs/private-eval/manifest.json --document-redaction-mode full --document-detection-strategy hybrid --document-checkpoint /path/trusted/document.pt
```

## Chỉ số

- Document matching: một-một, cùng loại, greedy theo IoU giảm dần tại IoU >= 0.5. TP là cặp khớp; FN là giấy thật không khớp; FP là proposal không khớp. Recall=TP/(TP+FN), null nếu không có giấy thật. Detector/hybrid và heuristic báo riêng; heuristic chưa biết loại không được tính là đúng loại. Không cộng hai nguồn thành một mAP.
- Sensitive coverage: số pixel của hợp vùng nhạy cảm giao mask cuối chia số pixel hợp vùng nhạy cảm. Pixel lộ=1-coverage. Vùng lộ là bbox nhạy cảm có ít nhất một pixel ngoài mask; tỷ lệ vùng lộ=số vùng lộ/tổng vùng nhạy cảm. Dùng hợp để không đếm pixel trùng hai lần.
- Mask được tính từ đúng các bbox redaction; blackout là mask toàn ảnh. Báo blackout riêng khỏi frame thường. Blur/pixelate chỉ được đánh giá vị trí mask, không chứng minh chống khôi phục. Bbox annotations cũng chỉ xấp xỉ; dùng segmentation annotation nếu cần đo pixel chính xác theo hình dạng.
- IoU vùng giấy không phải coverage nhạy cảm. Một bbox IoU cao vẫn có thể bỏ QR hoặc dòng chữ.
- Công cụ ảnh/video đánh giá từng frame độc lập, không tracking; latency là pipeline P50/P95, không FPS webcam.
- Benchmark webcam đo FPS capture và xử lý thực tế, frame bị thay thế, P50/P95 từng công đoạn và read-to-redaction. Duration không thể hủy native call bị treo. VRAM báo torch.max_memory_allocated, không phải toàn bộ bộ nhớ GPU/driver. FPS GUI xem trực tiếp trong desktop.

Chạy benchmark 640 và 800 riêng, cùng nguồn/ánh sáng/detector/mode, tối thiểu ba lần; ghi GPU/driver và cấu hình hiệu lực. Không suy ra 30 FPS từ ảnh tĩnh. Muốn đánh giá tracking phải gán nhãn và quan sát chuỗi thật, ghi thời gian mất vùng và tỷ lệ frame/vùng lộ; test unit không thay thế bước này.
