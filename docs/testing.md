# Kiểm thử bản desktop hiện tại

```bash
.venv/bin/python -m pytest -q
PRIVACYVISION_SMOKE_IMAGE=inputs/test.jpg .venv/bin/python -m pytest tests/integration -q
.venv/bin/python main.py
```

Unit dùng mock/synthetic, Qt offscreen; smoke ảnh thật opt-in, không xuất ảnh. Chọn fields/full, strict, strategy; kiểm tra full+heuristic+strict đen và cảnh trống không hiện Protected. Rút camera/làm model lỗi: preview phải xóa, chỉ trạng thái lỗi; không raw fallback. Với worker chậm, preview quá 2 giây phải xóa và GUI vẫn phản hồi. Dừng phiên trước khi đổi camera/model; phiên mới không mang track cũ. Capture nhanh hơn inference phải bỏ frame cũ, không tích lũy độ trễ. Không ghi raw frame để debug.

Benchmark và chỉ số coverage xem [đánh giá](evaluation.md); cấu hình xem [document protection](document-protection.md). Chưa xác nhận webcam thực tế chỉ từ unit tests.

## Tham khảo: thử nghiệm mặt cũ

Phần dưới dành riêng demo cũ có Detect Only, không phải đường desktop bảo vệ hiện tại.

# Kiểm thử ánh sáng và chuyển động

Chạy `.venv/bin/python examples/face_webcam/main.py --camera 1 --log outputs/lighting-motion.csv`.
Đổi camera thành 0 nếu cần. Phím 1: Detect Only; 2: Blur; 3: Pixelate;
4: Blackout; Q: thoát. Bounding box biểu thị vùng che đã mở rộng 20%.

Giữ một người trong khung hình và thử mỗi tình huống 20 giây với từng chế độ:

| Tình huống | Thao tác |
| --- | --- |
| Ánh sáng ổn định | Nhìn thẳng, đứng yên để lấy mốc |
| Quay đầu | Quay trái/phải từ từ rồi nhanh; thử góc nghiêng và nhìn lên/xuống |
| Chuyển động | Di chuyển ngang và tiến gần/ra xa camera |
| Thay đổi ánh sáng | Bật/tắt đèn, giảm sáng và thử ngược sáng |
| Kết hợp | Quay đầu trong lúc thay đổi ánh sáng |

Ghi thời gian bắt đầu/kết thúc từng tình huống (theo elapsed_s của CSV),
chế độ, số lần vùng che biến mất khi mặt vẫn trong hình, thời lượng mất dài
nhất và mức độ che đủ khuôn mặt. Lặp lại ba lần cùng vị trí camera.
Detect Only dùng để quan sát khả năng phát hiện; ba chế độ còn lại dùng
để đánh giá vùng che. Khi cần đo chính xác số frame bị lộ hoặc có nhiều
người, cần video được đồng ý ghi và gán nhãn từng frame để đối chiếu.

CSV ghi mỗi frame: thời gian, chế độ, số mặt, confidence thấp nhất,
FPS cập nhật mỗi giây và thời gian suy luận. FPS bằng 0 trong giây đầu
là đang khởi tạo phép đo. CSV không lưu ảnh. Với một người luôn trong
khung, chuỗi detected_faces=0 giúp xác định khoảng nghi ngờ mất phát hiện, nhưng
không chứng minh vùng che đủ hoặc thiếu khi faces>0. Confidence là điểm
phát hiện của model, không phải độ chính xác đã đo trên tập kiểm thử.

Ứng dụng dùng optical flow khi mất detection và giữ vùng che tối đa 0,6 giây
kể từ detection cuối. Nhãn Tracked là bám theo chuyển động; Held là giữ vị trí
khi không đủ điểm bám. Last conf là confidence của detection cuối, không phải
điểm tracking. CSV có detected_faces và tracked_faces để phân biệt detection
với vùng che được duy trì. So sánh các khoảng detected_faces=0 nhưng faces>0
và kiểm tra trực quan vùng che còn bám đúng mặt. Tracking có thể trôi khi đổi
sáng hoặc chuyển động nhanh; quá 0,6 giây không phát hiện lại thì vùng che mất. Chưa có kết luận thực nghiệm về mức độ mất
vùng che: môi trường triển khai thay đổi này không có webcam để thử.

## Chẩn đoán Qt Wayland và preview đen

Các dòng qt.qpa.wayland.textinput enableSurface/disableSurface chưa đủ để kết luận lỗi camera hoặc AI. Đọc trạng thái cuối GUI và dòng webcam_<stage>_failure:<ExceptionType>. GUI hiện ghi rõ lỗi tải model/camera/AI-redaction/validation/publication; không in nội dung exception có thể chứa dữ liệu riêng tư.

Nếu preview đen và trạng thái Strict blackout: full+heuristic+strict được thiết kế chặn toàn frame. Để thử detection, dừng phiên, tắt strict rõ ràng, chọn Fields only hoặc Full document/heuristic; coverage vẫn chưa xác minh. Không chọn một field checkpoint làm Document Detector.

Nếu cửa sổ hiển thị/focus bất thường trên Wayland, có thể thử XWayland khi hệ thống có backend xcb và XWayland:

```bash
QT_QPA_PLATFORM=xcb .venv/bin/python main.py
```

Đây là lựa chọn backend GUI, không sửa lỗi inference và không thay chính sách bảo vệ. Không tắt toàn bộ Qt/OpenCV logs để che triệu chứng. Qt mô tả QT_QPA_PLATFORM tại https://doc.qt.io/qt-6/qpa.html.

Nếu chưa có preview, thử dừng phiên rồi chọn device CPU, imgsz giấy tờ 640 và camera index đúng; theo dõi trạng thái. Không đổi driver/cài lại CUDA chỉ từ cảnh báo Wayland hoặc setPreferableTarget. Cảnh báo OpenCV graph engine cần phân biệt với exception thực sự; smoke test model thật có thể kiểm tra inference độc lập GUI:

```bash
PRIVACYVISION_SMOKE_IMAGE=inputs/test.jpg .venv/bin/python -m pytest tests/integration -q
```
