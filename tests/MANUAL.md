# Kiểm thử ánh sáng và chuyển động

Chạy `.venv/bin/python main.py --camera 1 --log outputs/lighting-motion.csv`.
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
