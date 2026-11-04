# Chẩn đoán camera Fedora

Dừng các ứng dụng đang dùng webcam trước khi benchmark. `scripts/benchmark_camera.py` chỉ đo camera, không AI, không hiển thị/lưu ảnh. Dùng `--camera` đúng chỉ số thiết bị.

```bash
.venv/bin/python scripts/benchmark_camera.py --camera 1 --seconds 3
.venv/bin/python scripts/benchmark_camera.py --camera 1 --buffers 4 --formats MJPG,YUYV --compare-exposure
v4l2-ctl -d /dev/video1 --all
```

Kết quả đã nhận từ người dùng: auto exposure ~16.8 FPS, manual 80/160 ~30.4 FPS; đổi buffer/format không giải quyết tốc độ auto. Kiểm tra chỉ đọc hệ thống từng cho thấy auto_exposure=3 và exposure_dynamic_framerate=1: auto được phép giảm FPS. Người dùng đã tự khôi phục cấu hình; ứng dụng không tự ép control này toàn hệ thống. Theo [Linux V4L2](https://docs.kernel.org/userspace-api/media/v4l/ext-ctrls-camera.html), auto priority có thể thay đổi FPS; exposure absolute dùng đơn vị 100 µs.

Manual exposure là tùy chọn, không bật mặc định. Nếu đọc được control gốc, wrapper cố khôi phục khi release; diagnostics báo kết quả driver chấp nhận, không bảo đảm driver không clamp. Giảm exposure có thể làm ảnh tối và giảm detection. Driver FPS 30 là cấu hình, không phải throughput đo được.

Ring driver mặc định 4 buffer, reader đọc liên tục vào một slot mới nhất. Frame cũ chưa inference có thể bị bỏ nhưng mọi frame hiển thị đều qua AI. Nếu camera-only thấp thì điều tra ánh sáng/control/thiết bị; nếu camera-only cao nhưng AI thấp thì xem timing từng model và VRAM. Không tự giảm bảo vệ hoặc bỏ detector để tăng con số FPS.
