# PrivacyVision

Ứng dụng che thông tin cá nhân cục bộ: khuôn mặt, biển số và các trường giấy tờ. Core AI nhận ảnh BGR độc lập giao diện; desktop PySide6 dùng worker và chỉ hiển thị frame đã xử lý. Không OCR/cloud, không train lại.

## Cấu trúc

```text
project/
├── main.py                     # Mở ứng dụng desktop đang dùng
├── src/privacy_vision/
│   ├── config.py               # Cấu hình detector/pipeline
│   ├── schemas.py              # Detection và kết quả
│   ├── pipeline.py             # Điều phối inference → xử lý → che
│   ├── image_io.py             # Unicode I/O, ghi atomic
│   ├── detectors/              # BaseDetector, YuNet, YOLO
│   ├── processing/             # Bbox, gom giấy, tracking, redaction
│   ├── desktop/                # Giao diện, worker, capture, cấu hình camera
│   └── cli/                    # Xử lý ảnh và chẩn đoán camera
├── scripts/                    # Lệnh ngắn cho ảnh, benchmark, webcam
├── tests/
│   ├── core/                   # Bbox, che, gom giấy, tracking, pipeline
│   ├── desktop/                # Qt offscreen, worker/capture bằng mock
│   ├── integration/            # Smoke test với ảnh/model thật, opt-in
│   └── legacy/                 # Regression của thử nghiệm mặt cũ
├── examples/face_webcam/       # Bản OpenCV cũ, giữ để tham khảo
├── models/                     # Bốn checkpoint ứng dụng
│   └── pretrained/             # yolo26n.pt gốc, không được pipeline tải
├── configs/example.toml        # Minh họa cấu hình API, CLI không tự đọc
├── docs/                       # Kiến trúc, chất lượng detection, kiểm thử
├── inputs/                     # Ảnh tự cung cấp, không đưa vào Git
└── outputs/                    # Kết quả, không đưa vào Git
```

## Cài đặt trên Fedora, Python 3.12

```bash
sudo dnf install python3.12 python3.12-devel
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[desktop,test]'
```

PyTorch CUDA phải phù hợp driver NVIDIA; chọn lệnh theo [PyTorch chính thức](https://pytorch.org/get-started/locally/). Kiểm tra:

```bash
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")'
```

Không cài đồng thời nhiều package OpenCV khác nhau. Pipeline không dùng `imshow`; demo cũ trong examples có dùng. Với triển khai chỉ headless, dùng môi trường riêng và thay OpenCV bằng headless nếu cần.

## Model

Chuẩn bị bốn file **đáng tin cậy** trong `models/`:

- `face_detection_yunet_2026may.onnx`: YuNet phát hiện cả khuôn mặt, CPU.
- `plate_best.pt`: YOLO biển số.
- `vietnamese_card_best.pt`: YOLO các field giấy tờ.
- `cccd_information_best.pt`: YOLO các field CCCD.

Pipeline không tự tải model/dataset và không sửa trọng số. `.pt` có thể thực thi mã khi deserialize, chỉ dùng checkpoint tin cậy. `models/pretrained/yolo26n.pt` là file nền cũ, không phải checkpoint nhận diện giấy tờ.

## Chạy desktop

Từ thư mục project:

```bash
.venv/bin/python main.py
```

Chọn camera, detector và chế độ black/blur/pixelate rồi bắt đầu. Dừng phiên trước khi đổi cấu hình. Đường chạy tương đương: `scripts/webcam_desktop.py`. Bản thử nghiệm cũ: `python examples/face_webcam/main.py --camera 1` (có Detect Only; không phải đường chạy desktop mới).

Mặc định desktop: camera 720p, mặt/biển số imgsz 640, **giấy tờ imgsz 800**, confidence giấy tờ 0.25. Có thể chọn 640/800/960 cho riêng Card/CCCD; cao hơn tăng chi phí inference và VRAM. Camera có thể không hỗ trợ độ phân giải/FPS yêu cầu; xem cấu hình thực nhận ở cuối cửa sổ. Hạ confidence có thể tăng recall nhưng cũng tăng nhận nhầm, không tự động làm detection chính xác hơn.

Dòng thống kê face/plate/card/cccd đếm detection mới trong frame hiện tại, không tính vùng giấy suy ra hoặc track đang giữ. Nó giúp phân biệt model không thấy field với trường hợp đã thấy field nhưng chưa suy ra được cả giấy. Chưa có bằng chứng chất lượng giấy tờ tương đương phát hiện mặt.

## Ảnh, video và benchmark

```bash
# Xử lý 1 ảnh
.venv/bin/python scripts/predict_image.py --input 'inputs/ảnh.jpg' --output outputs/redacted.png --mode pixelate --device auto --metadata-output outputs/metadata.json

# Xử lý cả file video (.mp4, .avi, .mov, .mkv) với tracking mượt mà
.venv/bin/python scripts/predict_video.py --input 'inputs/clip.mp4' --output outputs/redacted_clip.mp4 --mode pixelate --device auto --metadata-output outputs/video_meta.json

# Benchmark tốc độ inference ảnh
.venv/bin/python scripts/benchmark.py --input 'inputs/ảnh.jpg' --warmup 1 --runs 5
.venv/bin/python scripts/predict_image.py --help
.venv/bin/python scripts/predict_video.py --help
```

Hỗ trợ threshold riêng, bật/tắt detector, margin mỗi phía (0.1 = 10%), device auto/cpu/0 và imgsz từng detector. `--document-rotations 0,90,180,270` thử thêm góc Card/CCCD nhưng chậm hơn. Ảnh gốc và metadata/output phải khác đường dẫn. Benchmark tách detector/inference/redaction, không gồm tải model hoặc I/O ảnh.

Chẩn đoán camera, dừng desktop trước:

```bash
.venv/bin/python scripts/benchmark_camera.py --camera 1 --buffers 4 --formats MJPG,YUYV --compare-exposure
```

Lệnh chỉ in cấu hình/timing, không hiển thị/lưu ảnh. Trong phép đo người dùng đã cung cấp, auto exposure đạt ~16.8 FPS, manual 80/160 đạt ~30.4 FPS ở cả hai format. Không lấy FPS driver báo làm FPS thực đo. Ứng dụng không tự thay FPS động toàn hệ thống; xem [chẩn đoán camera](docs/camera.md).

## API và kiểm thử

```python
from privacy_vision.config import PipelineConfig
from privacy_vision.pipeline import PrivacyPipeline

pipeline = PrivacyPipeline(PipelineConfig(device="auto", whole_documents=True))
# frame: numpy.ndarray uint8 H×W×3 BGR
result = pipeline.process(frame)
# result.image: ảnh đã xử lý, result.to_metadata(): bbox/class/confidence/timing
```

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m pytest tests/core tests/desktop -q
PRIVACYVISION_SMOKE_IMAGE='inputs/ảnh.jpg' .venv/bin/python -m pytest tests/integration -q
```

Kiểm thử mặc định gồm core/desktop/legacy và smoke test opt-in. Kết quả nâng cấp và giới hạn môi trường được ghi tại [báo cáo nâng cấp](docs/upgrade-validation.md). Test synthetic không chứng minh recall giấy tờ thực tế.

Unit test mới dùng mock và hình tổng hợp, không cần GPU/model. Smoke test opt-in dùng model thật; test legacy YuNet cần checkpoint. Test Qt dùng offscreen. Xem [kiến trúc và giới hạn detection](docs/architecture.md), [kiểm thử thủ công bản mặt cũ](docs/testing.md).

## Quyền riêng tư và giới hạn

Không bảo đảm phát hiện hoặc che hết dữ liệu cá nhân. Blur/pixelate không bảo đảm chống khôi phục nội dung; ưu tiên black và PNG. Chưa hỗ trợ giấy tờ mới, giấy khai sinh, ảnh tối/nghiêng hoặc field quá nhỏ một cách chắc chắn. Không coi class tiêu đề cccd/bhyt/gplx là toàn bộ giấy; không cộng mAP hai dataset chồng lấn.

Không lưu bản gốc/crop hoặc log nội dung cá nhân. Detector bật mà lỗi thì dừng, không xuất frame gốc dự phòng. Ghi ảnh/metadata atomic riêng; nếu lỗi, output cũ có thể còn đó nên phải kiểm tra exit code. `.gitignore` không gỡ checkpoint đã được theo dõi từ trước; kiểm tra trước khi commit. Không push tự động.

## Tích hợp Virtual Camera (Google Meet / Zoom / MS Teams)

PrivacyVision hỗ trợ phát trực tiếp luồng video đã che mờ bảo vệ riêng tư tới **Virtual Webcam (Webcam ảo)** thông qua `pyvirtualcam` (hỗ trợ `v4l2loopback` trên Linux, `OBS Virtual Camera` trên Windows/macOS):

### 1. Chuẩn bị thiết bị webcam ảo (Linux)
Nếu chưa bật `v4l2loopback`, kích hoạt nhanh bằng lệnh:
```bash
sudo modprobe v4l2loopback devices=1 video_nr=0 card_label="PrivacyVision Cam" exclusive_caps=1
```
*(Trên hệ thống của bạn đã có sẵn `/dev/video0: OBS Virtual Camera`)*

### 2. Sử dụng trên Desktop GUI
1. Mở ứng dụng: `.venv/bin/python main.py`
2. Tích chọn ô **`🎥 Virtual Camera (Meet/Zoom)`** ở thanh điều khiển phía dưới.
3. Nhấn **`▶ Bắt đầu`**. Trạng thái sẽ hiển thị: `🟢 VCam: /dev/video0 (Live)`.
4. Mở Google Meet, Zoom hoặc MS Teams $\rightarrow$ Cài đặt Video $\rightarrow$ Chọn Camera: **"PrivacyVision Cam"** hoặc **"OBS Virtual Camera"**.
5. Mọi hình ảnh bạn gọi trực tuyến sẽ được che mờ khuôn mặt, biển số và giấy tờ tự động theo thời gian thực!

### 3. Sử dụng qua dòng lệnh (CLI)
```bash
# Chạy webcam test và tự động phát tới Virtual Camera
.venv/bin/python scripts/test_camera.py --virtual-cam

# Nhấn phím 'm' trên cửa sổ camera để BẬT/TẮT Virtual Camera tức thì
```

## Document protection và strict

GUI có Fields only / Full document, strategy heuristic / detector / hybrid và strict opt-in. CLI/API vẫn mặc định fields, GUI vẫn full như trước. Full thêm vùng giấy và luôn giữ các field làm lớp che dự phòng; heuristic không xác nhận mép giấy. Vùng heuristic không có confidence xác suất. Mọi frame tự động vẫn `coverage_unverified`.

Strict che đen khi có dấu hiệu thiếu: không có detection, field ngoài vùng detector toàn giấy, heuristic hoặc track mất dấu. **Full + heuristic + strict luôn che đen**, kể cả cảnh trống. Đây không phải bảo vệ tuyệt đối; detector thật vẫn có thể bỏ sót giấy khác. Lỗi kỹ thuật xóa preview/dừng phiên; không fallback raw frame. Preview quá 2 giây hoặc khác phiên bị xóa.

Document Detector mới chỉ là adapter cho checkpoint tin cậy do người dùng cung cấp; chưa có model mới được train trong nhiệm vụ. Thiếu checkpoint ở detector/hybrid báo lỗi. Không xác nhận hỗ trợ đáng tin cậy giấy khai sinh, A4 hoặc bất kỳ loại giấy nào bằng test synthetic.

```bash
.venv/bin/python scripts/predict_image.py --input inputs/test.jpg --output outputs/full.png --document-redaction-mode full
.venv/bin/python scripts/benchmark_webcam.py --camera 0 --duration 30 --document-redaction-mode full
.venv/bin/python scripts/evaluate_documents.py --manifest inputs/private-eval/manifest.json --document-redaction-mode full
```

Xem [cấu hình và chính sách strict](docs/document-protection.md), [bộ đánh giá thực tế](docs/evaluation.md). Benchmark webcam không lưu ảnh. Không đưa giấy tờ nhạy cảm hoặc manifest chứa thông tin thật vào repository công khai.
