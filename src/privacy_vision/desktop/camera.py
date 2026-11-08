"""Cấu hình camera qua giao thức V4L2 (Video4Linux2) trên Linux."""
from dataclasses import dataclass
import cv2
from pathlib import Path


@dataclass(frozen=True)
class CameraOptions:
    """Tùy chọn cấu hình thiết bị camera ghi hình."""
    fps: int = 30
    manual_exposure: bool = False
    exposure: int = 160  # Đơn vị phơi sáng tuyệt đối của V4L2 (thường tương đương 100 microgiây).
    buffers: int = 4  # Số lượng bộ đệm vòng (ring buffer) trong kernel driver.
    fourcc: str = 'MJPG'
    width: int = 640
    height: int = 480

    def __post_init__(self):
        if self.fps < 1 or not 1 <= self.buffers <= 10 or self.exposure < 1 or self.fourcc not in ('MJPG', 'YUYV') or min(self.width, self.height) < 1:
            raise ValueError('Tùy chọn camera không hợp lệ')


class ConfiguredCamera:
    """Đối tượng bọc camera đã được thiết lập cấu hình đầy đủ."""
    def __init__(self, capture, info, restore=None):
        self.capture, self.info = capture, info
        self.restore = restore

    def read(self):
        """Đọc khung hình tiếp theo từ camera."""
        return self.capture.read()

    def release(self):
        """Giải phóng thiết bị camera và hoàn trả lại các thông số exposure ban đầu."""
        restored = None
        try:
            if self.restore is not None:
                auto, exposure = self.restore
                manual_ok = self.capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
                exposure_ok = self.capture.set(cv2.CAP_PROP_EXPOSURE, exposure)
                auto_ok = self.capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, auto)
                restored = bool(manual_ok and exposure_ok and auto_ok)
        finally:
            self.capture.release()
        return restored


def open_camera(index, options: CameraOptions):
    """Mở và cấu hình phần cứng camera theo các tùy chọn đã định nghĩa."""
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError('Không thể kết nối đến camera')
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*options.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, options.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, options.height)
        accepted = cap.set(cv2.CAP_PROP_FPS, options.fps)
        buffer_accepted = cap.set(cv2.CAP_PROP_BUFFERSIZE, options.buffers)
        exposure_status = 'exposure: giữ cấu hình camera'
        restore = None
        raw = cap.set(cv2.CAP_PROP_MODE, 0)
        old_auto = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE) if raw else -1
        old_exposure = cap.get(cv2.CAP_PROP_EXPOSURE) if raw else -1
        if options.manual_exposure:
            if old_auto in (0, 1, 2, 3) and old_exposure >= 1:
                restore = (old_auto, old_exposure)
            manual = raw and cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            exposure = manual and cap.set(cv2.CAP_PROP_EXPOSURE, options.exposure)
            exposure_status = (
                f'exposure yêu cầu {options.exposure}, đọc lại {cap.get(cv2.CAP_PROP_EXPOSURE):g}'
                if exposure else 'camera từ chối exposure thủ công'
            )
        reported_fps = cap.get(cv2.CAP_PROP_FPS)
        info = f'FPS yêu cầu {options.fps}, driver {reported_fps:g}'
        actual_code = int(cap.get(cv2.CAP_PROP_FOURCC))
        actual_fourcc = ''.join(chr((actual_code >> (8 * i)) & 255) for i in range(4)).strip('\x00')
        info += f' | format yêu cầu {options.fourcc}, thực nhận {actual_fourcc!r}'
        info += f' | {cap.get(cv2.CAP_PROP_FRAME_WIDTH):g}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):g}'
        if raw:
            info += f' | auto_exposure={cap.get(cv2.CAP_PROP_AUTO_EXPOSURE):g}, exposure={cap.get(cv2.CAP_PROP_EXPOSURE):g}'
        try:
            name = Path(f'/sys/class/video4linux/video{index}/name').read_text().strip()
            info += f' | thiết bị {name}'
        except OSError:
            pass
        info += f' | buffer driver {cap.get(cv2.CAP_PROP_BUFFERSIZE):g} (yêu cầu {options.buffers})'
        if not buffer_accepted:
            info += ' (driver từ chối buffer)'
        if not accepted:
            info += ' (driver từ chối FPS)'
        return ConfiguredCamera(cap, info + ' | ' + exposure_status, restore)
    except Exception:
        cap.release()
        raise
