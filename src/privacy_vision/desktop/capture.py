"""Tiến trình đọc camera nền và cơ chế bộ đệm một khung hình mới nhất (Latest-frame slot).

Đảm bảo ảnh thô không bao giờ rò rỉ ra ngoài lớp worker và không gây trễ tích lũy hàng đợi.
"""
from dataclasses import dataclass
from threading import Condition, Event, Thread
from time import perf_counter
import numpy as np

ERROR_STATUS = 'Lỗi camera/AI — đã chặn hiển thị. Dừng rồi kiểm tra camera, model hoặc GPU.'


@dataclass(frozen=True)
class CapturedFrame:
    """Khung hình thô vừa đọc được từ phần cứng camera kèm thông tin thời gian."""
    image: np.ndarray
    read_at: float
    fps: float
    camera_info: str = ''


class LatestCapture:
    """Tiến trình thu nhận khung hình camera nền liên tục cập nhật frame mới nhất."""
    def __init__(self, camera, factory, preview_buffer):
        self.camera, self.factory, self.preview_buffer = camera, factory, preview_buffer
        self._condition = Condition()
        self._stop = Event()
        self._latest = None
        self._failed = False
        self.frames_read = 0
        self.frames_dropped = 0
        self._thread = Thread(target=self._run, name='privacyvision-capture', daemon=True)

    def start(self):
        """Khởi động luồng đọc camera nền."""
        self._thread.start()

    def _run(self):
        cap = None
        try:
            cap = self.factory(self.camera)
            info = getattr(cap, 'info', '')
            if not isinstance(info, str):
                info = ''
            frames, tick, fps = 0, perf_counter(), 0.0
            while not self._stop.is_set():
                ok, frame = cap.read()
                if self._stop.is_set():
                    break
                if not ok or frame is None:
                    raise RuntimeError('Không thể đọc dữ liệu từ camera')
                now = perf_counter()
                frames += 1
                if now - tick >= 1:
                    fps, frames, tick = frames / (now - tick), 0, now
                with self._condition:
                    self.frames_read += 1
                    if self._latest is not None:
                        self.frames_dropped += 1
                    self._latest = CapturedFrame(frame, now, fps, info)
                    self._condition.notify()
        except Exception:
            with self._condition:
                self._failed = True
                self._latest = None
                self.preview_buffer.clear(ERROR_STATUS)
                self._condition.notify_all()
        finally:
            if cap is not None:
                cap.release()

    def take(self, stopped):
        """Lấy khung hình thô mới nhất; trả về None nếu phiên bị dừng."""
        with self._condition:
            while self._latest is None and not self._failed:
                if stopped() or self._stop.is_set():
                    return None
                self._condition.wait(0.05)
            if self._failed:
                raise RuntimeError('Thu nhận khung hình thất bại')
            if stopped() or self._stop.is_set():
                return None
            item, self._latest = self._latest, None
            return item

    def publish(self, preview):
        """Xuất khung hình đã xử lý vào bộ đệm hiển thị."""
        with self._condition:
            if self._failed:
                raise RuntimeError('Camera gặp lỗi trong quá trình suy luận')
            if not self._stop.is_set():
                self.preview_buffer.publish(preview)

    def stop(self):
        """Dừng luồng đọc camera và chờ giải phóng tài nguyên."""
        self._stop.set()
        with self._condition:
            self._latest = None
            self._condition.notify_all()
        self._thread.join()
