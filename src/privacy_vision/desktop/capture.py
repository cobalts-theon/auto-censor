"""One camera producer, one latest-frame slot; raw frames stay inside worker layer."""
from dataclasses import dataclass
from threading import Condition, Event, Thread
from time import perf_counter
import numpy as np

ERROR_STATUS = 'Lỗi camera/AI — đã chặn hiển thị. Dừng rồi kiểm tra camera, model hoặc GPU.'


@dataclass(frozen=True)
class CapturedFrame:
    image: np.ndarray
    read_at: float
    fps: float
    camera_info: str = ''


class LatestCapture:
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
                    raise RuntimeError('Camera read failed')
                now = perf_counter()
                frames += 1
                if now-tick >= 1:
                    fps, frames, tick = frames/(now-tick), 0, now
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
        with self._condition:
            while self._latest is None and not self._failed:
                if stopped() or self._stop.is_set():
                    return None
                self._condition.wait(.05)
            if self._failed:
                raise RuntimeError('Capture failed')
            if stopped() or self._stop.is_set():
                return None
            item, self._latest = self._latest, None
            return item

    def publish(self, preview):
        # Serialize publication with capture failure so an in-flight inference
        # cannot restore a frame after the capture thread has reported an error.
        with self._condition:
            if self._failed:
                raise RuntimeError('Capture failed during inference')
            if not self._stop.is_set():
                self.preview_buffer.publish(preview)

    def stop(self):
        self._stop.set()
        with self._condition:
            self._latest = None
            self._condition.notify_all()
        self._thread.join()  # Release stays on the capture owner thread.
