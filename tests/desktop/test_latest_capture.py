from queue import Queue
from threading import Event
import numpy as np
import pytest
from privacy_vision.desktop.capture import LatestCapture
from privacy_vision.desktop.session import PreviewBuffer, Preview


def test_capture_keeps_newest_and_failure_blocks_inflight_result():
    queue = Queue()
    waiting_for_fourth = Event()
    released = Event()
    class Camera:
        reads = 0
        def read(self):
            self.reads += 1
            if self.reads == 101:
                waiting_for_fourth.set()
            return queue.get(timeout=2)
        def release(self):
            released.set()
    for value in range(1,101):
        queue.put((True, np.full((2,2,3), value, np.uint8)))
    buffer = PreviewBuffer()
    capture = LatestCapture(0, lambda _: Camera(), buffer)
    capture.start()
    try:
        assert waiting_for_fourth.wait(2)
        # Capture outran AI by 100 frames; retain only the newest.
        item = capture.take(lambda: False)
        assert np.all(item.image == 100)
        assert capture.frames_read == 100 and capture.frames_dropped == 99
        safe = Preview(np.zeros((2,2,3), np.uint8), 0, 0, {}, 'cpu')
        capture.publish(safe)
        queue.put((False, None))
        assert released.wait(2)
        assert buffer.take()[0] is None
        # A result finishing after capture failure must not resurrect preview.
        with pytest.raises(RuntimeError): capture.publish(safe)
        with pytest.raises(RuntimeError): capture.take(lambda: False)
    finally:
        queue.put((False, None))
        capture.stop()
    assert not capture._thread.is_alive()
