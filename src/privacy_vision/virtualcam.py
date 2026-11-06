"""Virtual Camera broadcaster for PrivacyVision.

Streams redacted frames to a virtual webcam device (e.g., v4l2loopback on Linux,
OBS Virtual Camera on Windows/macOS) so third-party applications like Google Meet,
Zoom, Microsoft Teams, and Discord can consume the privacy-protected stream directly.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import cv2

logger = logging.getLogger(__name__)


def check_virtual_cam_support() -> Dict[str, Any]:
    """Check whether pyvirtualcam is installed and what virtual devices are available."""
    try:
        import pyvirtualcam  # noqa: F401
    except ImportError:
        return {
            "available": False,
            "installed": False,
            "device": None,
            "label": None,
            "backend": None,
            "all_devices": [],
            "message": "pyvirtualcam chưa được cài đặt. Cài đặt bằng: pip install pyvirtualcam",
        }

    # Detect Linux v4l2loopback devices in sysfs
    devices: List[Tuple[str, str]] = []
    v4l_dir = Path("/sys/devices/virtual/video4linux")
    if v4l_dir.exists():
        for dev in sorted(v4l_dir.iterdir()):
            name_file = dev / "name"
            label = name_file.read_text().strip() if name_file.exists() else "Virtual Camera"
            devices.append((f"/dev/{dev.name}", label))

    if devices:
        primary_dev, primary_label = devices[0]
        return {
            "available": True,
            "installed": True,
            "device": primary_dev,
            "label": primary_label,
            "all_devices": devices,
            "backend": "v4l2loopback",
            "message": f"Sẵn sàng: {primary_label} ({primary_dev})",
        }

    # Fallback to auto-detection (useful for OBS Virtual Cam on Windows/macOS)
    return {
        "available": True,
        "installed": True,
        "device": None,
        "label": "Auto-detect",
        "all_devices": [],
        "backend": "auto",
        "message": "pyvirtualcam đã sẵn sàng (chế độ tự động phát hiện thiết bị)",
    }


class VirtualCamBroadcaster:
    """Manages streaming processed BGR frames to a virtual webcam device."""

    def __init__(
        self,
        device: Optional[str] = None,
        fps: int = 30,
        enabled: bool = False,
    ):
        self.device = device
        self.fps = fps
        self.enabled = enabled
        self._cam = None
        self._is_active = False
        self._width = 0
        self._height = 0
        self._actual_device = None
        self._actual_backend = None
        self.last_error: Optional[str] = None

    @property
    def is_active(self) -> bool:
        return self._is_active and self._cam is not None

    @property
    def device_name(self) -> str:
        return self._actual_device or self.device or "Virtual Camera"

    @property
    def actual_backend(self) -> Optional[str]:
        return self._actual_backend

    def start(self, width: int, height: int, fps: Optional[int] = None) -> bool:
        """Start the virtual camera stream at the specified resolution and frame rate."""
        if self._is_active and self._cam is not None:
            if self._width == width and self._height == height:
                return True
            self.stop()

        try:
            import pyvirtualcam

            target_fps = fps or self.fps or 30
            kwargs: Dict[str, Any] = {
                "width": width,
                "height": height,
                "fps": target_fps,
                "fmt": pyvirtualcam.PixelFormat.BGR,
            }
            if self.device:
                kwargs["device"] = self.device

            self._cam = pyvirtualcam.Camera(**kwargs)
            self._width = width
            self._height = height
            self._actual_device = getattr(self._cam, "device", self.device)
            self._actual_backend = getattr(self._cam, "backend", "unknown")
            self._is_active = True
            self.last_error = None
            logger.info(
                "Virtual camera started: %s (%dx%d @ %dfps, backend: %s)",
                self._actual_device,
                width,
                height,
                target_fps,
                self._actual_backend,
            )
            return True
        except Exception as exc:
            self.last_error = str(exc)
            logger.warning("Failed to start virtual camera: %s", exc)
            self._cam = None
            self._is_active = False
            return False

    def send(self, frame_bgr: np.ndarray) -> bool:
        """Send a BGR frame to the virtual camera. Automatically resizes if size mismatches."""
        if not self._is_active or self._cam is None:
            return False

        try:
            h, w = frame_bgr.shape[:2]
            if (w, h) != (self._width, self._height):
                frame_to_send = cv2.resize(frame_bgr, (self._width, self._height))
            else:
                frame_to_send = frame_bgr

            if not frame_to_send.flags["C_CONTIGUOUS"]:
                frame_to_send = np.ascontiguousarray(frame_to_send)

            self._cam.send(frame_to_send)
            return True
        except Exception as exc:
            self.last_error = str(exc)
            logger.warning("Error sending frame to virtual camera: %s", exc)
            return False

    def broadcast_frame(self, frame_bgr: np.ndarray, fps: Optional[int] = None) -> bool:
        """Helper to broadcast if enabled, starting or stopping as needed."""
        if not self.enabled:
            if self._is_active:
                self.stop()
            return False

        h, w = frame_bgr.shape[:2]
        if not self._is_active:
            ok = self.start(w, h, fps=fps)
            if not ok:
                return False

        return self.send(frame_bgr)

    def stop(self) -> None:
        """Stop and release the virtual camera."""
        if self._cam is not None:
            try:
                self._cam.close()
            except Exception as exc:
                logger.warning("Error closing virtual camera: %s", exc)
            finally:
                self._cam = None
        self._is_active = False

    def __enter__(self) -> "VirtualCamBroadcaster":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
