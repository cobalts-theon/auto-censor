"""Unit tests for Virtual Camera broadcaster (pyvirtualcam / Google Meet / Zoom)."""
from unittest.mock import Mock, patch
import numpy as np
import pytest

from privacy_vision.virtualcam import (
    check_virtual_cam_support,
    VirtualCamBroadcaster,
)


def test_check_virtual_cam_support_schema():
    info = check_virtual_cam_support()
    assert isinstance(info, dict)
    for key in ("available", "installed", "device", "backend", "message"):
        assert key in info


def test_virtual_cam_broadcaster_default_state():
    b = VirtualCamBroadcaster(device="/dev/video99", fps=25, enabled=False)
    assert not b.is_active
    assert not b.enabled
    assert b.device_name == "PrivacyVision Cam"
    assert b.actual_backend is None


def test_virtual_cam_broadcaster_mock_lifecycle():
    fake_cam = Mock()
    fake_cam.device = "/dev/video0"
    fake_cam.backend = "v4l2loopback"

    with patch("pyvirtualcam.Camera", return_value=fake_cam) as mock_cam_cls:
        b = VirtualCamBroadcaster(device="/dev/video0", fps=30)
        assert b.start(1280, 720, fps=30)
        assert b.is_active
        assert b.device_name == "/dev/video0"
        assert b.actual_backend == "v4l2loopback"

        # Calling start again with same size does not recreate
        assert b.start(1280, 720, fps=30)
        assert mock_cam_cls.call_count == 1

        # Calling start with different size recreates
        assert b.start(640, 480, fps=30)
        assert mock_cam_cls.call_count == 2
        fake_cam.close.assert_called()

        # Stop closes camera
        b.stop()
        assert not b.is_active


def test_virtual_cam_broadcaster_send_and_auto_resize():
    fake_cam = Mock()
    with patch("pyvirtualcam.Camera", return_value=fake_cam):
        b = VirtualCamBroadcaster(device="/dev/video0")
        assert not b.send(np.zeros((720, 1280, 3), dtype=np.uint8))  # not started yet

        b.start(1280, 720, fps=30)

        # Same size
        frame_hd = np.zeros((720, 1280, 3), dtype=np.uint8)
        assert b.send(frame_hd)
        assert fake_cam.send.call_count == 1
        sent_shape = fake_cam.send.call_args[0][0].shape
        assert sent_shape == (720, 1280, 3)

        # Different size (auto-resize)
        frame_vga = np.zeros((480, 640, 3), dtype=np.uint8)
        assert b.send(frame_vga)
        assert fake_cam.send.call_count == 2
        sent_shape_2 = fake_cam.send.call_args[0][0].shape
        assert sent_shape_2 == (720, 1280, 3)


def test_virtual_cam_broadcaster_broadcast_frame_flow():
    fake_cam = Mock()
    with patch("pyvirtualcam.Camera", return_value=fake_cam):
        b = VirtualCamBroadcaster(device="/dev/video0", enabled=False)
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        # Disabled: does not start or send
        assert not b.broadcast_frame(frame)
        assert not b.is_active

        # Enable: starts and sends
        b.enabled = True
        assert b.broadcast_frame(frame)
        assert b.is_active
        assert fake_cam.send.call_count == 1

        # Disable again: stops camera
        b.enabled = False
        assert not b.broadcast_frame(frame)
        assert not b.is_active
        fake_cam.close.assert_called()


def test_virtual_cam_error_handling():
    with patch("pyvirtualcam.Camera", side_effect=RuntimeError("Device busy")):
        b = VirtualCamBroadcaster(device="/dev/video0")
        assert not b.start(1280, 720)
        assert not b.is_active
        assert "Device busy" in str(b.last_error)


def test_virtual_cam_context_manager():
    fake_cam = Mock()
    with patch("pyvirtualcam.Camera", return_value=fake_cam):
        with VirtualCamBroadcaster() as b:
            b.start(640, 480)
            assert b.is_active
        fake_cam.close.assert_called()
        assert not b.is_active
