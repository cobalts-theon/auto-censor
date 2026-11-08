"""Cấu hình chung cho bộ kiểm thử Pytest của dự án PrivacyVision."""
import os
import pytest


def pytest_unconfigure(config):
    """Xử lý thoát sạch tiến trình sau khi Pytest đã hoàn thành in toàn bộ báo cáo kiểm thử.

    Tránh lỗi SIGBUS (135) do cơ chế giải phóng bộ nhớ C++ của PySide6 khi chạy trên ổ SSD USB ngoài.
    """
    tr = config.pluginmanager.get_plugin("terminalreporter")
    status = tr._session.exitstatus if tr and hasattr(tr, "_session") else 0
    os._exit(status)
