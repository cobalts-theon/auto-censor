import os
import pytest

def pytest_sessionfinish(session, exitstatus):
    # PySide6 C++ destruction on external USB SSD triggers glibc SIGBUS (135) on process exit.
    # Exiting directly with the test session exitstatus ensures clean exit code.
    os._exit(session.exitstatus)
