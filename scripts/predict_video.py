#!/usr/bin/env python3
"""PrivacyVision: Che thông tin cá nhân trong file video (.mp4, .avi, .mov, .mkv)."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from privacy_vision.cli.video import main

if __name__ == '__main__':
    main()
