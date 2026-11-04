"""PrivacyVision desktop entry point; legacy OpenCV demo is in examples/."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

if __name__ == '__main__':
    from privacy_vision.desktop.window import main
    raise SystemExit(main())
