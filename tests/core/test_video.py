import tempfile
from pathlib import Path
from unittest.mock import Mock
import cv2
import numpy as np
import pytest

from privacy_vision.cli.video import process_video, parser
from privacy_vision.schemas import PipelineResult, Detection


def create_synthetic_video(path, num_frames=10, width=160, height=120, fps=25.0):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    for i in range(num_frames):
        frame = np.full((height, width, 3), (i * 20) % 256, dtype=np.uint8)
        # Draw a small rectangle simulating an object
        cv2.rectangle(frame, (20, 20), (60, 60), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()


def test_process_video_basic(tmp_path):
    in_video = tmp_path / "input.mp4"
    out_video = tmp_path / "output.mp4"
    create_synthetic_video(in_video, num_frames=8, width=160, height=120, fps=20.0)

    # Mock pipeline returning processed frame and 1 detection
    mock_pipeline = Mock()
    det = Detection("face", 0, "face", 0.95, (20, 20, 60, 60))
    mock_pipeline.process.side_effect = lambda frame, tracker=None: PipelineResult(
        image=frame, detections=[det], timings_ms={"total": 5.0}
    )

    cb_calls = []
    def progress_cb(current, total, fps, elapsed):
        cb_calls.append((current, total))

    report = process_video(
        in_video, out_video, mock_pipeline,
        tracker=None, max_frames=None, preview=False, progress_callback=progress_cb
    )

    assert report["frames_processed"] == 8
    assert report["total_frames"] == 8
    assert report["resolution"] == [160, 120]
    assert report["counts"]["face"] == 8
    assert len(cb_calls) == 8
    assert out_video.is_file()
    assert out_video.stat().st_size > 0

    # Verify output video is readable
    cap = cv2.VideoCapture(str(out_video))
    assert cap.isOpened()
    read_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        assert frame.shape == (120, 160, 3)
        read_count += 1
    cap.release()
    assert read_count == 8


def test_process_video_max_frames(tmp_path):
    in_video = tmp_path / "input.mp4"
    out_video = tmp_path / "output_partial.mp4"
    create_synthetic_video(in_video, num_frames=15, width=160, height=120)

    mock_pipeline = Mock()
    mock_pipeline.process.side_effect = lambda frame, tracker=None: PipelineResult(
        image=frame, detections=[], timings_ms={"total": 2.0}
    )

    report = process_video(
        in_video, out_video, mock_pipeline,
        tracker=None, max_frames=5, preview=False
    )

    assert report["frames_processed"] == 5
    assert report["total_frames"] == 5

    cap = cv2.VideoCapture(str(out_video))
    read_count = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        read_count += 1
    cap.release()
    assert read_count == 5


def test_process_video_nonexistent_input(tmp_path):
    mock_pipeline = Mock()
    with pytest.raises(FileNotFoundError):
        process_video(tmp_path / "missing.mp4", tmp_path / "out.mp4", mock_pipeline)


def test_video_cli_parser():
    p = parser()
    args = p.parse_args(["--input", "in.mp4", "--output", "out.mp4", "--mode", "blur", "--no-tracking"])
    assert args.input == Path("in.mp4")
    assert args.output == Path("out.mp4")
    assert args.mode == "blur"
    assert args.no_tracking is True
