"""Local capture-to-redaction benchmark. Does not display or save frames."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import json
from time import perf_counter
import numpy as np
from privacy_vision.cli.image import parser, config_from_args
from privacy_vision.pipeline import PrivacyPipeline, validate_output
from privacy_vision.desktop.capture import LatestCapture
from privacy_vision.desktop.session import PreviewBuffer, open_capture
from privacy_vision.processing.tracking import RegionTracker


def main():
    p = parser(benchmark=True)
    for action in p._actions:
        if action.dest == 'input':
            action.required = False
    p.add_argument('--camera', type=int, default=0)
    p.add_argument('--duration', type=float, default=30.)
    args = p.parse_args()
    capture = None
    try:
        if not np.isfinite(args.duration) or args.duration <= 0 or args.warmup < 0:
            raise ValueError('Invalid benchmark duration/warmup')
        config = config_from_args(args)
        if not config.detectors:
            raise ValueError('At least one detector is required')
        pipeline = PrivacyPipeline(config)
        import torch
        cuda = getattr(pipeline, 'device', 'cpu') != 'cpu' and torch.cuda.is_available()
        if cuda:
            torch.cuda.reset_peak_memory_stats(int(pipeline.device))
        capture = LatestCapture(args.camera, open_capture, PreviewBuffer())
        capture.start()
        tracker = RegionTracker()
        for _ in range(args.warmup):
            item = capture.take(lambda: False)
            pipeline.process(item.image, tracker)
        samples, blackouts = [], 0
        initial_read, initial_drop = capture.frames_read, capture.frames_dropped
        started = perf_counter()
        while perf_counter()-started < args.duration:
            wait = perf_counter()
            item = capture.take(lambda: perf_counter()-started >= args.duration)
            if item is None:
                break
            taken = perf_counter()
            result = pipeline.process(item.image, tracker)
            validate_output(result.image, item.image.shape)
            done = perf_counter()
            samples.append({**result.timings_ms, 'capture_wait': (taken-wait)*1000,
                            'read_to_redaction': (done-item.read_at)*1000})
            blackouts += int(result.strict_blackout)
        elapsed = perf_counter()-started
        if not samples:
            raise ValueError('No benchmark samples')
        report = {'frames_processed': len(samples), 'elapsed_seconds': elapsed,
                  'fps_processed': len(samples)/elapsed,
                  'fps_capture': (capture.frames_read-initial_read)/elapsed,
                  'frames_replaced': capture.frames_dropped-initial_drop,
                  'strict_blackout_frames': blackouts, 'device': pipeline.device,
                  'configuration': {'mode': config.document_redaction_mode,
                      'strategy': config.document_detection_strategy, 'strict': config.strict_privacy_mode,
                      'imgsz': {c.source: c.imgsz for c in config.detectors}},
                  'latency_ms': {k: {'p50': float(np.percentile([s[k] for s in samples],50)),
                                     'p95': float(np.percentile([s[k] for s in samples],95))}
                                 for k in samples[0]},
                  'peak_torch_vram_bytes': torch.cuda.max_memory_allocated(int(pipeline.device)) if cuda else None,
                  'warning': 'No GUI output measured. Native capture/inference cannot be interrupted by duration.'}
        print(json.dumps(report, indent=2))
    except Exception as exc:
        p.exit(1, f'Benchmark failed: {type(exc).__name__}; no images saved.\n')
    finally:
        if capture is not None:
            capture.stop()

if __name__ == '__main__':
    main()
