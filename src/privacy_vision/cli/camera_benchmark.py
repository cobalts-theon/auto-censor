"""Camera-only timings. No frame is returned, shown, logged, or saved."""
from time import perf_counter
import numpy as np


def measure_camera(cap, seconds=3., warmup=10):
    if seconds <= 0 or warmup < 0:
        raise ValueError('seconds must be positive and warmup nonnegative')
    for _ in range(warmup):
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError('Camera read failed during warmup')
    durations = []
    start = perf_counter()
    while perf_counter()-start < seconds:
        tick = perf_counter()
        ok, frame = cap.read()
        end = perf_counter()
        if not ok or frame is None:
            raise RuntimeError('Camera read failed')
        durations.append((end-tick)*1000)
    elapsed = perf_counter()-start
    return {'frames': len(durations), 'elapsed_s': elapsed,
            'fps': len(durations)/elapsed,
            'read_mean_ms': float(np.mean(durations)) if durations else 0.,
            'read_p95_ms': float(np.percentile(durations,95)) if durations else 0.}


def main():
    import argparse
    import json
    from ..desktop.camera import CameraOptions, open_camera
    p=argparse.ArgumentParser(description='Đo camera không AI, không hiển thị/lưu ảnh. Hãy dừng webcam desktop trước.')
    p.add_argument('--camera',type=int,default=1)
    p.add_argument('--seconds',type=float,default=3.)
    p.add_argument('--fps',type=int,default=30)
    p.add_argument('--buffers',default='1,2,4',help='Các buffer driver cần so sánh')
    p.add_argument('--formats',default='MJPG',help='MJPG hoặc MJPG,YUYV để so sánh định dạng')
    p.add_argument('--compare-exposure',action='store_true',help='So sánh giữ nguyên exposure và manual 80/160; cố khôi phục sau mỗi lượt')
    args=p.parse_args()
    try:
        buffers=[int(b.strip()) for b in args.buffers.split(',')]
        if args.seconds <= 0 or not buffers:
            raise ValueError('Invalid arguments')
        formats=[f.strip() for f in args.formats.split(',')]
        exposures=[None,80,160] if args.compare_exposure else [None]
        configs=[CameraOptions(fps=args.fps,buffers=b,fourcc=f,
                 manual_exposure=e is not None,exposure=e or 160)
                 for f in formats for e in exposures for b in buffers]
    except ValueError:
        p.error('seconds > 0, fps > 0, buffers nằm trong 1..10')
    failed=False
    for options in configs:
        cap=None
        try:
            cap=open_camera(args.camera,options)
            report=measure_camera(cap,args.seconds)
            settings=cap.info
            restored=cap.release()
            cap=None
            print(json.dumps({'requested_buffers':options.buffers,'camera':args.camera,
                              'requested_format':options.fourcc,
                              'requested_exposure':options.exposure if options.manual_exposure else 'keep',
                              'exposure_restored':restored,
                              'settings':settings,**report},ensure_ascii=False))
        except Exception as exc:
            failed=True
            print(json.dumps({'requested_buffers':options.buffers,'error':type(exc).__name__,
                              'message':'Không đo được camera; kiểm tra camera có đang được ứng dụng khác sử dụng.'},ensure_ascii=False))
        finally:
            if cap is not None:
                cap.release()
    return 1 if failed else 0
