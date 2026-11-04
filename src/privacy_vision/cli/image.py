import argparse
import json
from pathlib import Path
from ..config import MODEL_FILES, DetectorConfig, PipelineConfig
from ..pipeline import PrivacyPipeline
from ..image_io import atomic_write, read_image


def parser(benchmark=False):
    p = argparse.ArgumentParser(description="PrivacyVision: xử lý ảnh cục bộ, không GUI")
    p.add_argument('--input', required=True, help='Ảnh BGR được đọc từ file cục bộ')
    p.add_argument('--output', required=not benchmark, help='Ảnh đã che; ưu tiên .png')
    p.add_argument('--mode', choices=['black', 'blur', 'pixelate'], default='black')
    p.add_argument('--enable', default='face,plate,card,cccd', help='Detector cách nhau bởi dấu phẩy; chuỗi rỗng để tắt hết')
    p.add_argument('--models-dir', type=Path, default=Path('models'))
    for s in ['plate', 'card', 'cccd']:
        p.add_argument('--conf-'+s, type=float, default=0.25)
    p.add_argument('--face-threshold', type=float, default=0.65)
    p.add_argument('--margin', type=float, default=0.1, help='Tỷ lệ mở rộng MỖI phía, 0.1 = 10%%')
    p.add_argument('--device', default='auto', help='auto, cpu hoặc chỉ số GPU (0)')
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--document-rotations', default='0', help='Góc xoay thử cho card/cccd: 0,90,180,270; nhiều góc sẽ chậm hơn')
    p.add_argument('--whole-documents', action='store_true', help='Thử che toàn giấy tờ; không rõ biên sẽ chỉ che trường phát hiện')
    for s in MODEL_FILES:
        p.add_argument('--imgsz-'+s, type=int, help='Ghi đè imgsz riêng detector')
    p.add_argument('--document-redaction-mode', choices=['fields', 'full'])
    p.add_argument('--document-detection-strategy', choices=['heuristic', 'detector', 'hybrid'], default='heuristic')
    p.add_argument('--document-checkpoint', type=Path)
    p.add_argument('--imgsz-document', type=int, default=640)
    p.add_argument('--document-margin', type=float, nargs=2, default=(.25, .30), metavar=('X', 'Y'))
    p.add_argument('--document-min-fields', type=int, default=3)
    p.add_argument('--strict-privacy-mode', action='store_true')
    p.add_argument('--metadata-output', type=Path)
    if benchmark:
        p.add_argument('--warmup', type=int, default=1)
        p.add_argument('--runs', type=int, default=5)
    return p


def main(benchmark=False):
    p = parser(benchmark)
    args = p.parse_args()
    try:
        enabled = [s.strip() for s in args.enable.split(',') if s.strip()]
        if len(set(enabled)) != len(enabled) or set(enabled)-MODEL_FILES.keys():
            raise ValueError('Invalid --enable')
        paths = [Path(args.input)] + ([Path(args.output)] if args.output else [])
        if args.metadata_output:
            paths.append(args.metadata_output)
        if len({x.resolve() for x in paths}) != len(paths):
            raise ValueError('Input, output and metadata paths must differ')
        if not Path(args.input).is_file():
            p.exit(1, 'Không tìm thấy file ảnh --input. Hãy chọn ảnh có thật; inputs/test.jpg chỉ là đường dẫn ví dụ.\n')
        config = config_from_args(args)
        for detector_config in config.detectors:
            if not detector_config.checkpoint.is_file():
                p.exit(1, f'Thiếu checkpoint cho detector {detector_config.source}; kiểm tra --models-dir.\n')
        pipeline = PrivacyPipeline(config)
        if benchmark:
            import numpy as np
            if args.warmup < 0 or args.runs < 1:
                raise ValueError('runs >= 1 and warmup >= 0 required')
            image = read_image(args.input)
            for _ in range(args.warmup):
                pipeline.process(image)
            samples = [pipeline.process(image).timings_ms for _ in range(args.runs)]
            report = {'runs': args.runs, 'warmup': args.warmup,
                      'device': getattr(pipeline, 'device', args.device),
                      'configuration': {'document_mode': config.document_redaction_mode, 'strategy': config.document_detection_strategy,
                                        'strict': config.strict_privacy_mode,
                                        'imgsz': {c.source: c.imgsz for c in config.detectors}},
                      'timings_ms': {k: {'mean': float(np.mean([s[k] for s in samples])),
                          'p50': float(np.percentile([s[k] for s in samples], 50)),
                          'p95': float(np.percentile([s[k] for s in samples], 95))} for k in samples[0]}}
        else:
            result = pipeline.process_file(args.input, args.output)
            report = result.to_metadata()
            print('Đã lưu ảnh đã che. Cần kiểm tra thủ công trước khi chia sẻ.')
        if args.metadata_output:
            atomic_write(args.metadata_output, json.dumps(report, ensure_ascii=False, indent=2).encode())
        if benchmark:
            print(json.dumps(report, indent=2))
    except Exception as exc:
        # No raw image, predictions, OCR or arbitrary exception details in logs.
        p.exit(1, f'Lỗi {type(exc).__name__}: xử lý không hoàn tất. Kiểm tra cấu hình, model và ảnh đầu vào.\n')


def config_from_args(args):
    enabled = [s.strip() for s in args.enable.split(',') if s.strip()]
    if len(set(enabled)) != len(enabled) or set(enabled) - MODEL_FILES.keys():
        raise ValueError('Invalid --enable')
    return PipelineConfig([
        DetectorConfig(s, args.models_dir / MODEL_FILES[s],
            args.face_threshold if s == 'face' else getattr(args, 'conf_' + s),
            getattr(args, 'imgsz_' + s) or args.imgsz) for s in enabled
    ], args.device, args.mode, args.margin,
        tuple(int(a.strip()) for a in args.document_rotations.split(',')), args.whole_documents,
        document_redaction_mode=args.document_redaction_mode,
        document_detection_strategy=args.document_detection_strategy,
        document_margin=tuple(args.document_margin), document_min_fields=args.document_min_fields,
        strict_privacy_mode=args.strict_privacy_mode,
        document_detector_config=DetectorConfig('document', args.document_checkpoint, imgsz=args.imgsz_document)
            if args.document_checkpoint else None)
