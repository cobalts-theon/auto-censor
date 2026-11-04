from dataclasses import replace
from time import perf_counter
import math
import numpy as np
from .config import PipelineConfig
from .schemas import PipelineResult, DocumentRegion
from .processing.boxes import normalize_box, unrotate_box
from .processing.redaction import redact
from .image_io import read_image, write_image

class PrivacyPipeline:
    def __init__(self, config: PipelineConfig | None = None, detectors=None, document_detector=None):
        self.config = config or PipelineConfig()
        if detectors is None:
            from .detectors.yunet import YuNetDetector
            from .detectors.yolo import YoloDetector, resolve_device
            self.device = resolve_device(self.config.device)
            detectors = [YuNetDetector(c) if c.source == "face" else YoloDetector(c, self.device)
                         for c in self.config.detectors]
        self.detectors = list(detectors)
        self.document_detector = document_detector
        if self.config.document_detection_strategy != 'heuristic' and document_detector is None:
            from .detectors.document import DocumentDetector
            from .detectors.yolo import resolve_device
            self.device = resolve_device(self.config.device)
            self.document_detector = DocumentDetector(self.config.document_detector_config, self.device)
        if [d.source for d in self.detectors] != [c.source for c in self.config.detectors]:
            raise ValueError("Injected detectors must match enabled config sources/order")

    def process(self, image: np.ndarray, tracker=None) -> PipelineResult:
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
            raise ValueError("Expected nonempty uint8 BGR image")
        start = perf_counter()
        detections, timings = [], {}
        h, w = image.shape[:2]
        pairs = list(zip(self.detectors, self.config.detectors))
        if self.config.document_detection_strategy != 'heuristic':
            pairs.append((self.document_detector, self.config.document_detector_config))
        for detector, config in pairs:
            tick = perf_counter()
            try:
                rows = []
                angles = self.config.document_rotations if detector.source in ('card', 'cccd') else (0,)
                for angle in angles:
                    rotated = image if angle == 0 else np.ascontiguousarray(np.rot90(image, angle // 90))
                    for d in detector.detect(rotated):
                        if len(d.bbox) != 4 or not all(math.isfinite(v) for v in d.bbox) or d.bbox[2] <= d.bbox[0] or d.bbox[3] <= d.bbox[1]:
                            raise ValueError('Invalid detector bbox')
                        rows.append(replace(d, bbox=unrotate_box(d.bbox, w, h, angle),
                            metadata={**d.metadata, 'rotation_degrees': angle}))
                for d in rows:
                    if d.source != detector.source or not math.isfinite(d.confidence) or not 0 <= d.confidence <= 1:
                        raise ValueError("Invalid detector output")
                    if detector.source == 'document':
                        from .detectors.document import DOCUMENT_CLASSES
                        if d.class_name not in DOCUMENT_CLASSES:
                            raise ValueError('Unsupported document class')
                    box = normalize_box(d.bbox, w, h, 0 if detector.source == 'document' else self.config.margin)
                    if box and d.confidence >= config.confidence:
                        detections.append(replace(d, bbox=box))
            except Exception as exc:
                raise RuntimeError(f"Detector {detector.source} failed; no image exported") from exc
            timings[detector.source] = (perf_counter()-tick)*1000
        timings["inference"] = (perf_counter()-start)*1000
        tick = perf_counter()
        current = list(detections)
        fields = [d for d in current if d.source in ('card', 'cccd')]
        documents = [d for d in current if d.source == 'document']
        detections = [d for d in current if d.source != 'document']
        regions = []
        def contains(box, field):
            a = field.bbox
            return box[0] <= a[0] and box[1] <= a[1] and box[2] >= a[2] and box[3] >= a[3]
        for d in documents:
            evidence = [f for f in fields if contains(d.bbox, f)]
            source = 'hybrid' if evidence and self.config.document_detection_strategy == 'hybrid' else 'detector'
            regions.append(DocumentRegion(d.bbox, d.confidence, d.class_name, source, len(evidence), evidence))
        if self.config.whole_documents and self.config.document_detection_strategy in ('heuristic', 'hybrid'):
            from .processing.documents import document_regions
            remaining = [f for f in fields if not any(contains(d.bbox, f) for d in documents)]
            proposals = document_regions(image, remaining, self.config.document_margin, self.config.document_min_fields)
            for d in proposals:
                if d.class_name == 'document_region':
                    evidence = d.metadata['supporting_detections']
                    hints = {f.class_name for f in evidence} & {'cccd', 'bhyt', 'gplx'}
                    kind = next(iter(hints)) if len(hints) == 1 else None
                    regions.append(DocumentRegion(d.bbox, None, kind, 'heuristic', d.metadata['field_count'], evidence))
        region_detections = []
        if self.config.whole_documents:
            from .schemas import Detection
            for r in regions:
                region_detections.append(Detection('document' if r.source != 'heuristic' else 'card', -1,
                    'document_region', r.confidence if r.confidence is not None else 0.0, r.bbox,
                    {'derived_region': True, 'region_source': r.source,
                     'confidence_kind': 'unscored_heuristic' if r.confidence is None else 'model_score'}))
            detections.extend(region_detections)
        timings['document_regions'] = (perf_counter()-tick)*1000
        tick = perf_counter()
        if tracker is not None:
            # Run tracking on ORIGINAL frame, then redact exactly once.
            tracked, untracked = [], []
            for d in detections:
                # Small field masks use only CURRENT detections. Holding dozens
                # of old field tracks creates ghost rectangles around documents.
                field = d.source in ('card','cccd') and d.class_name != 'document_region'
                (untracked if field else tracked).append(d)
            detections = tracker.update(image, tracked, perf_counter()) + untracked + region_detections
            timings['tracking'] = (perf_counter()-tick)*1000
            tick = perf_counter()
        reasons = ['scene_not_verified']
        risk = []
        if not current:
            risk.append('no_current_detections')
        if any(not any(contains(d.bbox, f) for d in documents) for f in fields):
            risk.append('fields_without_detector_region')
        if any(r.source == 'heuristic' for r in regions):
            risk.append('heuristic_boundary')
        if self.config.whole_documents and not documents:
            risk.append('no_detector_document_region')
        if any(d.metadata.get('tracking_state', 'detected') != 'detected' for d in detections):
            risk.append('tracking_unconfirmed')
        blackout = self.config.strict_privacy_mode and bool(risk)
        output = np.zeros_like(image) if blackout else redact(image, detections, self.config.mode)
        validate_output(output, image.shape)
        timings["redaction"] = (perf_counter()-tick)*1000
        timings["total"] = (perf_counter()-start)*1000
        return PipelineResult(output, detections, timings, regions, "coverage_unverified", reasons + risk, blackout)

    def process_file(self, input_path, output_path):
        from pathlib import Path
        if Path(input_path).resolve() == Path(output_path).resolve():
            raise ValueError("Input and output must differ")
        result = self.process(read_image(input_path))
        write_image(output_path, result.image)
        return result


def validate_output(output, shape):
    if not isinstance(output, np.ndarray) or output.dtype != np.uint8 or output.shape != shape:
        raise ValueError('Invalid redaction output')
