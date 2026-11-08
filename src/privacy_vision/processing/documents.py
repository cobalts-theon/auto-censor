"""Suy luận vùng bao tài liệu/thẻ từ các cụm trường thông tin cục bộ (Heuristic Document Clustering).

Tự động gom cụm các trường thông tin CCCD, BHYT, GPLX để xác định khung thẻ mà không cần mô hình toàn khung.
"""
from dataclasses import replace
import numpy as np
from .boxes import normalize_box
from ..schemas import Detection

# Các trường đặc trưng định danh tài liệu quan trọng
HINTS = {'cccd', 'bhyt', 'gplx', 'qr', 'finger_print'}


def canonical(name: str) -> str:
    """Chuẩn hóa tên trường định danh (ví dụ id_1, id_2 thành id)."""
    if name in ('id', 'id_'):
        return 'id'
    if name.endswith(('1', '2')):
        return name[:-1]
    return name


def _union(boxes):
    """Tính khung bao hợp nhất bao trùm tất cả các ô tọa độ đầu vào."""
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _near(a, b):
    """Kiểm tra hai trường thông tin có nằm gần nhau trong phạm vi một phôi thẻ hay không."""
    aw, ah = a[2] - a[0], a[3] - a[1]
    bw, bh = b[2] - b[0], b[3] - b[1]
    gx = max(0, max(a[0], b[0]) - min(a[2], b[2]))
    gy = max(0, max(a[1], b[1]) - min(a[3], b[3]))
    # Các dòng chữ có thể cách nhau theo chiều dọc; QR hoặc vân tay có thể nằm cạnh văn bản
    return gx <= 0.65 * max(aw, bw) and gy <= max(3 * max(ah, bh), 0.5 * max(aw, bw))


def _compatible(d, cluster):
    """Kiểm tra trường mới có hợp lý khi gộp vào cụm trường thông tin hiện tại không."""
    singletons = {'id', 'name', 'dob', 'gender', 'qr', 'finger_print', 'issue_date', 'expire_date'}
    for other in cluster:
        if not _near(d.bbox, other.bbox):
            return False
        if canonical(d.class_name) == canonical(other.class_name) and canonical(d.class_name) in singletons:
            a, b = d.bbox, other.bbox
            overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
            smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
            if overlap < 0.3 * smaller:
                return False  # Hai số thẻ hoặc hai họ tên riêng biệt chứng tỏ có hai thẻ khác nhau
    return True


def document_regions(image: np.ndarray, fields: list[Detection], margin=(0.25, 0.30), min_fields=3) -> list[Detection]:
    """Gom cụm các trường thông tin cục bộ thành vùng khung bao tài liệu hoàn chỉnh."""
    anchors = [
        replace(d, bbox=box)
        for d in fields
        if d.source in ('card', 'cccd') and not d.metadata.get('derived_region')
        if (box := normalize_box(d.bbox, image.shape[1], image.shape[0])) is not None
    ]
    h, w = image.shape[:2]
    clusters = []
    # Thứ tự sắp xếp ổn định tránh tình trạng nhảy cụm khi tọa độ chữ lệch 1 pixel giữa các frame
    for d in sorted(anchors, key=lambda d: (canonical(d.class_name), d.bbox[0], d.bbox[1], d.source)):
        compatible = [c for c in clusters if _compatible(d, c)]
        if compatible:
            cluster = min(
                compatible,
                key=lambda c: sum(abs((x.bbox[0] + x.bbox[2]) - (d.bbox[0] + d.bbox[2])) for x in c) / len(c),
            )
            cluster.append(d)
        else:
            clusters.append([d])

    output = []
    for cluster in clusters:
        names = {canonical(d.class_name) for d in cluster}
        enough = len(names) >= min_fields or (len(names) >= 2 and bool(names & HINTS))
        if not enough:
            for d in cluster:
                output.append(
                    replace(
                        d,
                        class_id=-1,
                        class_name='document_boundary_unresolved',
                        metadata={**d.metadata, 'method': 'fields_only', 'derived_region': True},
                    )
                )
            continue
        x1, y1, x2, y2 = _union([d.bbox for d in cluster])
        dx, dy = margin[0] * (x2 - x1), margin[1] * (y2 - y1)
        box = normalize_box((x1 - dx, y1 - dy, x2 + dx, y2 + dy), w, h)
        if box is None or (box[2] - box[0]) * (box[3] - box[1]) > 0.75 * w * h:
            output.extend(
                replace(
                    d,
                    class_id=-1,
                    class_name='document_boundary_unresolved',
                    metadata={**d.metadata, 'method': 'fields_only', 'derived_region': True},
                )
                for d in cluster
            )
            continue
        output.append(
            Detection(
                'card',
                -1,
                'document_region',
                0.0,
                box,
                {
                    'method': 'field_cluster_union',
                    'derived_region': True,
                    'field_count': len(names),
                    'unique_fields': len(names),
                    'region_source': 'heuristic',
                    'confidence_kind': 'unscored_heuristic',
                    'supporting_detections': list(cluster),
                },
            )
        )

    # Loại bỏ các vùng bao lồng nhau trùng lặp để tránh sinh nhiều vệt theo dõi thừa
    regions = sorted(
        (d for d in output if d.class_name == 'document_region'),
        key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]),
        reverse=True,
    )
    kept = []
    for d in regions:
        if not any(
            k.bbox[0] <= d.bbox[0]
            and k.bbox[1] <= d.bbox[1]
            and k.bbox[2] >= d.bbox[2]
            and k.bbox[3] >= d.bbox[3]
            and all(_compatible(e, k.metadata['supporting_detections']) for e in d.metadata['supporting_detections'])
            for k in kept
        ):
            kept.append(d)
    return kept + [d for d in output if d.class_name != 'document_region']
