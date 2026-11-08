"""Theo dõi đối tượng qua luồng quang học (Optical Flow) và khớp vị trí.

Không bỏ qua lượt suy luận nào của mô hình AI. Các vệt theo dõi (tracks)
hết hạn dựa trên thời điểm cuối cùng được mô hình xác nhận.
"""
from dataclasses import dataclass, replace
import cv2
import numpy as np
from .boxes import normalize_box
from .documents import canonical
from ..schemas import Detection


def _key(d):
    """Xác định khóa phân loại đối tượng để ghép cặp trong theo dõi."""
    if d.class_name == 'document_region':
        return ('document',)
    return (d.source, canonical(d.class_name))


def _iou(a, b):
    """Tính tỷ lệ diện tích giao trên diện tích hợp (Intersection over Union)."""
    area = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    total = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - area
    return area / total if total else 0.0


def _association(a, b):
    """Tính điểm tương đồng giữa vệt cũ và phát hiện mới để ghép nối liên tục."""
    score = _iou(a.bbox, b.bbox)
    if a.class_name != 'document_region' and a.source not in ('document', 'card', 'cccd'):
        return score
    x, y = a.bbox, b.bbox
    old_area = (x[2] - x[0]) * (x[3] - x[1])
    new_area = (y[2] - y[0]) * (y[3] - y[1])
    if a.source != 'document' and b.source != 'document' and new_area > 2.5 * old_area:
        return 0.0  # Đề xuất vùng đột ngột mở rộng cần xác nhận riêng với phương pháp heuristic.
    intersection = max(0, min(x[2], y[2]) - max(x[0], y[0])) * max(0, min(x[3], y[3]) - max(x[1], y[1]))
    smaller = min((x[2] - x[0]) * (x[3] - x[1]), (y[2] - y[0]) * (y[3] - y[1]))
    base = max(score, 0.8 * intersection / smaller) if smaller else score
    if base >= 0.2:
        return base
    # Khớp theo khoảng cách tâm: giữ vệt không bị đứt khi thẻ xoay hoặc di chuyển nhanh
    ca = ((x[0] + x[2]) / 2, (x[1] + x[3]) / 2)
    cb = ((y[0] + y[2]) / 2, (y[1] + y[3]) / 2)
    diag = ((x[2] - x[0]) ** 2 + (x[3] - x[1]) ** 2) ** 0.5
    dist = ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5
    if diag > 0 and dist < 4.5 * diag:
        return max(0.2, 0.75 * (1.0 - dist / (4.5 * diag)))
    return base


@dataclass
class _Track:
    """Đối tượng lưu vết nội bộ cho từng vùng bảo mật."""
    detection: Detection
    seen: float
    points: object = None
    state: str = 'detected'
    hits: int = 1
    filtered_box: tuple | None = None
    velocity: tuple[float, float] = (0.0, 0.0)
    observed_box: tuple | None = None

    def __post_init__(self):
        self.observed_box = self.detection.bbox
        if self.filtered_box is None:
            self.filtered_box = tuple(map(float, self.detection.bbox))


class RegionTracker:
    """Bộ theo dõi vùng bảo mật hỗ trợ ổn định khung hình và chống nhấp nháy."""
    def __init__(self, hold_seconds=0.6, document_hold_seconds=None):
        self.hold_seconds = hold_seconds
        self.document_hold_seconds = document_hold_seconds
        self.tracks = []
        self.previous = None
        self.input_shape = None

    def reset(self):
        """Đặt lại trạng thái bộ theo dõi."""
        self.tracks = []
        self.previous = None
        self.input_shape = None

    def update(self, image, detections, now):
        """Cập nhật các vệt theo dõi với khung hình và danh sách phát hiện mới."""
        h, w = image.shape[:2]
        # Giới hạn kích thước chạy Optical Flow để duy trì tốc độ cao ngay cả với ảnh 720p/1080p.
        scale = min(1.0, 640 / max(h, w))
        gray = cv2.cvtColor(cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale)))), cv2.COLOR_BGR2GRAY)
        sy, sx = gray.shape[0] / h, gray.shape[1] / w
        if self.previous is not None and (self.previous.shape != gray.shape or self.input_shape != (h, w)):
            self.tracks = []
            self.previous = None
        self.input_shape = (h, w)

        # Lọc bỏ các vệt đã quá thời gian lưu giữ (hold timeout)
        self.tracks = [
            t for t in self.tracks
            if now - t.seen <= (
                (self.document_hold_seconds if self.document_hold_seconds is not None
                 else (self.hold_seconds if t.detection.source == 'document' else min(0.25, self.hold_seconds)))
                if t.detection.class_name == 'document_region' else self.hold_seconds
            )
        ]
        for t in self.tracks:
            t.state = 'held'

        active = [
            t for t in self.tracks
            if t.points is not None and (
                t.detection.source in ('face', 'plate', 'document') or t.detection.class_name == 'document_region'
            )
        ]
        if self.previous is not None and active:
            points = np.concatenate([t.points for t in active])
            moved, status, _ = cv2.calcOpticalFlowPyrLK(self.previous, gray, points, None, winSize=(31, 31), maxLevel=4)
            if moved is not None and status is not None:
                back, valid_back, _ = cv2.calcOpticalFlowPyrLK(gray, self.previous, moved, None, winSize=(31, 31), maxLevel=4)
                if back is not None and valid_back is not None:
                    valid = (status.ravel() == 1) & (valid_back.ravel() == 1)
                    valid &= np.linalg.norm((back - points).reshape(-1, 2), axis=1) < 1.5
                    offset = 0
                    for t in active:
                        end = offset + len(t.points)
                        keep = valid[offset:end]
                        if keep.sum() >= max(4, len(keep) * 0.5):
                            src_pts = points[offset:end][keep]
                            dst_pts = moved[offset:end][keep]
                            delta = (dst_pts - src_pts).reshape(-1, 2)
                            median = np.median(delta, axis=0)
                            coherent = np.median(np.linalg.norm(delta - median, axis=1)) <= 2.5
                            dx, dy = median / [sx, sy]
                            a, b, c, d = t.detection.bbox
                            box = normalize_box((a + dx, b + dy, c + dx, d + dy), w, h)
                            max_dx = max(60.0, 0.6 * (c - a))
                            max_dy = max(60.0, 0.6 * (d - b))
                            if box and coherent and abs(dx) <= max_dx and abs(dy) <= max_dy:
                                t.detection = replace(t.detection, bbox=box)
                                t.state = 'tracked'
                                f = t.filtered_box
                                t.filtered_box = (f[0] + dx, f[1] + dy, f[2] + dx, f[3] + dy)
                            elif t.detection.class_name == 'document_region' and keep.sum() >= 4:
                                # Xử lý xoay/nghiêng: các điểm xoay quanh tâm làm lệch chuyển động tịnh tiến thuần.
                                # Dùng estimateAffinePartial2D để bám theo góc nghiêng và cập nhật khung bao.
                                M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts)
                                if M is not None and inliers is not None and inliers.sum() >= 3:
                                    corners = np.array([[[a * sx, b * sy], [c * sx, b * sy], [c * sx, d * sy], [a * sx, d * sy]]], dtype=np.float32)
                                    trans_corners = cv2.transform(corners, M)[0] / [sx, sy]
                                    rx1, ry1 = np.min(trans_corners, axis=0)
                                    rx2, ry2 = np.max(trans_corners, axis=0)
                                    rot_box = normalize_box((rx1, ry1, rx2, ry2), w, h)
                                    if rot_box:
                                        t.detection = replace(t.detection, bbox=rot_box)
                                        t.state = 'tracked'
                                        t.filtered_box = tuple(map(float, rot_box))
                        offset = end

        # Ghép cặp giữa vệt theo dõi hiện tại và phát hiện mới từ AI
        pairs = sorted(
            (
                (_association(t.detection, d), ti, di)
                for ti, t in enumerate(self.tracks)
                for di, d in enumerate(detections)
                if _key(t.detection) == _key(d)
            ),
            reverse=True,
        )
        used_t, used_d = set(), set()
        for score, ti, di in pairs:
            if score < 0.2 or ti in used_t or di in used_d:
                continue
            t, d = self.tracks[ti], detections[di]
            old = t.filtered_box
            geometry_confirmed = True
            if d.class_name == 'document_region':
                if d.source == 'document' or d.metadata.get('region_source') == 'detector':
                    # Khung do mô hình AI phát hiện thực tế: bám sát nhanh để che chắn tức thì
                    cx = 0.1 * (old[0] + old[2]) / 2 + 0.9 * (d.bbox[0] + d.bbox[2]) / 2
                    cy = 0.1 * (old[1] + old[3]) / 2 + 0.9 * (d.bbox[1] + d.bbox[3]) / 2
                    width = 0.2 * (old[2] - old[0]) + 0.8 * (d.bbox[2] - d.bbox[0])
                    height = 0.2 * (old[3] - old[1]) + 0.8 * (d.bbox[3] - d.bbox[1])
                    smooth = (cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2)
                else:
                    # Bám theo tọa độ nhanh đồng thời giảm rung lắc kích thước cụm trường
                    cx = 0.2 * (old[0] + old[2]) / 2 + 0.8 * (d.bbox[0] + d.bbox[2]) / 2
                    cy = 0.2 * (old[1] + old[3]) / 2 + 0.8 * (d.bbox[1] + d.bbox[3]) / 2
                    width = 0.85 * (old[2] - old[0]) + 0.15 * (d.bbox[2] - d.bbox[0])
                    height = 0.85 * (old[3] - old[1]) + 0.15 * (d.bbox[3] - d.bbox[1])
                    raw_area = (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1])
                    old_area = (old[2] - old[0]) * (old[3] - old[1])
                    nested = old[0] <= d.bbox[0] and old[1] <= d.bbox[1] and old[2] >= d.bbox[2] and old[3] >= d.bbox[3]
                    if nested and raw_area < 0.6 * old_area:
                        smooth = old
                        geometry_confirmed = False
                    else:
                        smooth = (cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2)
            else:
                smooth = tuple(0.25 * a + 0.75 * b for a, b in zip(old, d.bbox))

            t.filtered_box = tuple(smooth)
            # Khung bao hiển thị luôn bao trọn phát hiện mới, không để hở cạnh viền
            box = normalize_box(
                (min(smooth[0], d.bbox[0]), min(smooth[1], d.bbox[1]),
                 max(smooth[2], d.bbox[2]), max(smooth[3], d.bbox[3])),
                w, h,
            )
            t.detection = replace(d, bbox=box)
            if geometry_confirmed:
                elapsed = now - t.seen
                if d.class_name == 'document_region' and elapsed > 0:
                    a, b = t.observed_box, d.bbox
                    t.velocity = (
                        (b[0] + b[2] - a[0] - a[2]) / (2 * elapsed),
                        (b[1] + b[3] - a[1] - a[3]) / (2 * elapsed),
                    )
                    t.observed_box = d.bbox
                t.seen = now
                t.state = 'detected'
                t.hits += 1
            else:
                t.state = 'partial'
            used_t.add(ti)
            used_d.add(di)

        # Thêm các phát hiện mới chưa có trong danh sách vệt theo dõi
        self.tracks.extend(_Track(d, now) for i, d in enumerate(detections) if i not in used_d)
        for ti, t in enumerate(self.tracks):
            if ti not in used_t and t.hits < 2 and t.state == 'held':
                t.hits = 0  # Tránh xác nhận nhầm nếu chỉ là đề xuất lẻ loi một khung hình

        # Thu hồi các vệt cũ bị bao bọc khi đã có vệt lớn hơn được xác nhận
        confirmed = [
            t for t in self.tracks
            if t.detection.class_name == 'document_region' and t.hits >= 2 and t.state == 'detected'
        ]
        self.tracks = [
            t for t in self.tracks
            if not (
                t.detection.class_name == 'document_region'
                and t.state == 'held'
                and any(
                    c is not t
                    and c.detection.bbox[0] <= t.detection.bbox[0]
                    and c.detection.bbox[1] <= t.detection.bbox[1]
                    and c.detection.bbox[2] >= t.detection.bbox[2]
                    and c.detection.bbox[3] >= t.detection.bbox[3]
                    for c in confirmed
                )
            )
        ]

        result = []
        for t in self.tracks:
            if t.detection.class_name == 'document_region' and t.detection.source != 'document' and t.hits < 2:
                continue
            render_box = t.detection.bbox
            if t.detection.class_name == 'document_region' and t.state == 'held':
                if t.detection.source != 'document':
                    a, b, c, d = render_box
                    age = max(0.0, now - t.seen)
                    dx = float(np.clip(t.velocity[0] * age, -0.2 * (c - a), 0.2 * (c - a)))
                    dy = float(np.clip(t.velocity[1] * age, -0.2 * (d - b), 0.2 * (d - b)))
                    render_box = (min(a, a + dx) - abs(dx), min(b, b + dy) - abs(dy),
                                  max(c, c + dx) + abs(dx), max(d, d + dy) + abs(dy))
                else:
                    # Mô hình thực tế: suy luận theo vận tốc để chuyển động nhanh không làm rơi mất vùng che
                    a, b, c, d = t.filtered_box or t.detection.bbox
                    age = max(0.0, now - t.seen)
                    dx = float(np.clip(t.velocity[0] * age, -0.6 * (c - a), 0.6 * (c - a)))
                    dy = float(np.clip(t.velocity[1] * age, -0.6 * (d - b), 0.6 * (d - b)))
                    render_box = (a + dx, b + dy, c + dx, d + dy)
            box = normalize_box(render_box, w, h)
            if box is None:
                t.points = None
                continue
            rendered = replace(t.detection, bbox=box)
            x1, y1, x2, y2 = normalize_box((box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy), gray.shape[1], gray.shape[0])
            if t.state in ('detected', 'tracked'):
                if t.state == 'detected' or t.points is None or len(t.points) < 8:
                    points = (
                        cv2.goodFeaturesToTrack(gray[y1:y2, x1:x2], maxCorners=32, qualityLevel=0.005, minDistance=4)
                        if (t.detection.source in ('face', 'plate', 'document') or t.detection.class_name == 'document_region')
                        else None
                    )
                    if points is not None:
                        points += np.float32([x1, y1])
                    t.points = points
            result.append(
                replace(
                    rendered,
                    metadata={
                        **t.detection.metadata,
                        'tracking_state': t.state,
                        'seconds_since_detection': now - t.seen,
                    },
                )
            )
        self.previous = gray
        return result
