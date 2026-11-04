import math

def normalize_box(box, width: int, height: int, margin: float = 0):
    """Half-open integer bbox; margin is added on EACH side before clipping."""
    if len(box) != 4 or not all(math.isfinite(v) for v in box):
        return None
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return None
    dx, dy = (x2-x1)*margin, (y2-y1)*margin
    x1, y1 = max(0, min(width, math.floor(x1-dx))), max(0, min(height, math.floor(y1-dy)))
    x2, y2 = max(0, min(width, math.ceil(x2+dx))), max(0, min(height, math.ceil(y2+dy)))
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None

def unrotate_box(box, width: int, height: int, angle: int):
    """Map half-open bbox from np.rot90(image, angle//90) to original."""
    x1, y1, x2, y2 = box
    if angle == 0:
        return tuple(box)
    if angle == 90:
        return (width-y2, x1, width-y1, x2)
    if angle == 180:
        return (width-x2, height-y2, width-x1, height-y1)
    if angle == 270:
        return (y1, height-x2, y2, height-x1)
    raise ValueError('Unsupported rotation')
