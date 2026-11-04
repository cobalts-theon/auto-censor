"""Short-lived optical-flow redaction, adapted from the main.py FaceTracker.

No inference is skipped. Tracks expire based on their last model confirmation.
"""
from dataclasses import dataclass, replace
import cv2
import numpy as np
from .boxes import normalize_box
from .documents import canonical
from ..schemas import Detection


def _key(d):
    if d.class_name == 'document_region':
        return ('document',)
    return (d.source,canonical(d.class_name))


def _iou(a,b):
    area=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    total=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area
    return area/total if total else 0.


def _association(a,b):
    score=_iou(a.bbox,b.bbox)
    if a.class_name != 'document_region':
        return score
    x,y=a.bbox,b.bbox
    old_area=(x[2]-x[0])*(x[3]-x[1])
    new_area=(y[2]-y[0])*(y[3]-y[1])
    if a.source != 'document' and b.source != 'document' and new_area>2.5*old_area:
        return 0.  # A sudden large proposal needs its own confirmation for heuristic only.
    intersection=max(0,min(x[2],y[2])-max(x[0],y[0]))*max(0,min(x[3],y[3])-max(x[1],y[1]))
    smaller=min((x[2]-x[0])*(x[3]-x[1]),(y[2]-y[0])*(y[3]-y[1]))
    base = max(score, .8*intersection/smaller) if smaller else score
    if base >= 0.2:
        return base
    # Center proximity matching: prevents breaking track when card moves or rotates quickly
    ca = ((x[0]+x[2])/2, (x[1]+x[3])/2)
    cb = ((y[0]+y[2])/2, (y[1]+y[3])/2)
    diag = ((x[2]-x[0])**2 + (x[3]-x[1])**2)**0.5
    dist = ((ca[0]-cb[0])**2 + (ca[1]-cb[1])**2)**0.5
    if diag > 0 and dist < 2.5 * diag:
        return max(0.2, 0.75 * (1.0 - dist / (2.5 * diag)))
    return base


@dataclass
class _Track:
    detection: Detection
    seen: float
    points: object = None
    state: str = 'detected'
    hits: int = 1
    filtered_box: tuple | None = None
    velocity: tuple[float, float] = (0., 0.)
    observed_box: tuple | None = None

    def __post_init__(self):
        self.observed_box = self.detection.bbox
        if self.filtered_box is None:
            self.filtered_box = tuple(map(float,self.detection.bbox))


class RegionTracker:
    def __init__(self, hold_seconds=.6, document_hold_seconds=None):
        self.hold_seconds = hold_seconds
        self.document_hold_seconds = document_hold_seconds
        self.tracks = []
        self.previous = None
        self.input_shape = None

    def reset(self):
        self.tracks = []
        self.previous = None
        self.input_shape = None

    def update(self,image,detections,now):
        # Bound optical-flow cost even for 720p/1080p camera input.
        h,w=image.shape[:2]
        scale=min(1.,640/max(h,w))
        gray=cv2.cvtColor(cv2.resize(image,(max(1,round(w*scale)),max(1,round(h*scale)))),cv2.COLOR_BGR2GRAY)
        sy,sx=gray.shape[0]/h,gray.shape[1]/w
        if self.previous is not None and (self.previous.shape != gray.shape or self.input_shape != (h,w)):
            self.tracks=[]; self.previous=None
        self.input_shape=(h,w)
        doc_hold = self.document_hold_seconds if self.document_hold_seconds is not None else min(.25, self.hold_seconds)
        self.tracks=[t for t in self.tracks if now-t.seen <= (
            doc_hold if t.detection.class_name=='document_region' else self.hold_seconds)]
        for t in self.tracks: t.state='held'
        active=[t for t in self.tracks if t.points is not None and (
            t.detection.source in ('face','plate','document') or t.detection.class_name=='document_region')]
        if self.previous is not None and active:
            points=np.concatenate([t.points for t in active])
            moved,status,_=cv2.calcOpticalFlowPyrLK(self.previous,gray,points,None,winSize=(31,31),maxLevel=4)
            if moved is not None and status is not None:
                back,valid_back,_=cv2.calcOpticalFlowPyrLK(gray,self.previous,moved,None,winSize=(31,31),maxLevel=4)
                if back is not None and valid_back is not None:
                    valid=(status.ravel()==1)&(valid_back.ravel()==1)
                    valid &= np.linalg.norm((back-points).reshape(-1,2),axis=1)<1.5
                    offset=0
                    for t in active:
                        end=offset+len(t.points); keep=valid[offset:end]
                        if keep.sum()>=max(4,len(keep)*.5):
                            src_pts = points[offset:end][keep]
                            dst_pts = moved[offset:end][keep]
                            delta=(dst_pts-src_pts).reshape(-1,2)
                            median=np.median(delta,axis=0)
                            coherent=np.median(np.linalg.norm(delta-median,axis=1))<=2.5
                            dx,dy=median/[sx,sy]
                            a,b,c,d=t.detection.bbox
                            box=normalize_box((a+dx,b+dy,c+dx,d+dy),w,h)
                            if box and coherent and abs(dx)<=.3*(c-a) and abs(dy)<=.3*(d-b):
                                t.detection=replace(t.detection,bbox=box); t.state='tracked'
                                f=t.filtered_box
                                t.filtered_box=(f[0]+dx,f[1]+dy,f[2]+dx,f[3]+dy)
                            elif t.detection.class_name == 'document_region' and keep.sum() >= 6:
                                # Tilt / rotation handling: points rotate around center, breaking translation coherence.
                                # Use estimateAffinePartial2D to track rotation and update bounding envelope.
                                M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts)
                                if M is not None and inliers is not None and inliers.sum() >= 4:
                                    corners = np.array([[[a*sx, b*sy], [c*sx, b*sy], [c*sx, d*sy], [a*sx, d*sy]]], dtype=np.float32)
                                    trans_corners = cv2.transform(corners, M)[0] / [sx, sy]
                                    rx1, ry1 = np.min(trans_corners, axis=0)
                                    rx2, ry2 = np.max(trans_corners, axis=0)
                                    rot_box = normalize_box((rx1, ry1, rx2, ry2), w, h)
                                    if rot_box:
                                        t.detection = replace(t.detection, bbox=rot_box)
                                        t.state = 'tracked'
                                        t.filtered_box = tuple(map(float, rot_box))
                        offset=end
        pairs=sorted(((_association(t.detection,d),ti,di)
            for ti,t in enumerate(self.tracks) for di,d in enumerate(detections)
            if _key(t.detection)==_key(d)),reverse=True)
        used_t,used_d=set(),set()
        for score,ti,di in pairs:
            if score<.2 or ti in used_t or di in used_d: continue
            t,d=self.tracks[ti],detections[di]
            old=t.filtered_box
            geometry_confirmed=True
            if d.class_name=='document_region':
                if d.source == 'document' or d.metadata.get('region_source') == 'detector':
                    # Real detector bounding box: follow promptly so it sticks firmly!
                    cx = .1 * (old[0] + old[2]) / 2 + .9 * (d.bbox[0] + d.bbox[2]) / 2
                    cy = .1 * (old[1] + old[3]) / 2 + .9 * (d.bbox[1] + d.bbox[3]) / 2
                    width = .2 * (old[2] - old[0]) + .8 * (d.bbox[2] - d.bbox[0])
                    height = .2 * (old[3] - old[1]) + .8 * (d.bbox[3] - d.bbox[1])
                    smooth = (cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2)
                else:
                    # Follow position promptly while damping field-union size jitter.
                    cx=.2*(old[0]+old[2])/2+.8*(d.bbox[0]+d.bbox[2])/2
                    cy=.2*(old[1]+old[3])/2+.8*(d.bbox[1]+d.bbox[3])/2
                    width=.85*(old[2]-old[0])+.15*(d.bbox[2]-d.bbox[0])
                    height=.85*(old[3]-old[1])+.15*(d.bbox[3]-d.bbox[1])
                    # A nested partial union carries little information about card
                    # position or size. Hold the established geometry for this pass.
                    raw_area=(d.bbox[2]-d.bbox[0])*(d.bbox[3]-d.bbox[1])
                    old_area=(old[2]-old[0])*(old[3]-old[1])
                    nested=old[0]<=d.bbox[0] and old[1]<=d.bbox[1] and old[2]>=d.bbox[2] and old[3]>=d.bbox[3]
                    if nested and raw_area < .6*old_area:
                        smooth=old
                        geometry_confirmed=False
                    else:
                        smooth=(cx-width/2,cy-height/2,cx+width/2,cy+height/2)
            else:
                smooth=tuple(.25*a+.75*b for a,b in zip(old,d.bbox))
            # Never feed rounded/coverage-expanded render bounds into filtering.
            t.filtered_box=tuple(smooth)
            # Include current detection in full; smoothing must not expose edges.
            box=normalize_box((min(smooth[0],d.bbox[0]),min(smooth[1],d.bbox[1]),
                               max(smooth[2],d.bbox[2]),max(smooth[3],d.bbox[3])),w,h)
            t.detection=replace(d,bbox=box)
            if geometry_confirmed:
                elapsed = now - t.seen
                if d.class_name == 'document_region' and elapsed > 0:
                    a, b = t.observed_box, d.bbox
                    t.velocity = ((b[0]+b[2]-a[0]-a[2])/(2*elapsed),
                                  (b[1]+b[3]-a[1]-a[3])/(2*elapsed))
                    t.observed_box = d.bbox
                t.seen=now; t.state='detected'; t.hits+=1
            else:
                t.state='partial'
            used_t.add(ti); used_d.add(di)
        self.tracks.extend(_Track(d,now) for i,d in enumerate(detections) if i not in used_d)
        for ti,t in enumerate(self.tracks):
            if ti not in used_t and t.hits<2 and t.state=='held':
                t.hits=0  # Two isolated proposals must not confirm a document.
        # When a newly confirmed larger region replaces an old overlapping
        # hypothesis, retire the old one instead of drawing both until timeout.
        confirmed=[t for t in self.tracks if t.detection.class_name=='document_region'
                   and t.hits>=2 and t.state=='detected']
        self.tracks=[t for t in self.tracks if not (
            t.detection.class_name=='document_region' and t.state=='held' and any(
                c is not t and c.detection.bbox[0]<=t.detection.bbox[0] and
                c.detection.bbox[1]<=t.detection.bbox[1] and
                c.detection.bbox[2]>=t.detection.bbox[2] and
                c.detection.bbox[3]>=t.detection.bbox[3] for c in confirmed))]
        result=[]
        for t in self.tracks:
            if t.detection.class_name=='document_region' and t.detection.source != 'document' and t.hits<2:
                # Fields are still redacted while a whole-document proposal
                # waits for a second model-supported observation.
                continue
            render_box = t.detection.bbox
            if t.detection.class_name == 'document_region' and t.state == 'held':
                if t.detection.source != 'document':
                    # Extrapolate only within a short bounded envelope for heuristic.
                    a,b,c,d = render_box
                    age = max(0., now-t.seen)
                    dx = float(np.clip(t.velocity[0]*age, -.2*(c-a), .2*(c-a)))
                    dy = float(np.clip(t.velocity[1]*age, -.2*(d-b), .2*(d-b)))
                    render_box = (min(a,a+dx)-abs(dx), min(b,b+dy)-abs(dy),
                                  max(c,c+dx)+abs(dx), max(d,d+dy)+abs(dy))
                else:
                    # Real detector: velocity-guided hold so rapid motions don't leave the redaction behind
                    a, b, c, d = t.filtered_box or t.detection.bbox
                    age = max(0., now - t.seen)
                    dx = float(np.clip(t.velocity[0] * age, -0.5 * (c - a), 0.5 * (c - a)))
                    dy = float(np.clip(t.velocity[1] * age, -0.5 * (d - b), 0.5 * (d - b)))
                    render_box = (a + dx, b + dy, c + dx, d + dy)
            box=normalize_box(render_box,w,h)
            if box is None: t.points=None; continue
            rendered=replace(t.detection,bbox=box)
            x1,y1,x2,y2=normalize_box((box[0]*sx,box[1]*sy,box[2]*sx,box[3]*sy),gray.shape[1],gray.shape[0])
            if t.state in ('detected', 'tracked'):
                if t.state == 'detected' or t.points is None or len(t.points) < 8:
                    points=(cv2.goodFeaturesToTrack(gray[y1:y2,x1:x2],maxCorners=24,qualityLevel=.01,minDistance=5)
                            if (t.detection.source in ('face','plate','document') or t.detection.class_name=='document_region') else None)
                    if points is not None: points+=np.float32([x1,y1])
                    t.points=points
            result.append(replace(rendered,metadata={**t.detection.metadata,
                'tracking_state':t.state,'seconds_since_detection':now-t.seen}))
        self.previous=gray
        return result
