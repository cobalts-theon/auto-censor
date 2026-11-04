"""Infer document regions from local field clusters; no full-frame fallback."""
from dataclasses import replace
import numpy as np
from .boxes import normalize_box
from ..schemas import Detection

HINTS = {'cccd','bhyt','gplx','qr','finger_print'}


def canonical(name: str) -> str:
    if name in ('id','id_'):
        return 'id'
    if name.endswith(('1','2')):
        return name[:-1]
    return name


def _union(boxes):
    return (min(b[0] for b in boxes),min(b[1] for b in boxes),
            max(b[2] for b in boxes),max(b[3] for b in boxes))


def _near(a,b):
    aw,ah=a[2]-a[0],a[3]-a[1]
    bw,bh=b[2]-b[0],b[3]-b[1]
    gx=max(0,max(a[0],b[0])-min(a[2],b[2]))
    gy=max(0,max(a[1],b[1])-min(a[3],b[3]))
    # Text rows can be separated vertically; QR/fingerprint may sit beside text.
    return gx <= .65*max(aw,bw) and gy <= max(3*max(ah,bh),.5*max(aw,bw))


def _compatible(d, cluster):
    singletons={'id','name','dob','gender','qr','finger_print','issue_date','expire_date'}
    for other in cluster:
        if not _near(d.bbox,other.bbox):
            return False
        if canonical(d.class_name)==canonical(other.class_name) and canonical(d.class_name) in singletons:
            a,b=d.bbox,other.bbox
            overlap=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
            smaller=min((a[2]-a[0])*(a[3]-a[1]),(b[2]-b[0])*(b[3]-b[1]))
            if overlap < .3*smaller:
                return False  # Two separate IDs/names are evidence of two cards.
    return True


def document_regions(image: np.ndarray, fields: list[Detection], margin=(.25, .30), min_fields=3) -> list[Detection]:
    anchors=[replace(d, bbox=box) for d in fields
             if d.source in ('card','cccd') and not d.metadata.get('derived_region')
             if (box := normalize_box(d.bbox, image.shape[1], image.shape[0])) is not None]
    h,w=image.shape[:2]
    # Complete-link clustering prevents a chain of adjacent fields bridging two
    # distant cards. Same class from two models is counted only once for evidence.
    clusters=[]
    # A stable semantic order avoids rebuilding different clusters when text
    # rows jitter a pixel left/right between frames.
    for d in sorted(anchors,key=lambda d:(canonical(d.class_name),d.bbox[0],d.bbox[1],d.source)):
        compatible=[c for c in clusters if _compatible(d,c)]
        if compatible:
            cluster=min(compatible,key=lambda c:sum(abs((x.bbox[0]+x.bbox[2])-(d.bbox[0]+d.bbox[2])) for x in c)/len(c))
            cluster.append(d)
        else:
            clusters.append([d])
    output=[]
    for cluster in clusters:
        names={canonical(d.class_name) for d in cluster}
        enough=len(names)>=min_fields or (len(names)>=2 and bool(names&HINTS))
        if not enough:
            for d in cluster:
                output.append(replace(d,class_id=-1,class_name='document_boundary_unresolved',
                    metadata={**d.metadata,'method':'fields_only','derived_region':True}))
            continue
        x1,y1,x2,y2=_union([d.bbox for d in cluster])
        dx,dy=margin[0]*(x2-x1),margin[1]*(y2-y1)
        box=normalize_box((x1-dx,y1-dy,x2+dx,y2+dy),w,h)
        if box is None or (box[2]-box[0])*(box[3]-box[1])>.75*w*h:
            output.extend(replace(d,class_id=-1,class_name='document_boundary_unresolved',
                metadata={**d.metadata,'method':'fields_only','derived_region':True}) for d in cluster)
            continue
        output.append(Detection('card',-1,'document_region',0.0,box,
            {'method':'field_cluster_union','derived_region':True,
             'field_count':len(names),'unique_fields':len(names),
             'region_source':'heuristic','confidence_kind':'unscored_heuristic',
             'supporting_detections': list(cluster)}))
    # Nested proposals do not add mask coverage, but would create independent
    # temporal tracks that appear/disappear at different times.
    regions=sorted((d for d in output if d.class_name=='document_region'),
                   key=lambda d:(d.bbox[2]-d.bbox[0])*(d.bbox[3]-d.bbox[1]),reverse=True)
    kept=[]
    for d in regions:
        if not any(k.bbox[0]<=d.bbox[0] and k.bbox[1]<=d.bbox[1] and
                   k.bbox[2]>=d.bbox[2] and k.bbox[3]>=d.bbox[3] and
                   all(_compatible(e, k.metadata['supporting_detections'])
                       for e in d.metadata['supporting_detections']) for k in kept):
            kept.append(d)
    return kept+[d for d in output if d.class_name!='document_region']
