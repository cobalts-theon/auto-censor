"""Evaluate local labelled images/video frames; writes aggregate numbers only."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import json
import numpy as np
import cv2
from privacy_vision.cli.image import parser, config_from_args
from privacy_vision.pipeline import PrivacyPipeline
from privacy_vision.image_io import read_image


def mask(boxes, shape):
    from privacy_vision.processing.boxes import normalize_box
    result = np.zeros(shape[:2], bool)
    for box in boxes:
        b = normalize_box(box, shape[1], shape[0])
        if b is None or tuple(box) != b:
            raise ValueError('Invalid or out-of-bounds annotation/prediction bbox')
        x,y,r,bottom=b
        result[y:bottom,x:r]=True
    return result


def iou(a,b):
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    area=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/area if area>0 else 0.


def matched_counts(predictions, truth):
    used_p,used_t=set(),set()
    pairs=sorted(((iou(p.bbox,t['bbox']),i,j) for i,p in enumerate(predictions)
                  for j,t in enumerate(truth) if p.document_type==t['document_type']), reverse=True)
    for score,i,j in pairs:
        if score>=.5 and i not in used_p and j not in used_t:
            used_p.add(i); used_t.add(j)
    return {'tp':len(used_t),'fn':len(truth)-len(used_t),'fp':len(predictions)-len(used_p)}


def main():
    p=parser(benchmark=True)
    for action in p._actions:
        if action.dest=='input': action.required=False
    p.add_argument('--manifest',type=Path,required=True)
    args=p.parse_args()
    try:
        config=config_from_args(args)
        pipeline=PrivacyPipeline(config)
        manifest=json.loads(args.manifest.read_text())
        stats={k:{'tp':0,'fn':0,'fp':0} for k in ('detector','heuristic')}
        coverage={k:{'frames':0,'sensitive_pixels':0,'covered_pixels':0,'sensitive_regions':0,'exposed_regions':0}
                  for k in ('ordinary','blackout')}
        samples=[]
        for entry in manifest['samples']:
            path=args.manifest.parent / entry['path']
            if 'frame_index' in entry:
                cap=cv2.VideoCapture(str(path))
                try:
                    cap.set(cv2.CAP_PROP_POS_FRAMES,entry['frame_index'])
                    ok,image=cap.read()
                    if not ok: raise ValueError('Cannot read labelled video frame')
                finally: cap.release()
            else:
                image=read_image(path)
            truth=entry['documents']
            for t in truth: mask([t['bbox']],image.shape)
            result=pipeline.process(image)  # Independent frame evaluation; no temporal claims.
            samples.append(result.timings_ms['total'])
            for kind in stats:
                predictions=[r for r in result.document_regions if (r.source=='heuristic')==(kind=='heuristic')]
                # Heuristic type hints map to the public taxonomy.
                from dataclasses import replace
                types={'cccd':'identity_card','bhyt':'health_insurance_card','gplx':'driver_license'}
                predictions=[replace(r,document_type=types.get(r.document_type,r.document_type)) for r in predictions]
                counts=matched_counts(predictions,truth)
                for key,value in counts.items(): stats[kind][key]+=value
            sensitive=entry.get('sensitive_regions')
            group=coverage['blackout' if result.strict_blackout else 'ordinary']
            group['frames']+=1
            if sensitive is not None:
                actual=np.ones(image.shape[:2],bool) if result.strict_blackout else mask([d.bbox for d in result.detections],image.shape)
                target=mask([r['bbox'] for r in sensitive],image.shape)
                group['sensitive_pixels']+=int(target.sum())
                group['covered_pixels']+=int((target & actual).sum())
                group['sensitive_regions']+=len(sensitive)
                group['exposed_regions']+=sum(bool((mask([r['bbox']],image.shape)&~actual).any()) for r in sensitive)
        if not samples: raise ValueError('Empty evaluation manifest')
        for group in stats.values():
            total=group['tp']+group['fn']
            group['recall']=group['tp']/total if total else None
        for group in coverage.values():
            n=group['sensitive_pixels']
            group['coverage']=group['covered_pixels']/n if n else None
            group['exposed_pixel_ratio']=1-group['coverage'] if n else None
            n=group['sensitive_regions']
            group['exposed_region_ratio']=group['exposed_regions']/n if n else None
        print(json.dumps({'documents_iou_0_5':stats,'sensitive_mask_coverage':coverage,
            'pipeline_latency_ms':{'p50':float(np.percentile(samples,50)),'p95':float(np.percentile(samples,95))},
            'warning':'Independent frames; mask coverage does not establish blur/pixelate resistance or webcam FPS.'},indent=2))
    except Exception as exc:
        p.exit(1,f'Evaluation failed: {type(exc).__name__}; no source images exported.\n')

if __name__=='__main__': main()
