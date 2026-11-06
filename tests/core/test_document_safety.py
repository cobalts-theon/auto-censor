from pathlib import Path
from unittest.mock import Mock
import numpy as np
import pytest
from privacy_vision.config import PipelineConfig, DetectorConfig
from privacy_vision.pipeline import PrivacyPipeline
from privacy_vision.schemas import Detection, DocumentRegion
from privacy_vision.processing.documents import document_regions
from privacy_vision.processing.tracking import RegionTracker

FRAME = np.full((200,400,3), 239, np.uint8)

def fields(dx=0):
    return [Detection('cccd', i, name, .9, (dx+20,20+i*20,dx+120,30+i*20))
            for i,name in enumerate(('id','name','dob'))]

def pipeline(rows=(), **kwargs):
    config = PipelineConfig([DetectorConfig('cccd',Path('unused'))], margin=0, **kwargs)
    return PrivacyPipeline(config, [Mock(source='cccd', detect=Mock(return_value=list(rows)))])


def test_full_keeps_all_fields_and_unscored_regions():
    rows = fields()+[Detection('cccd',0,'id',.9,(350,10,399,20))]
    result = pipeline(rows, document_redaction_mode='full').process(FRAME)
    assert len(result.document_regions)==1
    assert result.document_regions[0].confidence is None
    assert result.coverage_status=='coverage_unverified'
    for d in rows:
        x,y,r,b=d.bbox
        assert not result.image[y:b,x:r].any()
    assert result.to_metadata()['document_regions'][0]['confidence'] is None
    import json
    json.dumps(result.to_metadata())


@pytest.mark.parametrize('rows', [[], fields()[:1], fields()])
@pytest.mark.parametrize('mode', ['black','blur','pixelate'])
def test_strict_heuristic_never_claims_whole_paper_safe(rows, mode):
    r=pipeline(rows, whole_documents=True, strict_privacy_mode=True, mode=mode).process(FRAME)
    assert r.strict_blackout and not r.image.any()
    assert r.coverage_status=='coverage_unverified'


def test_fields_default_backward_compatible_and_invalid_boxes():
    invalid=[Detection('cccd',0,'id',.9,(float('nan'),1,8,9)),
             Detection('cccd',0,'name',.9,(40,40,20,20))]
    assert document_regions(FRAME, invalid)==[]
    with pytest.raises(RuntimeError): pipeline(invalid).process(FRAME)
    r=pipeline().process(FRAME)
    np.testing.assert_array_equal(r.image,FRAME)
    assert r.document_regions==[]


def test_redaction_error_never_writes_file(tmp_path,monkeypatch):
    from privacy_vision.image_io import write_image
    import privacy_vision.pipeline as module
    src,out=tmp_path/'source.png',tmp_path/'out.png'
    write_image(src,FRAME)
    monkeypatch.setattr(module,'redact',Mock(side_effect=RuntimeError('private')))
    with pytest.raises(RuntimeError): pipeline(fields()).process_file(src,out)
    assert not out.exists()


def test_invalid_redaction_output_rejected(monkeypatch):
    import privacy_vision.pipeline as module
    monkeypatch.setattr(module,'redact',lambda *a: np.zeros((1,1),np.float32))
    with pytest.raises(ValueError): pipeline(fields()).process(FRAME)


@pytest.mark.parametrize('strategy',['detector','hybrid'])
def test_missing_document_checkpoint(strategy):
    with pytest.raises(ValueError,match='checkpoint'):
        PipelineConfig(document_detection_strategy=strategy)


def test_hybrid_keeps_field_outside_document_and_tracks_provenance(tmp_path):
    checkpoint=tmp_path/'document.pt'; checkpoint.touch()
    cfg=PipelineConfig([DetectorConfig('cccd',Path('unused'))], margin=0,
        document_redaction_mode='full', document_detection_strategy='hybrid',
        document_detector_config=DetectorConfig('document',checkpoint))
    rows=fields()+[Detection('cccd',0,'qr',.9,(340,150,370,180))]
    doc=Detection('document',0,'identity_card',.8,(10,10,140,100))
    p=PrivacyPipeline(cfg,[Mock(source='cccd',detect=Mock(return_value=rows))],
        document_detector=Mock(source='document',detect=Mock(return_value=[doc])))
    r=p.process(FRAME)
    assert r.document_regions[0].source=='hybrid'
    assert r.document_regions[0].confidence==.8
    assert not r.image[10:100,10:140].any()
    assert not r.image[150:180,340:370].any()
    assert 'fields_without_detector_region' in r.coverage_reasons
    cfg.strict_privacy_mode=True
    assert p.process(FRAME).strict_blackout


def test_strict_detector_evidence_still_unverified(tmp_path):
    checkpoint=tmp_path/'doc.pt'; checkpoint.touch()
    cfg=PipelineConfig([], document_redaction_mode='full', document_detection_strategy='detector',
        strict_privacy_mode=True, document_detector_config=DetectorConfig('document',checkpoint))
    doc=Detection('document',0,'identity_card',.8,(10,10,140,100))
    p=PrivacyPipeline(cfg,[],Mock(source='document',detect=Mock(return_value=[doc])))
    r=p.process(FRAME)
    assert not r.strict_blackout and r.coverage_status=='coverage_unverified'
    assert not r.image[10:100,10:140].any()


def test_tracker_prediction_expiry_and_explicit_reset():
    def doc(x): return Detection('card',-1,'document_region',0,(x,30,x+100,100))
    t=RegionTracker()
    t.update(FRAME,[doc(20)],0)
    t.update(FRAME,[doc(25)],.05)
    held=t.update(FRAME,[],.1)
    assert held and held[0].bbox[2]>=125
    assert held[0].metadata['tracking_state']=='held'
    assert t.update(FRAME,[],.4)==[]
    t.update(FRAME,[doc(20)],.5)
    t.reset()
    assert t.update(FRAME,[],.55)==[]


def test_alias_and_configuration_validation():
    assert PipelineConfig(whole_documents=True).document_redaction_mode=='full'
    assert PipelineConfig(document_redaction_mode='full').whole_documents
    with pytest.raises(ValueError): PipelineConfig(whole_documents=True,document_redaction_mode='fields')
    with pytest.raises(ValueError): PipelineConfig(document_margin=(.8,.2))


def test_document_adapter_rejects_field_checkpoint_taxonomy(monkeypatch):
    from privacy_vision.detectors.document import DocumentDetector
    from privacy_vision.detectors.yolo import YoloDetector
    def init(self,*args): self.model=Mock(names={0:'cccd',1:'name'})
    monkeypatch.setattr(YoloDetector,'__init__',init)
    with pytest.raises(ValueError,match='taxonomy'):
        DocumentDetector(DetectorConfig('document',Path('unused')),'cpu')


def test_cli_old_and_new_flags_share_configuration():
    from privacy_vision.cli.image import parser,config_from_args
    args=parser().parse_args(['--input','local.png','--output','out.png','--whole-documents',
                             '--imgsz-card','800','--strict-privacy-mode'])
    config=config_from_args(args)
    assert config.document_redaction_mode=='full' and config.strict_privacy_mode
    assert next(c.imgsz for c in config.detectors if c.source=='card')==800
    assert next(c.imgsz for c in config.detectors if c.source=='plate')==640


def test_flat_wall_rejected_while_content_document_accepted(tmp_path):
    import cv2
    from privacy_vision.pipeline import _is_flat_background

    # 1. White wall with simulated sensor noise
    np.random.seed(42)
    wall = np.clip(np.random.normal(215, 5.0, (200, 250, 3)), 0, 255).astype(np.uint8)
    assert _is_flat_background(wall) is True

    # 2. Document with structured text / lines
    doc = np.full((200, 250, 3), 245, dtype=np.uint8)
    cv2.putText(doc, 'CAN CUOC CONG DAN', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(doc, 'So: 0123456789', (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.rectangle(doc, (160, 50), (220, 130), (50, 50, 50), -1)  # Avatar box
    assert _is_flat_background(doc) is False

    # 3. Head in a corner against > 55% white wall
    head_wall = np.clip(np.random.normal(215, 5.0, (200, 250, 3)), 0, 255).astype(np.uint8)
    cv2.circle(head_wall, (50, 50), 35, (40, 50, 70), -1)  # Head occupying top-left corner only
    assert _is_flat_background(head_wall) is True

